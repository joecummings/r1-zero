import time
from functools import partial
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed
import torchtune.training as training
import vllm

from omegaconf import DictConfig, ListConfig

from ray.util.queue import Full as QueueFull

from tensordict import lazy_stack, NonTensorStack, TensorDictBase
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.collectors import (
    LocalWeightUpdaterBase,
    RemoteWeightUpdaterBase,
    SyncDataCollector,
)

from collections import defaultdict
from torchrl.envs import LLMEnv
from torchtune import config, utils
from torchtune.dev.rl.utils import stateless_init_process_group
from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker
from torchtune.dev.rl.datatypes.trajectory import UnscoredTrajectory, GroupedUnscoredTrajectories
log = utils.get_logger()


class SyncLLMCollector(SyncDataCollector):
    """A simplified version of SyncDataCollector for LLM inference."""

    def __init__(
        self,
        cfg,
        llm,
        policy,
        queue,
        worker_id,
        *,
        dialog_turns_per_batch: int = -1,
        # -1 is never ending (until shutdown)
        total_dialog_turns: int = -1,
        async_envs: bool = False,
        reset_at_each_iter: bool = False,
        local_weight_updater: LocalWeightUpdaterBase | None = None,
        remote_weight_updater: RemoteWeightUpdaterBase | None = None,
    ):
        if async_envs:
            raise NotImplementedError

        self.cfg = cfg
        self.rollout_queue = queue
        self.worker_id = worker_id
        self._is_collector_zero = self.worker_id == 0

        self.tp_size = self.cfg.vllm.tp_size
        self.batch_size = self.cfg.vllm.batch_size
        self._sequence_counter = 0  # Used to assign unique sequence IDs to each sample

        self.inference_server = LLM(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            worker_cls=VLLMWorkerWrapper,
            tensor_parallel_size=self.tp_size,
        )

        # local import below LLM call to avoid vLLM no CUDA GPUs available error
        from torchtune import config

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        policy_kwargs = {
            "generate_kwargs": dict(
                n=1,
                max_tokens=self.cfg.max_generated_tokens,
                temperature=self.cfg.temperature,
            ),
            "pad_output": True,
            "padding_value": self._tokenizer.pad_id,
        }
        self.policy_kwargs = policy_kwargs

        collate_name = self.cfg.get(
            "collate_fn", "torchtune.dev.grpo.data.padded_collate_rl"
        )
        dataloader = self._setup_data(
            self.cfg.dataset,
            self.cfg.get("shuffle", True),
            self.batch_size,
            collate_name,
            dataloader_state_dict=None,
        )

        # local import below LLM call to avoid vLLM no CUDA GPUs available error
        from torchrl.envs import LLMEnv

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            tokenizer=None,
            str2str=True,
            batch_size=self.batch_size,
            repeats=self.cfg.grpo_samples,
        )

        super().__init__(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=dialog_turns_per_batch,
            total_frames=total_dialog_turns,
            local_weight_updater=local_weight_updater,
            remote_weight_updater=remote_weight_updater,
            reset_at_each_iter=reset_at_each_iter,
            use_buffers=False,
            # This argument allows a non-TensorDictModule policy to be assumed
            # to be compatible with the collector
            trust_policy=True,
        )

        log.info("done init LLMCollector")

    @property
    def remote_weight_updater(self) -> RemoteWeightUpdaterBase:
        return self._remote_weight_updater

    @remote_weight_updater.setter
    def remote_weight_updater(self, value: RemoteWeightUpdaterBase | None):
        self._remote_weight_updater = value

    def _postprocess_for_queue(self, data):
        """Postprocesses vLLM output into grouped trajectories for the rollout queue.

        It does it by, extracting data from the batch, creating unique identifiers, creating trajectories
        and grouping them by grpo_samples.

        Args:
            data: TensorDict with vLLM output (tokens, log_probs, etc.).

        Returns:
            tuple: (grouped_trajectories, total_generated_tokens)
                - grouped_trajectories: List of GroupedUnscoredTrajectories enqueued.
                - total_generated_tokens: Total number of response tokens generated.

        """
        # local import to avoid vLLM no CUDA GPUs available error
        from torchtune import training

        def get_policy_version() -> int:
            worker = self.inference_server.llm_engine.model_executor.driver_worker.worker
            return getattr(worker, "policy_version", -1).item() if hasattr(worker, "policy_version") else -1
        
        batch_size = data['tokens'].shape[0]
        sequence_ids = [f"worker{self.worker_id}_{self._sequence_counter + i}" for i in range(batch_size)]
        self._sequence_counter += batch_size

        data = data.squeeze()
        
        # Data extraction and unpadding
        prompt_tokens_padded = data['tokens'].cpu()              # [B, longest_prompt]
        response_tokens_padded = data['tokens_response'].cpu()   # [B, longest_response]
        logprobs_padded = data['log_probs'].cpu()                # [B, longest_response]
        answers = data['answers']                                # List[str], len B
        policy_version = get_policy_version()

        trajectories = defaultdict(list)
        for i in range(batch_size):
            # extract data
            prompt_tokens = prompt_tokens_padded[i]
            response_tokens = response_tokens_padded[i]
            logprobs = logprobs_padded[i]
            answer = answers[i]
            
            # remove padding tokens
            prompt_mask = prompt_tokens != self._tokenizer.pad_id
            response_mask = response_tokens != self._tokenizer.pad_id
            prompt_len = prompt_mask.sum()
            response_len = response_mask.sum()
            
            prompt_tokens = prompt_tokens[prompt_mask]
            response_tokens = response_tokens[response_mask]
            logprobs = logprobs[response_mask]
            
            # group id is used to calculate advantages
            group_idx = i // self.cfg.grpo_samples
            group_id = f"{self.worker_id}_{self._sequence_counter}_{group_idx}"
            
            # create trajectory
            traj = UnscoredTrajectory(
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                prompt_len=prompt_len,
                response_len=response_len,
                logprobs=logprobs,
                answer=answer,
                sequence_id=sequence_ids[i],
                policy_version=policy_version,
                group_id=group_id
            )

            # group trajectories by group_id
            trajectories[traj.group_id].append(traj)

        
        # create GroupedUnscoredTrajectories for each group
        grouped_trajectories = []
        for group_id, traj_list in sorted(trajectories.items()):
            total_tokens = sum(traj.prompt_len + traj.response_len for traj in traj_list)
            grouped_traj = GroupedUnscoredTrajectories(
                trajectories=traj_list,
                group_id=group_id,
                total_tokens=total_tokens
            )
            grouped_trajectories.append(grouped_traj)

        total_generated_tokens = sum(traj.response_len for traj in trajectories)
        return grouped_trajectories, total_generated_tokens

    def update_policy_weights_(
        self,
        policy_weights: TensorDictBase | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        **kwargs,
    ) -> None:
        self.local_weight_updater(policy_weights, **kwargs)

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        if self._is_collector_zero:
            self._metric_logger = logger

    def _log_metrics(
        self,
        step_idx,
        time_total_rollout,
        time_generate,
        total_generated_tokens,
        # full_queue_data_discard,
        gpu_memory,
    ):
        """Log metrics for the LLMCollector, only on collector zero."""
        if not self._is_collector_zero:
            return

        pct_time_model_running = (
            (time_generate / time_total_rollout) * 100 if time_total_rollout > 0 else 0
        )
        tokens_per_second = (
            total_generated_tokens / time_generate if time_generate > 0 else 0
        )
        div_gib = 1024**3

        log_dict = {
            "vllm_actor_performance/total_rollout_time (s)": time_total_rollout,
            "vllm_actor_performance/pct_time_model_running (%)": pct_time_model_running,
            "vllm_actor_performance/tokens_per_second": tokens_per_second,
            "vllm_actor_performance/gpu_memory_peak_allocated (GiB)": gpu_memory[
                "allocated"
            ]
            / div_gib,
            "vllm_actor_performance/gpu_memory_peak_reserved (GiB)": gpu_memory[
                "reserved"
            ]
            / div_gib,
            "vllm_actor_performance/gpu_memory_peak_active (GiB)": gpu_memory["active"]
            / div_gib,
            # "queues/vllm_full_queue_data_discard": full_queue_data_discard,
            "queues/rollout_queue_size": self.rollout_queue.qsize(),
        }

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def run(self):
        num_steps = (self.cfg.num_steps // self.cfg.vllm.num_workers) + 1
        for i in range(num_steps):
            self.rollout(i)
            if i % self.cfg.vllm.steps_before_sync == 0:
                log.info(f"{self.worker_id} about to update weights")
                ray.get(
                    self.remote_weight_updater.update_weights.remote(
                        weights=None, worker_ids=self.worker_id
                    )
                )

    def rollout(self, idx) -> TensorDictBase:
        if self.reset_at_each_iter or self._shuttle is None:
            data = self.env.reset()
        else:
            data = self._shuttle

        trajectories = []
        collected_frames = 0
        time_generate = 0
        time_step_start = time.perf_counter()

        while collected_frames < self.frames_per_batch:
            policy_input = data
            env_input, generation_time = self.policy(
                policy_input,
                self.inference_server,
                **self.policy_kwargs,
            )
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            time_generate += generation_time

            # carry over collector data without messing up devices
            collector_data = self._shuttle.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            self._shuttle.set("collector", collector_data)
            self._update_traj_ids(env_output)
            data = self._shuttle
            trajectories.append(data)
            collected_frames += data.numel()

        data = lazy_stack(trajectories, -1)

        if self.rollout_queue is not None:
            assert self.replay_buffer is None
            postprocessed_results, total_generated_tokens = self._postprocess_for_queue(
                data
            )

            while True:
                for grouped_traj in postprocessed_results:
                    try:
                        self.rollout_queue.put_nowait(grouped_traj, block=True, timeout=1.0)
                        break
                    except QueueFull:
                        self.rollout_queue.get()  # Remove the oldest item to make space
                        log.warn("rollout queue full. Discarding data.")

        if self._is_collector_zero:
            # End timing the rollout step
            time_total_rollout = time.perf_counter() - time_step_start

            # TODO: training.get_memory_stats() crashes vLLM
            # Log metrics
            gpu_memory = {
                "allocated": torch.cuda.max_memory_allocated(device="cuda:0"),
                "reserved": torch.cuda.max_memory_reserved(device="cuda:0"),
                "active": torch.cuda.memory_stats(device="cuda:0").get(
                    "active_bytes.all.peak", 0
                ),
            }
            time_total_rollout = time.perf_counter() - time_step_start
            self._log_metrics(
                step_idx=idx,
                time_total_rollout=time_total_rollout,
                time_generate=time_generate,
                total_generated_tokens=total_generated_tokens,
                # full_queue_data_discard=full_queue_data_discard,
                gpu_memory=gpu_memory,
            )

        return data

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        # Not importing here and doing these imports globally will cause VLLM worker
        # to have no cuda devices during cuda lazy init for some reason?? Even when
        # this method is not actually called...
        from torchtune import config
        from torchtune.config._utils import _get_component_from_path
        from torchtune.datasets import ConcatDataset

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        collate_fn = _get_component_from_path(collate_fn)
        sampler = StatefulDistributedSampler(
            ds,
            # FIXME: hardcoding num_replicas and rank for now
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            # FIXME: set seed?
            # seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            raise AssertionError("Haven't handled dataloader_state_dict yet")
            dataloader.load_state_dict(dataloader_state_dict)
            # B/c we currently only save at epoch boundaries, if we cut the previous epoch short
            # we need to force the dataloader to finish the last iteration before it's actually used
            list(dataloader)
        return dataloader


class VLLMWorkerWrapper(Worker):
    """
    vLLM worker for Ray.

    vLLMParameterServer will always take rank 0 in the stateless process group
    initialized by this worker. And the tp ranks associated with the LLM class
    will be in the range [1, tp_size].
    """

    def __init__(self, *args, **kwargs):
        import os

        print(f"visible devices {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset

        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

        self.version = torch.tensor([0], device="cuda")

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # src=0 because fsdp worker 0 has been assigned as "0" in this process group
        self._model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_policy_version(self):
        self._model_update_group.broadcast(
            self.version, src=0, stream=torch.cuda.current_stream()
        )
        self.policy_version = self.version
        torch.cuda.synchronize()
