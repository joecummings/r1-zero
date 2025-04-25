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

from tensordict import lazy_stack, TensorDictBase
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.collectors import (
    WeightUpdateSenderBase as LocalWeightUpdaterBase,
    WeightUpdateReceiverBase as RemoteWeightUpdaterBase,
    SyncDataCollector,
)

from torchtune.dev.grpo.envs import LLMEnv

from torchtune import config, utils
from torchtune.dev.rl.datatypes import Trajectory
from torchtune.dev.rl.utils import stateless_init_process_group
from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker

log = utils.get_logger()


# not decorating with @ray.remote here because num_gpus should vary based on tp_size
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
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self.rollout_queue = queue
        self.worker_id = worker_id
        self._is_collector_zero = self.worker_id == 0

        self.batch_size = self.cfg.vllm.batch_size

        # TDOO: tp_size + distributed_executor_backend needs to be fixed and is coming in a follow up
        self.inference_server = LLM(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            enable_chunked_prefill=True,
            dtype="bfloat16",
            worker_cls=vLLMWorkerWrapper,
            # tensor_parallel_size=vllm_tp_size,
            # gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            # FIXME: Need to use placement_groups in order to use distributed_executor_backend="ray"
            # distributed_executor_backend="ray",
        )

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
        """
        This is a helper that should be deleted once the TensorClass stuff has been figured out.
        """
        data = data.squeeze()
        query_responses = torch.cat([data["tokens"], data["tokens_response"]], dim=-1)
        prompt_tokens = data["tokens"]
        response_tokens = data["tokens_response"]
        logprobs = data["log_probs"]
        query_response_padding_masks = torch.ne(query_responses, self._tokenizer.pad_id)
        answers = data["answers"]
        if hasattr(
            self.inference_server.llm_engine.model_executor.driver_worker.worker,
            "policy_version",
        ):
            policy_version = (
                self.inference_server.llm_engine.model_executor.driver_worker.worker.policy_version.item()
            )
        else:
            policy_version = 0

        response_padding_masks = torch.eq(response_tokens, self._tokenizer.pad_id)
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        del response_padding_masks

        postprocessed_results = Trajectory(
            query_responses=query_responses,
            responses=response_tokens,
            logprobs=logprobs,
            ref_logprobs=None,
            query_response_padding_masks=query_response_padding_masks,
            seq_lens=seq_lens,
            answers=answers,
            policy_version=policy_version,
            rewards=None,
            advantages=None,
            successes=None,
            reward_metadata=None,
        )

        total_generated_tokens = seq_lens.sum().item()
        return postprocessed_results, total_generated_tokens

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
        else:
            self._metric_logger = None

    def _log_metrics(
        self,
        step_idx,
        time_total_rollout,
        time_generate,
        total_generated_tokens,
        # full_queue_data_discard,
        gpu_memory,
    ):
        """Log metrics for the vLLMRolloutActor, only on actor zero."""
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
                print(
                    f"{self.worker_id} about to update weights, {self.remote_weight_updater}"
                )
                self.remote_weight_updater.update_weights.remote(
                    weights=None, worker_ids=self.worker_id
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
                try:
                    self.rollout_queue.put_nowait(postprocessed_results)
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
            # self._log_metrics(
            #     step_idx=idx,
            #     time_total_rollout=time_total_rollout,
            #     time_generate=time_generate,
            #     total_generated_tokens=total_generated_tokens,
            #     # full_queue_data_discard=full_queue_data_discard,
            #     gpu_memory=gpu_memory,
            # )

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
        # Not importing here and doing these imports globally will cause vLLM worker
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
            assert False, "Haven't handled dataloader_state_dict yet"
            dataloader.load_state_dict(dataloader_state_dict)
            # B/c we currently only save at epoch boundaries, if we cut the previous epoch short
            # we need to force the dataloader to finish the last iteration before it's actually used
            list(dataloader)
        return dataloader


class vLLMWorkerWrapper(Worker):
    """vLLM Rollout Model worker for Ray."""

    def __init__(self, *args, **kwargs):
        import os

        print(f"visible devices {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)

    def init_weight_update_group(self, master_address, master_port, rank, world_size):

        # FIXME: Forgot why I changed rank_offset arg to rank
        # but likely need to uncomment this for the >1 vllm worker case
        # from vllm.distributed.parallel_state import get_world_group
        # rank = get_world_group().rank + rank_offset

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
