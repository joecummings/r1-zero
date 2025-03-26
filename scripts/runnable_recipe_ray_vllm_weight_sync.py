#!/usr/bin/env python3
"""
README!! What's going on in this script?

At a high level, this script is grpo_full_finetune_distributed.py but it
1. Uses vLLM for generation instead of torchtune generate
2. Uses ray for orchestrating data parallel actors and vllm actors + unsharded ref actor
3. The dataloader is now owned by the vllm worker (rather than 1 dataloader per FSDP worker)
4. Uses a ray.util.Queue (this wraps a remote actor with a queue) as a "replay buffer"
   for the vllm workers to put their generated tokens into, and the FSDP workers to get them from
5. Of the items in GRPOTrajectory
        a. query_responses: vllm worker computes and puts into queue
        b. logprobs vllm worker puts into queue
        c. ref_logprobs: computed by unsharded RefActor which contains torchtune model
        d. rewards: fsdp worker computes
        e. sucecesses: fsdp worker computes
        f. advantages: fsdp worker computes
        g. masks: fsdp worker computes
        h. position_ids: fsdp worker computes
6. In config, we set ``steps_before_sync``. After ``steps_before_sync * num_data_parallel_worker`` steps
   the vllm worker will "sleep" and spin in a while loop until the FSDP workers are done syncing their weights
7. Weight sync currently blocks the train loop and is done by each fsdp workers .full_tensor (allgather) on each DTensor,
   calling tune_to_hf and then rank 0 broadcasting (+ also calling the vllm collective rpc to make it also issues
   the broadcast and then load weights call)

With this script, I can observe successes + rewards increasing over training steps, which is
a good sign. (See screenshot in Cabernet sprint notes.) But there are several issues with this script:
1. Peak memory usage for the fsdp worker is significantly higher than the original recipe in a fresh conda environmnet.
   This could be because
    a. I turned compile off (it seems to be broken with torch 2.5.1 that vllm requires)
    b. [FIXED] I turned activation checkpointing off (there wasn't an explcit reason for this it just got omitted accidentally)
2. I have an assert that num_vllm_workers == 1 and vllm_tp_size == 1 for now. There's no real reason for this,
   I expect the code to mostly generalize, just need to go over parts where I might have hardcoded this assumption of 1
   vllm worker.
3. Epochs is definitely not handled correctly right now :P
4. There's many FIXMEs peppered through this script

The run command is

    python runnable_recipe_ray_vllm_weight_sync.py --config ../recipes/configs/dev/qwen3B_full_grpo.yaml


"""

import functools
import os
import time
from functools import partial
from logging import log
from re import S
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest import result
from warnings import warn

from tensordict.tensorclass import NonTensorStack
from torchrl.modules import vLLMWrapper

import ray
import torch
import torch.nn as nn
import torchtune
import torchtune.training as training

from omegaconf import DictConfig, ListConfig
from ray.util.placement_group import placement_group

from ray.util.queue import Full as QueueFull, Queue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tensordict import is_tensorclass, pad_sequence, TensorClass, TensorDict
from torch.optim import Optimizer

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.dev.grpo.types import GRPOStats, GRPOTrajectory
from torchtune.models import qwen2_5
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

# from torchtune.rlhf import Trajectory

from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.lr_schedulers import get_lr

from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

log = utils.get_logger("DEBUG")
import torch.nn.functional as F


def pad_and_stack(tensors, pad_value=0):
    """
    Pads a list of tensors to the same shape and stacks them.

    Args:
        tensors (list of torch.Tensor): List of tensors to pad and stack.
        pad_value (scalar): The value to pad with (default is 0).

    Returns:
        torch.Tensor: A tensor with shape (N, *max_dims) where each tensor is padded to max_dims.
    """
    # Assume all tensors have the same number of dimensions.
    num_dims = tensors[0].dim()

    # Compute the maximum size for each dimension.
    max_shape = [max(t.size(d) for t in tensors) for d in range(num_dims)]

    padded_tensors = []
    for tensor in tensors:
        # Build padding amounts for each dimension (pad on the right only).
        # F.pad expects the padding tuple in reverse order: (pad_left_last_dim, pad_right_last_dim, ...).
        pad = []
        for dim in range(num_dims - 1, -1, -1):
            pad_amount = max_shape[dim] - tensor.size(dim)
            pad.extend([0, pad_amount])
        padded = F.pad(tensor, pad, value=pad_value)
        padded_tensors.append(padded)

    return torch.stack(padded_tensors)


class Trajectory(TensorClass["nocast"]):
    query_responses: torch.Tensor
    responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    query_response_padding_masks: torch.Tensor
    seq_lens: torch.Tensor
    answers: torch.Tensor
    policy_version: int


class PaddedTrajectoryBatch(TensorClass["nocast"]):
    """
    These are only single-turn trajectories (prompt + response).
    TODO design together what a multi-turn interaction looks like.
    Not needed at this stage.
    """

    policy_model_id: int  # Fingerprints the policy model who wrote the response
    prompts_str: str  # size is [batch]
    responses_str: str  # size is [batch, grpo_size]

    prompt_token_ids: torch.LongTensor  # [batch, max_prompt_len]
    response_token_ids: torch.LongTensor  # [batch, grpo_size, max_response_len]

    logprobs: torch.Tensor  # [batch, grpo_size, max_response_len]
    ref_logprobs: torch.Tensor  # [batch, grpo_size, max_response_len]


class PackedTrajectoryBatch(TensorClass["nocast"]):
    """
    These are only single-turn trajectories (prompt + response).
    TODO design together what a multi-turn interaction looks like.
    Not needed at this stage.
    """

    policy_model_id: int  # Fingerprints the policy model who wrote the response
    prompts_str: str
    responses_str: str

    # TODO fix these later
    # Legenda of sizes:
    # B = batch size
    # P[i] = prompt len for element i in batch
    # R[i] = response len for element i in batch

    packed_token_ids: torch.LongTensor  # sum(P[i] + R[i] for i in range(B))
    packed_sequence_ids: (
        torch.LongTensor
    )  # size is same as packed_token_ids. Each value indicates what sequence they belong to, eg [1, 1, 1, 2, 2, 3, 3, 3, 3]
    packed_loss_mask: (
        torch.LongTensor
    )  # size is the same as packed_token_ids. This mask disables loss for prompt/system tokens.

    # TODO figure this out later. I assume we need to match packed_token_ids lens and have zero-padding for prompt tokens
    packed_logprobs: torch.Tensor  # size is same as packed_token_ids. TODO check
    ref_logprobs: torch.Tensor  # size is same as packed_token_ids. TODO check


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


@ray.remote(num_cpus=8, num_gpus=1)
class RefActor:
    def __init__(self, *args, **kwargs):
        assert "rollout_queue" in kwargs, "Must pass queue to vLLMRefActor"
        assert "replay_buffer" in kwargs, "Must pass replay_buffer to vLLMRefActor"
        assert "cfg" in kwargs, "Must pass cfg to vLLMRefActor"

        self.actor_id = kwargs.pop("actor_id", -1)
        self._is_actor_zero = self.actor_id == 0

        self.cfg = kwargs.pop("cfg")
        self.rollout_queue = kwargs.pop("rollout_queue")
        self.replay_buffer = kwargs.pop("replay_buffer")
        self._device = utils.get_device(device=self.cfg.device)
        self._dtype = training.get_dtype(self.cfg.dtype, device=self._device)
        ref_checkpoint_dict = self.load_ref_checkpoint(
            cfg_ref_checkpointer=self.cfg.ref_checkpointer
        )
        self._ref_model = self._setup_model(
            self.cfg.model, ref_checkpoint_dict[training.MODEL_KEY]
        )
        self._temperature = self.cfg.temperature

        self.metric_logger = None  # Placeholder for the logger

        device_type = self.cfg.device
        self._log_peak_memory_stats = self.cfg.get("log_peak_memory_stats", True)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        if self._is_actor_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        if self._is_actor_zero:
            print(f"setting metric logger {logger} for actor id", self.actor_id)
            self._metric_logger = logger

    def load_ref_checkpoint(self, cfg_ref_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the reference checkpoint state from file and validate.
        """
        self._ref_checkpointer = config.instantiate(
            cfg_ref_checkpointer, resume_from_checkpoint=False
        )

        ref_checkpoint_dict = self._ref_checkpointer.load_checkpoint()

        return ref_checkpoint_dict

    def _setup_model(self, cfg_model, ref_model_state_dict):
        from torchtune.training import disable_dropout

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            ref_model = config.instantiate(cfg_model)

        with training.set_default_dtype(self._dtype), self._device:
            for m in ref_model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        for k, v in ref_model_state_dict.items():
            ref_model_state_dict[k] = v.to(self._device)

        ref_model.load_state_dict(ref_model_state_dict, assign=True, strict=True)

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(ref_model)

        disable_dropout(ref_model)
        print("done setting up ref model")
        return ref_model

    def _log_metrics(
        self,
        step_idx,
        time_total_ref_step,
        time_model_running,
        time_waiting_buffer,
        # full_queue_data_discard,
        # rollout_queue_length,
    ):
        """Log metrics for the RefActor, only on actor zero."""
        if not self._is_actor_zero:
            return

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"ref_actor_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        pct_time_model_running = (
            (time_model_running / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )
        pct_time_waiting_buffer = (
            (time_waiting_buffer / time_total_ref_step) * 100
            if time_total_ref_step > 0
            else 0
        )

        log_dict.update(
            {
                "ref_actor_performance/time_total_ref_step (s)": time_total_ref_step,
                "ref_actor_performance/time_model_running (s)": time_model_running,
                "ref_actor_performance/pct_time_model_running (%)": pct_time_model_running,
                "ref_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "ref_actor_performance/pct_time_waiting_buffer (%)": pct_time_waiting_buffer,
                # "queues/ref_actor_full_queue_data_discard": full_queue_data_discard,
                # "queues/rollout_queue_length": rollout_queue_length,
            }
        )

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def run(self):
        import time

        print("running ref actor")
        idx = 0
        while True:
            print(f"{idx=}")
            # FIXME: what should be the shutdown condition for this worker?
            if idx == 400:
                break

            # Start measuring total step time
            time_step_start = time.perf_counter()
            trajectory = None
            if self._is_actor_zero:
                rollout_queue_length = self.rollout_queue.qsize()
            while trajectory is None:
                try:
                    if self._is_actor_zero:
                        print(f"Getting from rollout_queue queue.")
                    trajectory = self.rollout_queue.get(timeout=0.5)

                    # Move tensors back to GPU
                    trajectory = [
                        (
                            tensor.to(self._device)
                            if isinstance(tensor, torch.Tensor)
                            else tensor
                        )
                        for tensor in trajectory
                    ]
                except ray.util.queue.Empty:
                    trajectory = None
                    time.sleep(0.1)
            time_wait_end = time.perf_counter()
            time_waiting_buffer = time_wait_end - time_step_start

            (
                query_responses,
                responses,
                logprobs,
                query_response_padding_masks,
                seq_lens,
                answers,
                policy_version,
            ) = trajectory

            context_length = query_responses.shape[1] - responses.shape[1]

            masks = generation.get_causal_mask_from_padding_mask(
                query_response_padding_masks
            )
            position_ids = generation.get_position_ids_from_padding_mask(
                query_response_padding_masks
            )

            # Reset GPU memory stats before model_running
            torch.cuda.reset_peak_memory_stats()

            time_grpo_steps_start = time.perf_counter()
            with torch.no_grad():
                ref_logits = self._ref_model(
                    query_responses, input_pos=position_ids, mask=masks
                )
            time_model_running = time.perf_counter() - time_grpo_steps_start

            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, responses, self._temperature
            )

            del ref_logits, position_ids, masks
            # masking of ref_logprobs is done in grpo_step

            # Repack trajectory with policy_version
            batch_size = logprobs.shape[:1] if logprobs.ndim > 1 else ()
            trajectory = Trajectory(
                query_responses=query_responses,
                responses=responses,
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                query_response_padding_masks=query_response_padding_masks,
                seq_lens=seq_lens,
                answers=answers,
                policy_version=policy_version,
                batch_size=batch_size,
            )
            print(f"putting trajectory {trajectory} into actor queue")

            # Move tensors to CPU before putting into the queue
            trajectory = trajectory.cpu()

            # Update circular queue
            self.replay_buffer.add(trajectory)

            # End of step timing
            time_total_ref_step = time.perf_counter() - time_step_start

            # log metrics
            if self._is_actor_zero:
                self._log_metrics(
                    step_idx=idx,
                    time_total_ref_step=time_total_ref_step,
                    time_model_running=time_model_running,
                    time_waiting_buffer=time_waiting_buffer,
                    # TODO: what should we do with this? We can log the total number of elements written in the buffer instead
                    # full_queue_data_discard=full_queue_data_discard,
                    # rollout_queue_length=rollout_queue_length,
                )

            torch.cuda.empty_cache()

            idx += 1


class vLLMRolloutActor:
    def __init__(self, *args, queue, cfg, actor_id=-1, **kwargs):
        import os

        import torch

        print(f"Actor CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Device count: {torch.cuda.device_count()}")

        self.cfg = cfg
        self._max_generated_tokens = self.cfg.max_generated_tokens
        self.grpo_samples = self.cfg.grpo_samples
        self._temperature = self.cfg.temperature
        # FIXME: I don't know what this is for and haven't used this yet
        self._top_k = self.cfg.top_k
        self.batch_size = self.cfg.vllm.batch_size
        self._steps_before_sync = self.cfg.steps_before_sync * self.cfg.num_fsdp_workers

        self.actor_id = actor_id
        self._is_actor_zero = self.actor_id == 0

        self.rollout_queue = queue
        self.llm = LLM(*args, **kwargs)

        from torchtune import config

        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        collate_name = self.cfg.get(
            "collate_fn", "torchtune.dev.grpo.data.padded_collate_rl"
        )
        # NOTE: the data mix is now a bit different, vllm now owns the dataloader rather
        # than the data parallel worker
        # A separate design I saw in OpenRLHF was for data parallel workers to
        # place their inputs from dataloader into a vllm queue
        # where each vllm instance has 1 queue per assigned dp worker
        self._dataloader = self._setup_data(
            self.cfg.dataset,
            shuffle=self.cfg.shuffle,
            batch_size=self.batch_size,
            collate_fn=collate_name,
        )

        # I'm using this to stop the generation until weight sync is done
        # FIXME: Should really use a lock
        self.sleeping = False

        # Initialize policy version for tracking trajectory age
        self.policy_version = 0

        self.metric_logger = None  # Placeholder for the logger
        generate_kwargs = {
            "n": self.grpo_samples,
            "max_tokens": self._max_generated_tokens,
            "temperature": self._temperature,
            # nondeterministically returns more than 1??
            # "logprobs": 1,
        }
        self.llm_wrapped = vLLMWrapper(self.llm, return_log_probs=True, generate_kwargs=generate_kwargs, pad_output=False, inplace=False, from_text=False)

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        self._metric_logger = logger

    def start_weight_update(self, param_list, policy_version):
        # Update the policy version when weights are synchronized
        self.policy_version = policy_version
        for name, dtype, shape in param_list:
            self.llm.collective_rpc("update_weight", args=(name, dtype, shape))

    def llm_collective_rpc(self, *args, **kwargs):
        self.llm.collective_rpc(*args, **kwargs)

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

    def wake_up(self):
        self.sleeping = False
        print(f"{self.__class__.__name__} (pid={os.getpid()}) woke up", flush=True)

    def is_sleeping(self):
        return self.sleeping

    def print_me(self, string):
        print(string, flush=True)

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
        if not self._is_actor_zero:
            return

        pct_time_model_running = (
            (time_generate / time_total_rollout) * 100 if time_total_rollout > 0 else 0
        )
        tokens_per_second = (
            total_generated_tokens / time_generate if time_generate > 0 else 0
        )
        div_GiB = 1024**3

        log_dict = {
            "vllm_actor_performance/total_rollout_time (s)": time_total_rollout,
            "vllm_actor_performance/pct_time_model_running (%)": pct_time_model_running,
            "vllm_actor_performance/tokens_per_second": tokens_per_second,
            "vllm_actor_performance/gpu_memory_peak_allocated (GiB)": gpu_memory[
                "allocated"
            ]
            / div_GiB,
            "vllm_actor_performance/gpu_memory_peak_reserved (GiB)": gpu_memory[
                "reserved"
            ]
            / div_GiB,
            "vllm_actor_performance/gpu_memory_peak_active (GiB)": gpu_memory["active"]
            / div_GiB,
            # "queues/vllm_full_queue_data_discard": full_queue_data_discard,
            # "queues/rollout_queue_size": self.rollout_queue.qsize(),
        }

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def vllm_to_trajectories(
        self, result: List[RequestOutput]
    ):  # -> PaddedTrajectoryBatch:
        prompts_str = result.get("prompts_str")
        responses_str = result.get("prompts_str")
        prompt_tokens = result.get("tokens", as_list=True)
        response_tokens = result.get("tokens_response", as_list=True)
        log_probs = result.get("log_probs", as_list=True)
        # TODO: we can ask the wrapper to pad for us - do we want padding on the left or right?
        padded_prompt_tokens = pad_and_stack(
            prompt_tokens, pad_value=self._tokenizer.pad_id
        )
        padded_response_tokens = pad_and_stack(
            response_tokens, pad_value=self._tokenizer.pad_id
        )
        padded_logprobs = pad_and_stack(log_probs, pad_value=0.0)
        batch_size = result.batch_size
        trajectory_batch = PaddedTrajectoryBatch(
            policy_model_id=self.policy_version,
            prompts_str=prompts_str,
            responses_str=responses_str,
            prompt_tokens=padded_prompt_tokens,
            response_tokens=padded_response_tokens,
            logprobs=padded_logprobs,
            # Initialize ref_logprobs as zeros - will be filled by RefActor
            ref_logprobs=torch.zeros_like(padded_logprobs),
            batch_size=batch_size,
        )
        return
        # return [padded_prompt_tokens, padded_response_tokens, padded_logprobs]

    # def compute_advantages(self, batch):
    #     rewards_full, successes_full, reward_metadata = batched_rewards(
    #         self._tokenizer, responses, answers, device=self._device
    #     )

    #     # B, G, num_funcs -> B, G
    #     rewards_sum = rewards_full.sum(-1)

    #     # advantage
    #     advantages = (rewards_sum - rewards_sum.mean(1, keepdim=True)) / (
    #         rewards_sum.std(1, keepdim=True) + 1e-4
    #     )
    #     advantages = advantages.reshape(batch_size * grpo_size)

    def rollout(self):
        for idx, batch in enumerate(self._dataloader):
            time_step_start = time.perf_counter()

            print(f"batch {idx}")
            # might want to do so for 0 also if weights are directly broadcasted from dp?
            if idx != 0 and idx % self._steps_before_sync == 0:
                # === sleep until weight synchronization is complete ===
                # this discards the current kv-cache, which I think is what we want (?)
                self.llm.reset_prefix_cache()
                # FIXME: use a lock
                self.sleeping = True

                print("started sleeping")
                start = time.time()
                while self.sleeping:
                    time.sleep(0.1)
                print(
                    f"vLLMRolloutActor slept for {time.time() - start} seconds while waiting for weight update"
                )

            if idx == 400:
                return

            # FIXME: tokens is currently on cpu, is this right?s
            tokens, answers = batch["tokens"], batch["answers"]
            # A downside is they only seem to take in List[List[int]] and not torch.Tensor :(
            # batch_tokens = batch_tokens.numpy().tolist()

            print(f'tokens shape: {batch["tokens"].shape}')

            # Reset GPU memory stats before generation
            torch.cuda.reset_peak_memory_stats(device="cuda:0")

            time_generate_start = time.perf_counter()
            # do the generation
            data = TensorDict(tokens=tokens, batch_size=len(tokens))
            print('sending data to llm', data)
            # Flatten the data since we have b x grpo_samples data
            data = self.llm_wrapped(data).view(-1)
            print('got data from llm', data)
            # TODO: if we use from_text=True in the wrapper we get these values for free
            data["prompts_str"] = NonTensorStack(*["" for _ in range(data.shape[0])])
            data["response_str"] = NonTensorStack(*["" for _ in range(data.shape[0])])
            # result = self.llm.generate(
            #     prompts=None,
            #     prompt_token_ids=tokens,
            #     sampling_params=sampling_params,
            #     use_tqdm=False,
            # )

            time_generate = time.perf_counter() - time_generate_start

            postprocessed_results = self.vllm_to_trajectories(data)

            print("hereherehere")

            # Unpack to compute padded tokens percentage and tokens per second
            (
                query_responses,
                responses,
                logprobs,
                query_response_padding_masks,
                seq_lens,
            ) = postprocessed_results
            # Compute total generated tokens for tokens per second
            total_generated_tokens = seq_lens.sum().item()

            print(f"query_responses shape: {query_responses.shape}")
            print(f"logprobs shape: {logprobs.shape}")
            print(f"responses shape: {responses.shape}")
            print(
                f"query_response_padding_masks shape: {query_response_padding_masks.shape}"
            )

            postprocessed_results.append(answers)
            postprocessed_results.append(self.policy_version)

            # print(self._tokenizer.decode(batch_tokens[0]))
            # print("===")
            # print(self._tokenizer.decode(postprocessed_results[0][0].cpu().numpy().tolist()))

            # Move tensors to CPU before putting into the queue
            postprocessed_results = [
                tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
                for tensor in postprocessed_results
            ]

            # Update circular queue and count full queue tries
            # full_queue_data_discard = 0
            while True:
                try:
                    self.rollout_queue.put_nowait(postprocessed_results)
                    break
                except QueueFull:
                    self.rollout_queue.get()  # Remove the oldest item to make space
                    # full_queue_data_discard += 1
                    print(f"rollout queue full. Discarding data.")

            if self._is_actor_zero:
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


class vLLMWorkerWrapper(Worker):
    """vLLM Rollout Model worker for Ray."""

    def __init__(self, *args, **kwargs):
        import os

        print(f"visible devices {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)

    def init_weight_update_group(self, master_address, master_port, rank, world_size):
        from vllm.distributed.parallel_state import get_world_group

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

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # src=0 because fsdp worker 0 has been assigned as "0" in this process group
        self._model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


@ray.remote(num_cpus=8, num_gpus=1)
class PyTorchActorModel:
    def __init__(
        self,
        cfg,
        environment_variables,
        replay_buffer,
    ):
        import torch

        # shared queue to get trajectories + logprobs from vllm
        self.replay_buffer = replay_buffer

        self.cfg = cfg

        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        device_type = self.cfg.device
        self._log_peak_memory_stats = self.cfg.get("log_peak_memory_stats", True)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # if self._log_peak_memory_stats and self._device.type != "cuda":
        #     log.info(
        #         "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
        #     )
        #     self._log_peak_memory_stats = False

        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)

        self.distributed_backend = training.get_distributed_backend(
            device_type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # ============= [START] This bit replaces init_process_group =============
        # === Simulate torchrun to set env vars ===
        import os

        for var in environment_variables:
            os.environ[var] = str(environment_variables[var])
        # =========================================

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
        print(self.device_mesh)
        # ============= [END] This bit replaces init_process_group =============

        world_size, rank = utils.get_world_size_and_rank()
        self.rank = rank
        self.world_size = world_size
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # activation checkpointing
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )

        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # recipe state attributes
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        # This was only used for generate, so I don't think I need it?
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        # self._forward_batch_size = cfg.forward_batch_size

        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.save_every_n_epochs
        self._total_steps = cfg.num_steps

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._optimizer = self._setup_optimizer(cfg_optimizer=cfg.optimizer)
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        # TODO: generalize this to any chunked loss
        if self._loss_fn.__class__.__name__ == "GRPOWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._lr_scheduler = None
        # FIXME: need to get _steps_per_epoch when dataloader is no longer per fsdp worker but instead wrapped in vLLM
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.get("lr_scheduler", None),
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        self._steps_before_sync = cfg.steps_before_sync

        # Initialize policy version for tracking age of trajectories
        self.policy_version = 0
        self.metric_logger = None  # Placeholder for the logger

        print("done setup")

    def set_metric_logger(self, logger):
        """Store the MetricLoggerActor handle."""
        if self._is_rank_zero:
            print("setting metric logger {logger} for rank", self.rank)
            self._metric_logger = logger

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def init_model_update_group(self, master_address, master_port, rank, world_size):
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            # FIXME: hardcoding, not sure if this is right
            torch.device(f"cuda:0"),
        )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        model_state_dict: Dict[str, Any],
        # ref_model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """
        from torchtune.training import disable_dropout

        # utils.log_rank_zero(
        #     log,
        #     "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        # )
        # init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)
            # ref_model = config.instantiate(cfg_model)

        # ref_model.eval()
        # for p in ref_model.parameters():
        #     p.requires_grad = False

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)
            # training.compile_model(ref_model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=True,
        )

        # training.shard_model(
        #     model=ref_model,
        #     shard_conditions=fsdp_shard_conditions,
        #     cpu_offload=fsdp_cpu_offload,
        #     reshard_after_forward=True,
        # )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

            # for m in ref_model.modules():
            #     if hasattr(m, "rope_init"):
            #         m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # training.load_from_full_model_state_dict(
        #     ref_model,
        #     ref_model_state_dict,
        #     self._device,
        #     strict=True,
        #     cpu_offload=fsdp_cpu_offload,
        # )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)
        # training.validate_no_params_on_meta_device(ref_model)

        # utils.log_rank_zero(
        #     log,
        #     f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        # )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)
        # disable_dropout(ref_model)

        # synchronize before training begins
        torch.distributed.barrier()

        return model  # , ref_model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict=None
    ) -> Optional[Optimizer]:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            assert opt_state_dict is None, "Optimizer state dict is not supported yet"
            # training.load_from_full_optimizer_state_dict(
            #     self._model,
            #     optimizer,
            #     opt_state_dict,
            #     self._device,
            # )
        return optimizer

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            context_length (int): input ids sequence length

        Returns:
            GRPOStats: An instance of :class:`~torchtune.rlhf.PPOStats`, a NamedTuple containing:
               - loss (torch.Tensor): The total PPO loss.
               - ratios (torch.Tensor): The ratio between the current and old policy probabilities.
               - clipfrac (torch.Tensor): The fraction of ratios that were clipped.
               - approx_policy_kls: Average estimated KL divergence between the policy before and after the optimisation step.

        """
        torch.cuda.empty_cache()
        if self._is_rank_zero:
            print("context_length", context_length)
            print([f.shape for f in trajectory if f is not None])

        # Create an output mask to avoid computing model.output on tokens we won't train on
        # We don't need to compute logits for the prompt tokens (except the last one which is used for the first prediction)
        # and we don't need the logit for the last token (as there's no next token to predict)
        output_mask = torch.zeros_like(
            trajectory.query_responses, dtype=torch.bool, device=self._device
        )
        output_mask[:, context_length - 1 : -1] = True

        # estimate logprobs from the policy at the current optimisation step
        with self.activations_handling_ctx:
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
                output_mask=output_mask,
            )

        # pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        # pi_logprobs = rlhf.batched_logits_to_logprobs(
        #     pi_logits,
        #     trajectory.query_responses[:, context_length:],
        #     self._temperature,
        #     chunk_size=1,
        # )

        # pi_logprobs[trajectory.response_padding_masks] = 1.0
        # trajectory.ref_logprobs[trajectory.response_padding_masks] = 1.0

        if self._is_rank_zero:
            # print(
            #     "ref_logprobs shape", trajectory.ref_logprobs.shape, pi_logprobs.shape
            # )
            # print(torch.abs(pi_logprobs - trajectory.ref_logprobs).max())
            # print(torch.abs(pi_logprobs - trajectory.ref_logprobs).mean())
            # print(pi_logprobs)
            print(trajectory.ref_logprobs)

        torch.cuda.empty_cache()

        # Extract response targets, aligned with pi_logits
        targets = trajectory.query_responses[:, context_length:]

        # Compute GRPO loss
        loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs = self._loss_fn(
            pi_logits=pi_logits,
            targets=targets,
            ref_logprobs=trajectory.ref_logprobs,
            advantages=trajectory.advantages,
            padding_masks=~trajectory.response_padding_masks,
        )
        with torch.no_grad():
            mask = ~trajectory.response_padding_masks  # True for non-padded tokens
            approx_policy_kls = (
                0.5 * ((pi_logprobs - trajectory.logprobs)[mask].pow(2)).mean()
            )

        del pi_logprobs, pi_logits

        torch.cuda.empty_cache()
        loss.backward()

        print("done grpo step")

        return GRPOStats(
            loss,
            policy_loss,
            kl_loss,
            ratios,
            clipfrac,
            approx_policy_kls,
        )

    def set_vllm_engines(self, engines):
        self._vllm_engines = engines

    def cleanup_after_step(
        self,
        trajectory: GRPOTrajectory,
        l_grpo_stats: list[GRPOStats],
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats

    def _log_metrics(
        self,
        step_idx,
        trajectory,
        grpo_stats,
        total_step_time,
        time_grpo_steps,
        time_waiting_buffer,
        time_weight_sync,
        time_weight_gather,
        number_of_tokens,
        padded_tokens_percentage,
        policy_age,
        rewards_mean,
        successes_mean,
        rewards_mean_per_func,
        successes_mean_per_func,
        reward_metada,
        train_replay_buffer_size,
    ):
        """Log metrics for the PyTorchActorModel, only on rank zero after reductions."""
        # Compute metrics that require all ranks
        grpo_stats_stacked = GRPOStats(*map(torch.stack, zip(*grpo_stats)))

        # Only log on rank zero
        if not self._is_rank_zero:
            return

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"train_actor_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        # Training metrics
        log_dict.update(
            {
                "train_actor_training/loss": grpo_stats_stacked.loss.mean().item(),
                "train_actor_training/policy_loss": grpo_stats_stacked.policy_loss.mean().item(),
                "train_actor_training/num_stop_tokens": trajectory.response_padding_masks.any(
                    -1
                )
                .sum()
                .item(),
                "train_actor_training/kl_loss": grpo_stats_stacked.kl_loss.mean().item(),
                "train_actor_training/ratios": grpo_stats_stacked.ratios.mean().item(),
                "train_actor_training/clipfrac": grpo_stats_stacked.clipfrac.mean().item(),
                "train_actor_training/approx_policy_kls": grpo_stats_stacked.approx_policy_kls.mean().item(),
                "train_actor_training/response_lengths": trajectory.seq_lens.float()
                .mean()
                .item(),
            }
        )

        # rewards and successes
        log_dict.update(
            {
                "train_actor_rewards/rewards_mean": rewards_mean.item(),
                "train_actor_rewards/successes_mean": successes_mean.item(),
            }
        )

        # Per-function rewards and successes
        for func_name, mean in zip(reward_metada["func_names"], rewards_mean_per_func):
            log_dict[f"train_actor_rewards/rewards_func_{func_name}_mean"] = mean.item()
        for func_name, mean in zip(
            reward_metada["func_names"], successes_mean_per_func
        ):
            log_dict[f"train_actor_rewards/successes_func_{func_name}_mean"] = (
                mean.item()
            )

        # Performance metrics
        log_dict.update(
            {
                "train_actor_performance/total_step_time (s)": total_step_time,
                "train_actor_performance/time_grpo_steps (s)": time_grpo_steps,
                "train_actor_performance/pct_time_grpo_steps (%)": (
                    (time_grpo_steps / total_step_time) * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/tokens_per_second": (
                    number_of_tokens / total_step_time if total_step_time > 0 else 0
                ),
                "train_actor_performance/time_weight_sync (s)": time_weight_sync,
                "train_actor_performance/pct_time_weight_sync (%)": (
                    (time_weight_sync / total_step_time) * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/padded_tokens_percentage (%)": padded_tokens_percentage,
                "train_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "train_actor_performance/pct_time_waiting_buffer (%)": (
                    (time_waiting_buffer / total_step_time) * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/time_weight_gather (s)": time_weight_gather,
                "train_actor_performance/pct_time_weight_gather (%)": (
                    (time_weight_gather / total_step_time) * 100
                    if total_step_time > 0
                    else 0
                ),
            }
        )

        # Queue metrics
        log_dict.update(
            {
                "queues/train_actor_policy_age_mean": policy_age,
                "queues/train_replay_buffer_size": train_replay_buffer_size,
            }
        )

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def train(self):
        if self._is_rank_zero:
            self._vllm_engines[0].print_me.remote(
                "hello vllm worker, it's fsdp rank zero!"
            )
        from torchtune import generation
        from torchtune.dev.grpo.rewards import batched_rewards

        training.cleanup_before_training()
        self._optimizer.zero_grad()

        training_completed = False
        grpo_size = self.grpo_samples
        batch_size = self.batch_size

        self._profiler.start()

        # for curr_epoch in range(self._epochs_run, self.total_epochs):
        for curr_epoch in range(1):
            print("starting")

            dataloader_done = False
            while not dataloader_done:
                time_step_start = time.perf_counter()

                # Measure time waiting for buffer
                time_waiting_buffer_start = time.perf_counter()
                trajectory = None
                if self._is_rank_zero:
                    train_replay_buffer_size = self.replay_buffer.write_count
                while not len(self.replay_buffer):
                    print("waiting for replay buffer")
                    time.sleep(1.0)
                if self._is_rank_zero:
                    print(f"{self.rank=} Getting from replay_buffer queue.")
                trajectory = self.replay_buffer.sample(batch_size=batch_size)[0]
                trajectory = trajectory.to(self._device)
                time_waiting_buffer = time.perf_counter() - time_waiting_buffer_start

                print(f"{self.rank=} got from queue traj {trajectory}")

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and self._steps_run
                    == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    print("starting _record_memory_history")
                    torch.cuda.memory._record_memory_history()

                # # hacky way of coordinating between vllm and actor that dataloader is done
                # # since vllm worker now "owns" the dataloader
                # if len(trajectory) == 1:
                #     print(f"{self.rank=} got done")
                #     dataloader_done = True
                #     torch.distributed.barrier()
                #     torch.cuda.synchronize()
                #     print(f"{self.rank=} returning")
                #     return

                # print(f"{self.rank=} got trajectory, {len(trajectory)}, {trajectory[0].device}")
                if is_tensorclass(trajectory):
                    # we should be here
                    query_responses = trajectory.query_responses
                    responses = trajectory.responses
                    logprobs = trajectory.logprobs
                    ref_logprobs = trajectory.ref_logprobs
                    query_response_padding_masks = (
                        trajectory.query_response_padding_masks
                    )
                    seq_lens = trajectory.seq_lens
                    answers = trajectory.answers
                    policy_version = trajectory.policy_version
                else:
                    # FIXME: I don't think we end up here anymore
                    # we should not be here
                    query_responses = trajectory["query_responses"]
                    responses = trajectory["responses"]
                    logprobs = trajectory["logprobs"]
                    ref_logprobs = trajectory["ref_logprobs"]
                    query_response_padding_masks = trajectory[
                        "query_response_padding_masks"
                    ]
                    seq_lens = trajectory["seq_lens"]
                    answers = trajectory["answers"]
                    policy_version = trajectory["policy_version"]
                    print(
                        f"expected a tensorclass but got a tensordict on rank {self.rank}"
                    )

                # Compute padded tokens percentage
                total_tokens = query_responses.numel()
                padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
                padded_tokens_percentage = (
                    (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                )
                number_of_tokens = seq_lens.sum().item()

                # Reset peak memory stats before GRPO steps
                torch.cuda.reset_peak_memory_stats()

                # FIXME: move stop token tensor to __init__
                response_padding_masks, responses = (
                    rlhf.truncate_sequence_at_first_stop_token(  # [B x G, L]
                        responses,
                        torch.tensor(self._tokenizer.stop_tokens, device=self._device),
                        self._tokenizer.pad_id,
                    )
                )

                masks = generation.get_causal_mask_from_padding_mask(
                    query_response_padding_masks
                )
                position_ids = generation.get_position_ids_from_padding_mask(
                    query_response_padding_masks
                )
                del query_response_padding_masks

                context_length = query_responses.shape[1] - responses.shape[1]

                # compute rewards
                responses = responses.reshape(batch_size, grpo_size, -1)
                rewards_full, successes_full, reward_metadata = batched_rewards(
                    self._tokenizer, responses, answers, device=self._device
                )

                # B, G, num_funcs -> B, G
                rewards_sum = rewards_full.sum(-1)

                # advantage
                advantages = (rewards_sum - rewards_sum.mean(1, keepdim=True)) / (
                    rewards_sum.std(1, keepdim=True) + 1e-4
                )
                advantages = advantages.reshape(batch_size * grpo_size)

                trajectory = GRPOTrajectory(
                    query_responses=query_responses,
                    logprobs=logprobs,
                    ref_logprobs=ref_logprobs,
                    advantages=advantages,
                    masks=masks,
                    position_ids=position_ids,
                    response_padding_masks=response_padding_masks,
                    seq_lens=training.get_unmasked_sequence_lengths(
                        response_padding_masks
                    ),
                )

                # for logging
                torch.distributed.reduce(
                    rewards_full, dst=0, op=torch.distributed.ReduceOp.AVG
                )
                torch.distributed.reduce(
                    successes_full, dst=0, op=torch.distributed.ReduceOp.AVG
                )
                rewards_mean_per_func = rewards_full.mean(dim=(0, 1)).cpu()
                successes_mean_per_func = successes_full.mean(dim=(0, 1)).cpu()
                rewards_mean = rewards_mean_per_func.mean()
                successes_mean = successes_mean_per_func.mean()

                del rewards_full, successes_full, responses, rewards_sum

                # TODO: do we need a barrier here?
                torch.distributed.barrier()

                # Measure compute time across all GRPO steps
                time_grpo_steps = 0
                time_grpo_steps_start = time.perf_counter()
                grpo_stats: list[GRPOStats] = []
                for _ in range(self._ppo_epochs):

                    step_stats = self.grpo_step(trajectory, context_length)

                    grpo_stats.append(step_stats)

                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                    torch.distributed.barrier()
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    torch.distributed.barrier()

                    self.global_step += 1

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    print(f"{self.rank=} finished step {self._steps_run}")

                time_grpo_steps = time.perf_counter() - time_grpo_steps_start
                self._steps_run += 1

                # Compute policy age
                policy_age = self.policy_version - policy_version

                # Handle weight synchronization
                time_weight_sync = 0
                time_weight_gather = 0
                if self._steps_run % self._steps_before_sync == 0:
                    print("started weight sync")
                    torch.distributed.barrier()

                    time_weight_gather_start = time.perf_counter()
                    # gather all parameters
                    new_sd = {}
                    for k, v in self._model.state_dict().items():
                        new_sd[k] = v.full_tensor()
                    torch.cuda.synchronize()
                    time_weight_gather = time.perf_counter() - time_weight_gather_start

                    if self._is_rank_zero:
                        print(f"Done gather in {time_weight_gather}")

                    time_sync_start = time.perf_counter()
                    self.sync_weights(new_sd)
                    del new_sd
                    time_weight_sync = time.perf_counter() - time_sync_start

                # Log metrics
                total_step_time = time.perf_counter() - time_step_start
                total_step_time = time.perf_counter() - time_step_start
                policy_age = self.policy_version - policy_version

                if self._is_rank_zero:
                    self._log_metrics(
                        step_idx=self.global_step,
                        trajectory=trajectory,
                        grpo_stats=grpo_stats,
                        total_step_time=total_step_time,
                        time_grpo_steps=time_grpo_steps,
                        time_waiting_buffer=time_waiting_buffer,
                        time_weight_sync=time_weight_sync,
                        time_weight_gather=time_weight_gather,
                        number_of_tokens=number_of_tokens,
                        padded_tokens_percentage=padded_tokens_percentage,
                        policy_age=policy_age,
                        rewards_mean=rewards_mean,
                        successes_mean=successes_mean,
                        rewards_mean_per_func=rewards_mean_per_func,
                        successes_mean_per_func=successes_mean_per_func,
                        reward_metada=reward_metadata,
                        train_replay_buffer_size=train_replay_buffer_size,
                    )

                # Step profiler
                # Note that this is called within gradient accumulation block, hence
                # will include multiple forward / backward passes if gradient accumulation > 1
                self._profiler.step()

                # Stop tracking CUDA memory now that active steps are complete
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and self._steps_run
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and self._device.type == "cuda"
                ):
                    print("stop _record_memory_history")
                    torch.cuda.memory._record_memory_history(enabled=None)

                torch.distributed.barrier()
                self.cleanup_after_step(trajectory, grpo_stats)

                if self._steps_run == self._total_steps:
                    training_completed = True
                    return

        self._profiler.stop()

    def sync_weights(self, new_sd):
        # Increment policy version
        self.policy_version += 1

        if self._is_rank_zero:
            # Convert to vLLM-compatible format
            # FIXME: don't hardcode kwargs here
            new_sd = qwen2_tune_to_hf(new_sd, num_heads=16, num_kv_heads=2, dim=2048)
            # Prepare parameter metadata list
            param_list = [(k, v.dtype, v.shape) for k, v in new_sd.items()]

            # Start weight update on vLLM workers (non-blocking)
            vllm_update_refs = [
                eng.start_weight_update.remote(param_list, self.policy_version)
                for eng in self._vllm_engines
            ]

            # Broadcast each parameter to vLLM workers
            for k, v in new_sd.items():
                self._model_update_group.broadcast(
                    v, 0, stream=torch.cuda.current_stream()
                )

            # Wait for vLLM workers to finish updating
            ray.get(vllm_update_refs)

            # Wake up vLLM workers to resume rollouts
            for eng in self._vllm_engines:
                eng.wake_up.remote()
        else:
            # Non-zero training ranks don’t participate in vLLM weight sync
            pass

        torch.distributed.barrier()
        print("waking up", flush=True)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()


@ray.remote(num_cpus=1, num_gpus=0)
class MetricLoggerActor:
    def __init__(self, cfg):
        self.logger = config.instantiate(cfg.metric_logger)

    def log_dict(self, log_dict, step=None):
        # allowing actors to use their own step counters
        self.logger.log_dict(log_dict, step=step)

    def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()


class RayGRPORecipe:
    def setup(self, cfg):
        self.cfg = cfg

        # Store worker counts as instance variables
        self.num_vllm_workers = cfg.vllm.num_workers
        self.vllm_tp_size = cfg.vllm.tp_size
        self.num_ref_workers = cfg.num_ref_workers
        self.num_fsdp_workers = cfg.num_fsdp_workers

        # Initialize queues
        self.rollout_queue = Queue(
            actor_options={"num_cpus": 10, "num_gpus": 0},
            maxsize=cfg.vllm.queue_maxsize,
        )
        self.replay_buffer = RayReplayBuffer(
            storage=functools.partial(LazyStackStorage, max_size=1000),
            batch_size=1,
            remote_config={"num_cpus": 10, "num_gpus": 0},
        )

        # Create workers using config values directly
        self.rollout_workers = self._create_vllm_workers()
        self.ref_workers = self._create_ref_workers()
        self.actor_workers = self._create_fsdp_group(
            worker_cls=PyTorchActorModel, fsdp_world_size=self.num_fsdp_workers
        )
        self._init_weight_sync_pg()

        # needs to happens after workers are created
        # or there are conflicts with the placement group
        self._set_metric_logger_to_actors()

    def start_ray(self):
        total_gpus = (
            self.num_vllm_workers * self.vllm_tp_size
            + self.num_ref_workers
            + self.num_fsdp_workers
        )
        total_cpus = 32 * total_gpus + 2
        ray.init(num_cpus=total_cpus, num_gpus=total_gpus)
        print("Cluster resources:", ray.cluster_resources())

    def _set_metric_logger_to_actors(self):
        self.metric_logger = MetricLoggerActor.remote(self.cfg)
        # Pass the logger handle to each worker
        for worker in self.rollout_workers:
            worker.set_metric_logger.remote(self.metric_logger)
        for worker in self.ref_workers:
            worker.set_metric_logger.remote(self.metric_logger)
        for worker in self.actor_workers:
            worker.set_metric_logger.remote(self.metric_logger)

    def _create_fsdp_group(self, worker_cls, fsdp_world_size: int):
        addr, port = get_ip(), get_open_port()
        fsdp_workers = []
        for i in range(fsdp_world_size):
            env_vars = {
                "RANK": str(i),
                "WORLD_SIZE": fsdp_world_size,
                "MASTER_ADDR": addr,
                "MASTER_PORT": port,
            }
            worker = worker_cls.remote(
                self.cfg,
                env_vars,
                self.replay_buffer,
            )
            fsdp_workers.append(worker)
        return fsdp_workers

    def _create_ref_worker(self):
        worker = RefActor.remote(
            rollout_queue=self.rollout_queue,
            replay_buffer=self.replay_buffer,
            cfg=self.cfg,
        )
        return worker

    def _create_vllm_workers(self):
        llms = []
        for i in range(self.num_vllm_workers):
            # Define placement group for this worker
            pg_inference = placement_group([{"GPU": 1, "CPU": 10}] * self.vllm_tp_size)
            ray.get(pg_inference.ready())
            print(
                f"Placement group for vLLM worker {i} ready with {self.vllm_tp_size} GPUs"
            )
            scheduling_inference = PlacementGroupSchedulingStrategy(
                placement_group=pg_inference,
                placement_group_capture_child_tasks=True,
            )

            # Create the remote actor without specifying resources directly
            llm = (
                ray.remote(
                    num_cpus=0,
                    num_gpus=0,  # No additional GPUs/CPUS needed outside placement group
                    scheduling_strategy=scheduling_inference,
                )(vLLMRolloutActor)
                .options(max_concurrency=5)
                .remote(
                    model="Qwen/Qwen2.5-3B",
                    enforce_eager=True,
                    enable_chunked_prefill=True,
                    dtype="bfloat16",
                    worker_cls=vLLMWorkerWrapper,
                    tensor_parallel_size=self.vllm_tp_size,
                    distributed_executor_backend="ray",
                    queue=self.rollout_queue,
                    cfg=self.cfg,
                    actor_id=i,
                )
            )
            llms.append(llm)
        return llms

    def _create_ref_workers(self):
        workers = []
        for i in range(self.num_ref_workers):
            worker = RefActor.remote(
                rollout_queue=self.rollout_queue,
                replay_buffer=self.replay_buffer,
                cfg=self.cfg,
                actor_id=i,
            )
            workers.append(worker)
        return workers

    def _init_weight_sync_pg(self):
        addr, weight_update_port = get_ip(), get_open_port()
        weight_sync_world_size = self.num_vllm_workers * self.vllm_tp_size + 1

        # only FSDP rank 0 is in the weight sync process group
        handle = self.actor_workers[0].init_model_update_group.remote(
            addr,
            weight_update_port,
            0,
            weight_sync_world_size,
        )

        # this call makes all vllm workers part of weight sync group
        [
            worker.llm_collective_rpc.remote(
                "init_weight_update_group",
                args=(
                    addr,
                    weight_update_port,
                    i * self.vllm_tp_size + 1,
                    weight_sync_world_size,
                ),
            )
            for (i, worker) in enumerate(self.rollout_workers)
        ]

        # only need to .get one of the handles since this will block until
        # all participating ranks init
        ray.get(handle)

        self.rollout_workers = self.rollout_workers

        ray.get(self.actor_workers[0].set_vllm_engines.remote(self.rollout_workers))

    def train(self):
        rollout_handles = [worker.rollout.remote() for worker in self.rollout_workers]
        self.rollout_workers[0].print_me.remote("hello vllm worker, it's __main__")
        ref_handles = [worker.run.remote() for worker in self.ref_workers]
        worker_handles = [worker.train.remote() for worker in self.actor_workers]
        ray.get(rollout_handles + ref_handles + worker_handles)
        ray.get(self.actor_workers[0].cleanup.remote())

    def stop_ray(self):
        ray.shutdown()


@config.parse
def recipe_main(cfg: DictConfig) -> None:

    if cfg.get("enable_expandable_segments", True):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    recipe = RayGRPORecipe()
    recipe.setup(cfg)
    recipe.train()
    recipe.stop_ray()


if __name__ == "__main__":
    recipe_main()
