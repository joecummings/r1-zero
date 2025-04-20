from typing import Any, Dict, List

import ray
import torch
import torchtune.training as training
from omegaconf import DictConfig

from torchtune import config, generation, rlhf, utils
from torchtune.dev.grpo.rewards import RewardOutput, RewardBase
from torchtune.dev.rl.datatypes import Trajectory

log = utils.get_logger("DEBUG")


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
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self._dtype = training.get_dtype(self.cfg.dtype, device=self._device)
        ref_checkpoint_dict = self.load_ref_checkpoint(
            cfg_ref_checkpointer=self.cfg.ref_checkpointer
        )
        self._ref_model = self._setup_model(
            self.cfg.model, ref_checkpoint_dict[training.MODEL_KEY]
        )
        self._temperature = self.cfg.temperature

        self.metric_logger = None  # Placeholder for the logger

        self.grpo_samples = self.cfg.grpo_samples
        self.vllm_batch_size = self.cfg.vllm.batch_size

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

        self.STOP_TOKENS_TENSOR = torch.tensor(
            self._tokenizer.stop_tokens, device=self._device
        )

        self.reward_functions: List[RewardBase] = [config.instantiate(reward_fn) for reward_fn in self.cfg.reward_functions]

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
        rollout_queue_size,
        reward_outputs,
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
                "queues/rollout_queue_size": rollout_queue_size,
            }
        )

        rewards_mean = torch.tensor([reward_output.total_reward.mean() for reward_output in reward_outputs]).mean()
        successes_mean = torch.tensor([reward_output.successes.mean() for reward_output in reward_outputs]).mean()
        log_dict.update(
            {
                "ref_actor_rewards/rewards_mean": rewards_mean.item(),
                "ref_actor_rewards/successes_mean": successes_mean.item(),
            }
        )

        for reward_output in reward_outputs:
            log_dict.update(reward_output.log(prefix="ref_actor_rewards"))


        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def run(self):
        import time

        log.info("running ref actor")
        idx = 0
        while True:
            if idx == self.cfg.num_steps:
                break

            # Start measuring total step time
            time_step_start = time.perf_counter()
            trajectory = None
            if self._is_actor_zero:
                rollout_queue_size = self.rollout_queue.qsize()
            while trajectory is None:
                try:
                    if self._is_actor_zero:
                        log.info("Getting from rollout_queue queue.")
                    trajectory = self.rollout_queue.get(timeout=0.5)
                    trajectory = trajectory.to(self._device)
                except ray.util.queue.Empty:
                    trajectory = None
                    time.sleep(0.1)
            time_wait_end = time.perf_counter()
            time_waiting_buffer = time_wait_end - time_step_start

            context_length = (
                trajectory.query_responses.shape[1] - trajectory.responses.shape[1]
            )

            masks = generation.get_causal_mask_from_padding_mask(
                trajectory.query_response_padding_masks
            )
            position_ids = generation.get_position_ids_from_padding_mask(
                trajectory.query_response_padding_masks
            )

            # Reset GPU memory stats before model_running
            torch.cuda.reset_peak_memory_stats()

            time_grpo_steps_start = time.perf_counter()
            with torch.no_grad():
                ref_logits = self._ref_model(
                    trajectory.query_responses, input_pos=position_ids, mask=masks
                )
            time_model_running = time.perf_counter() - time_grpo_steps_start

            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, trajectory.responses, self._temperature
            )

            batch_size = self.cfg.vllm.batch_size  # B
            group_size = self.grpo_samples  # G

            del ref_logits, position_ids, masks
            # masking of ref_logprobs is done in grpo_step

            # Extract components from raw trajectory: these have size [B * G, T]
            print(f"Extracting components from raw trajectory: {trajectory}")
            query_responses = trajectory.query_responses
            responses = trajectory.responses
            query_response_padding_masks = trajectory.query_response_padding_masks
            seq_lens = trajectory.seq_lens
            answers = trajectory.answers  # list[str] of len (B * G)

            # Compute padded tokens percentage
            total_tokens = query_responses.numel()
            padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
            padded_tokens_percentage = (
                (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
            )
            number_of_tokens = seq_lens.sum().item()

            # Truncate sequences at first stop token
            (
                response_padding_masks,
                responses,
            ) = rlhf.truncate_sequence_at_first_stop_token(
                responses,
                self.STOP_TOKENS_TENSOR.to(self._device),
                self._tokenizer.pad_id,
            )

            # Generate masks and position IDs
            masks = generation.get_causal_mask_from_padding_mask(
                query_response_padding_masks
            )
            position_ids = generation.get_position_ids_from_padding_mask(
                query_response_padding_masks
            )
            context_length = query_responses.shape[1] - responses.shape[1]
            del query_response_padding_masks

            # Compute rewards
            response_ids = responses.reshape(batch_size * group_size, -1)
            responses_str = [self._tokenizer.decode(response_ids[i].tolist()) for i in range(batch_size * group_size)]
            reward_outputs: List[RewardOutput] = []
            for reward_fn in self.reward_functions:
                reward_outputs.append(reward_fn(
                    response_ids, responses_str, answers, device=self._device
                ))

            group_rewards = torch.stack([reward_output.total_reward for reward_output in reward_outputs], dim=-1) # (B * G, num_funcs)
            group_rewards = group_rewards.reshape(batch_size, group_size, -1) 
            # Compute advantages: B, G, num_funcs -> B, G
            group_rewards = group_rewards.sum(-1)
            # To compute advantage, subtract the mean of the group rewards from each group reward
            group_advantages = (group_rewards - group_rewards.mean(1, keepdim=True)) / (
                group_rewards.std(1, keepdim=True) + 1e-4
            )  # (B, G)

            # Repack trajectory with policy_version

            trajectory = Trajectory(
                query_responses=trajectory.query_responses,
                responses=trajectory.responses,
                logprobs=trajectory.logprobs,
                ref_logprobs=ref_logprobs,
                query_response_padding_masks=trajectory.query_response_padding_masks,
                seq_lens=trajectory.seq_lens,
                answers=trajectory.answers,
                policy_version=trajectory.policy_version,
                advantages=group_advantages.reshape(batch_size * group_size),  # (B, G)
                batch_size=batch_size * group_size,
            )

            log.info(f"Constructed trajectory: {trajectory}")
            # Move tensors to CPU before putting into the queue
            trajectory = trajectory.cpu()

            # Update circular queue
            self.replay_buffer.extend(trajectory)

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
                    rollout_queue_size=rollout_queue_size,
                    reward_outputs=reward_outputs,
                )

            torch.cuda.empty_cache()

            idx += 1
