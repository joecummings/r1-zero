from typing import Any, Dict

import ray
import torch
import torchtune.training as training
from omegaconf import DictConfig

from torchtune import config, generation, rlhf, utils
from torchtune.dev.grpo.rewards import batched_rewards
from torchtune.dev.rl.datatypes.trajectory import GroupedUnscoredTrajectories, ScoredTrajectory

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
        self._compile = cfg.get("compile", False)

        ref_checkpoint_dict = self.load_ref_checkpoint(
            cfg_ref_checkpointer=self.cfg.ref_checkpointer
        )
        self._ref_model = self._setup_model(
            self.cfg.model, ref_checkpoint_dict[training.MODEL_KEY]
        )

        #TODO: replace with constante IGNORE_INDEX
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

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

        if self._compile:
            training.compile_model(ref_model, verbose=self._is_rank_zero)

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
        step_idx: int,
        grouped_traj: GroupedUnscoredTrajectories,
        time_total_ref_step: float,
        time_model_running: float,
        time_waiting_buffer: float,
        rollout_queue_size: int,
        rewards_mean: torch.Tensor,
        successes_mean: torch.Tensor,
        rewards_mean_per_func: torch.Tensor,
        successes_mean_per_func: torch.Tensor,
        reward_metadata: Dict,
        number_of_tokens: int
    ) -> None:
        """Log metrics for the RefActor, only on actor zero.

        Args:
            step_idx (int): Current training step index.
            grouped_traj (GroupedUnscoredTrajectories): Processed group of trajectories.
            time_total_ref_step (float): Total time for the ref step.
            time_model_running (float): Time spent running the model.
            time_waiting_buffer (float): Time spent waiting for buffer.
            rollout_queue_size (int): Size of the rollout queue.
            rewards_mean (torch.Tensor): Mean reward across all sequences.
            successes_mean (torch.Tensor): Mean success across all sequences.
            rewards_mean_per_func (torch.Tensor): Mean rewards per function.
            successes_mean_per_func (torch.Tensor): Mean successes per function.
            reward_metadata (Dict): Metadata including function names.
            number_of_tokens (int): Total number of response tokens processed.

        Returns:
            None

        Example:
            ref_actor._log_metrics(1, grouped_traj, 2.5, 1.2, 0.3, 10, ...)
        """
        if not self._is_actor_zero:
            return

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {f"ref_actor_performance/memory/{k}": v for k, v in memory_stats.items()}
            )
        tokens_per_second = number_of_tokens / time_total_ref_step if time_total_ref_step > 0 else 0

        log_dict.update({
            "ref_actor_performance/time_total_ref_step (s)": time_total_ref_step,
            "ref_actor_performance/time_model_running (s)": time_model_running,
            "ref_actor_performance/pct_time_model_running (%)": (
                time_model_running / time_total_ref_step * 100 if time_total_ref_step > 0 else 0
            ),
            "ref_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
            "ref_actor_performance/pct_time_waiting_buffer (%)": (
                time_waiting_buffer / time_total_ref_step * 100 if time_total_ref_step > 0 else 0
            ),
            "ref_actor_performance/response_tokens_processed": number_of_tokens,
            "ref_actor_performance/tokens_per_second": tokens_per_second,
            "queues/rollout_queue_size": rollout_queue_size,
            "ref_actor_rewards/rewards_mean": rewards_mean.item(),
            "ref_actor_rewards/successes_mean": successes_mean.item(),
        })

        function_names = reward_metadata.get("func_names", [])
        for func_name, r_mean, s_mean in zip(function_names, rewards_mean_per_func, successes_mean_per_func):
            log_dict[f"ref_actor_rewards/rewards_func_{func_name}_mean"] = r_mean.item()
            log_dict[f"ref_actor_rewards/successes_func_{func_name}_mean"] = s_mean.item()

        if self._metric_logger:
            ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))
        else:
            log.warning(f"RefActor {self.actor_id} metric logger not set, cannot log metrics.")

    def run(self):
        """Main loop for processing grouped trajectories from the rollout queue."""
        import time

        log.info(f"RefActor {self.actor_id} starting run loop.")
        idx = 0
        while idx < self.cfg.num_steps:
            time_step_start = time.perf_counter()
            rollout_queue_size = self.rollout_queue.qsize() if self._is_actor_zero else -1

            #TODO: replace with generator
            try:
                grouped_traj = self.rollout_queue.get(block=True, timeout=1.0)
                grouped_traj = self._move_to_device(grouped_traj)
            except ray.util.queue.Empty:
                log.debug(f"RefActor {self.actor_id} rollout queue empty, sleeping.")
                time.sleep(1.0)
                continue
            except Exception as e:
                log.error(f"RefActor {self.actor_id} error getting from queue: {e}")
                time.sleep(1.0)
                continue

            time_wait_end = time.perf_counter()
            time_waiting_buffer = time_wait_end - time_step_start
            trajectories = grouped_traj.trajectories

            #TODO: need to turn the queue into a generator and sample N trajectories
            # checking for total_tokens, until sum(total_tokens) > target_tokens_per_batch
            # then pack and pass to model, and keep the last bach that didnt fit
            # as the started for the next one
            # it would be an iterative dataset that yields a batch
            packed_trajectory = pack_sequences(
                sequences=trajectories,
                device=self._device,
                pad_token_id=self._tokenizer.pad_id,
                target_tokens_per_batch=self.cfg.target_tokens_per_batch
            )

            #TOOD: need position ids here
            with torch.no_grad():
                time_model_start = time.perf_counter()
                ref_logits = self._ref_model(
                    packed_trajectory.tokens,
                    mask=packed_trajectory.attention_mask,
                    position_ids=packed_trajectory.position_ids
                )
                time_model_running = time.perf_counter() - time_model_start
                
                # get logprobs by calculating CrossEntropyLoss only for the response tokens
                rep_response_logprobs = self._loss_fn(
                    ref_logits[packed_trajectory.response_mask],
                    packed_trajectory.target_tokens,
                )
                del ref_logits
    
                # Get a list of logprobs
                ref_logprobs_list = torch.split(rep_response_logprobs.cpu(), packed_trajectory.response_lens)
                del rep_response_logprobs

                # Get rewards and successes
                rewards_by_fn, successes_by_fn, reward_metadata = batched_rewards(
                    tokenizer=self._tokenizer, 
                    completions=[t.response_tokens for t in trajectories], 
                    answers=[t.answer for t in trajectories], 
                    device='cpu'
                )

                def compute_advantages(packed_trajectory: PackedTrajectory, rewards_by_fn: torch.Tensor):
                    #TODO: this relies a lot on order of trajectories in the batch
                    # This may be safe, but its a big assumption and may break silenly.
                    
                    assert rewards_by_fn.shape[0] % self.cfg.grpo_samples == 0, "rewards_by_fn must have shape[0] divisible by grpo_samples"

                    group_indices = packed_trajectory.group_indices
                    group_counts = self.cfg.grpo_samples
                    num_groups = rewards_by_fn.shape[0] // group_counts

                    # Compute per-group mean and std using scatter operations
                    rewards_sum = torch.zeros(num_groups, device=rewards_by_fn.device).scatter_add_(
                        0, group_indices, rewards_by_fn.sum(-1)
                    )
                    mean_rewards = rewards_sum / group_counts

                    # Compute variance in one pass
                    rewards_diff = rewards_by_fn.sum(-1) - mean_rewards[group_indices]
                    rewards_var_sum = torch.zeros(num_groups, device=rewards_by_fn.device).scatter_add_(
                        0, group_indices, rewards_diff ** 2
                    )
                    std_rewards = torch.sqrt(rewards_var_sum / group_counts).clamp(min=1e-4)

                    # Compute advantages
                    advantages = (rewards_by_fn.sum(-1) - mean_rewards[group_indices]) / std_rewards[group_indices]
                    return advantages

                advantages = compute_advantages(packed_trajectory, rewards_by_fn)

            # Prepare trajectories for replay buffer
            scored_trajectories_cpu = []
            for i, traj in enumerate(trajectories):
                scored_traj = ScoredTrajectory(
                    prompt_tokens=traj.prompt_tokens,
                    response_tokens=traj.response_tokens,
                    prompt_len=traj.prompt_len,
                    response_len=traj.response_len,
                    logprobs=traj.logprobs,
                    ref_logprobs=ref_logprobs_list[i],
                    rewards=rewards_by_fn[i],
                    advantages=advantages[i],
                    successes=successes_by_fn[i],
                    answer=traj.answer,
                    sequence_id=traj.sequence_id,
                    group_id=traj.group_id,
                    reward_metadata=reward_metadata,
                    policy_version=traj.policy_version
                )
                scored_trajectories_cpu.append(scored_traj)

            self.replay_buffer.extend(scored_trajectories_cpu)

            time_total_ref_step = time.perf_counter() - time_step_start
            number_of_tokens = sum(traj.response_len for traj in trajectories)

            if self._is_actor_zero:
                self._log_metrics(
                    step_idx=idx,
                    grouped_traj=grouped_traj,
                    time_total_ref_step=time_total_ref_step,
                    time_model_running=time_model_running,
                    time_waiting_buffer=time_waiting_buffer,
                    rollout_queue_size=rollout_queue_size,
                    rewards_mean=rewards_by_fn.mean(),
                    successes_mean=successes_by_fn.mean(),
                    rewards_mean_per_func=rewards_by_fn.mean(dim=0),
                    successes_mean_per_func=successes_by_fn.mean(dim=0),
                    reward_metadata=reward_metadata,
                    number_of_tokens=number_of_tokens
                )

            idx += 1

        log.info(f"RefActor {self.actor_id} completed run loop after {idx} steps.")
