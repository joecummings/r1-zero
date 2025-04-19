# Implementing Packing for GRPO Pipeline# Implementing Packing for GRPO Pipeline

## Goals

1. Implement dynamic sequence packing based on **target token count per batch** within `RefActor` and `PyTorchActorModel` using standard Python `@dataclass` objects, replacing static padding/concatenation for improved efficiency (memory and computation). **Packing involves concatenating sequences**, not padding them individually to a batch max length.
2. Refine data structures and data flow (`SyncLLMCollector` -> `RefActor` -> `ReplayBuffer` -> `PyTorchActorModel`) to support data packing, storing unpacked, processed sequence data (`ScoredTrajectory`) in the replay buffer. **One dataclass instance represents one sequence sample**, except for `GroupedUnscoredTrajectories` and `PackedTrajectory`.
3. Ensure correctness of reference logprob calculation, reward computation, and advantage mapping, considering that advantages for a group are computed only after all sequences in that group are processed by `RefActor`.
4. Maintain compatibility with GRPO loss calculation (`GRPOWithChunkedOutputLoss`), which operates on packed batches (`PackedTrajectory`) generated dynamically by `PyTorchActorModel`.
5. Utilize separate prompt and response tokens (`prompt_tokens`, `response_tokens`) in trajectory objects, relying on `response_len` for length information. Concatenation happens *during* packing.

## Key Considerations

- **Dataclasses over TensorClass:** Standard Python `@dataclass` will be used for data representation.

- **Sequence Representation:** Each `UnscoredTrajectory` or `ScoredTrajectory` instance represents a *single sequence* with separate prompt and response tokens.

- **Token-Based Batching:** `RefActor` and ` masking.
  PyTorchActorModel` pack sequences dynamically to meet a `target_tokens_per_batch` limit. Packing involves concatenating sequences into tensors like `PackedTrajectory.tokens`.

- **Packing Strategy: Unpad -> Concatenate -> Pad (if needed):**

  1. Individual sequences (prompt, response) are stored *unpadded*.
  2. During packing, the prompt and response of *each* sequence are concatenated (`[P1, R1]`, `[P2, R2]`, ...).
  3. These full sequences are then concatenated horizontally (`[P1, R1, P2, R2, ...]`).
  4. If the total length is less than `target_tokens_per_batch`, pad the last response to avoid recreating the large `tokens` tensor.

- **Distributed Packing & Group Handling:**

  - `SyncLLMCollector`: Produces `GroupedUnscoredTrajectories` for the `rollout_queue`.
  - `RefActor`: Processes entire groups atomically, splitting into multiple `PackedTrajectory` batches if needed. Stores individual `ScoredTrajectory` in the replay buffer.
  - `PyTorchActorModel`: Samples `ScoredTrajectory`, packs into `PackedTrajectory`, trains.

- **Data Flow:**

  - `SyncLLMCollector` -> `rollout_queue` \[`GroupedUnscoredTrajectories`\]
  - `RefActor` -> `replay_buffer` \[`ScoredTrajectory`\]
  - `PyTorchActorModel` (packs and trains).

- **Replay Buffer Content:** Stores individual, unpacked `ScoredTrajectory` instances.

- **Masking is Key:** Masks (`response_mask`, `sequence_mask`, `attention_mask`) generated during packing are critical for operations on concatenated tensors.

- **Response Lengths:** Tracked per sequence via `response_len`, excluding padding.

- **Loss Function:** `GRPOWithChunkedOutputLoss` uses `PackedTrajectory` with `response_mask` and pre-masked `targets` (shape: `total_response_tokens`).

- **Concatenated Tokens:** `PackedTrajectory.tokens` stores `[P1, R1, P2, R2, ..., Pn, Rn, PAD...]`.

- **Function Docstring Format:**

  ```python
  def fn(arg: type) -> type:
      """Description.
      
      Args:
          arg (type): Description.
      
      Returns:
          type: Description.
      
      Example:
          ...
      """
  ```

## Data Structures

```python
from dataclasses import dataclass
import torch
from typing import List, Dict, Optional

@dataclass
class UnscoredTrajectory:
    prompt_tokens: torch.Tensor  # Unpadded prompt tokens, shape: (prompt_len,)
    response_tokens: torch.Tensor  # Unpadded response tokens, shape: (response_len,)
    prompt_len: int  # Length of the prompt
    response_len: int  # Length of the response (unpadded)
    logprobs: torch.Tensor  # Logprobs from vLLM for response tokens, shape: (response_len,)
    answer: str  # Ground truth answer string
    sequence_id: str  # Unique ID for this sequence
    policy_version: int  # Policy version from vLLM worker
    group_id: int  # Unique group identifier, assigned in SyncLLMCollector

@dataclass
class GroupedUnscoredTrajectories:
    trajectories: List[UnscoredTrajectory]  # List of grpo_samples trajectories
    group_id: int  # Unique group identifier
    total_tokens: int  # Sum of (prompt_len + response_len) for all trajectories

@dataclass
class ScoredTrajectory:
    prompt_tokens: torch.Tensor  # Unpadded prompt tokens, shape: (prompt_len,)
    response_tokens: torch.Tensor  # Unpadded response tokens, shape: (response_len,)
    prompt_len: int  # Length of the prompt
    response_len: int  # Length of the response (unpadded)
    logprobs: torch.Tensor  # Logprobs from vLLM, shape: (response_len,)
    ref_logprobs: torch.Tensor  # Reference logprobs from RefActor, shape: (response_len,)
    rewards: torch.Tensor  # Per-sequence reward vector, shape: (num_funcs,)
    advantages: torch.Tensor  # Scalar per-sequence advantage
    successes: torch.Tensor  # Per-sequence success vector, shape: (num_funcs,)
    answer: str  # Ground truth answer string
    sequence_id: str  # Unique ID for this sequence
    group_id: int  # Unique group identifier
    reward_metadata: Dict  # Metadata from reward calculation (e.g., func_names)
    policy_version: int  # Policy version when sequence was generated

@dataclass
class PackedTrajectory:
    tokens: torch.Tensor  # Concatenated [P1, R1, P2, R2, ...], potentially padded, shape: (packed_seq_len,)
    attention_mask: torch.Tensor  # Block-diagonal causal mask, shape: (packed_seq_len, packed_seq_len)
    response_mask: torch.Tensor  # Boolean mask for response tokens, shape: (packed_seq_len,)
    sequence_mask: torch.Tensor  # Integer mask indicating sequence index, shape: (packed_seq_len,)
    prompt_lens: torch.Tensor  # Length of each prompt, shape: (num_sequences,)
    response_lens: torch.Tensor  # Length of each response, shape: (num_sequences,)
    sequence_map: torch.Tensor  # Start/end indices per sequence, shape: (num_sequences, 2)
    packed_seq_len: int  # Total length of tokens tensor
    ref_logprobs: Optional[torch.Tensor] = None  # Ref logprobs for responses, shape: (total_response_tokens,)
    advantages: Optional[torch.Tensor] = None  # Advantages for responses, shape: (total_response_tokens,)
    targets: Optional[torch.Tensor] = None  # Target tokens for loss, shape: (total_response_tokens,)
    sequence_ids: List[str] = None  # List of sequence IDs for logging
    group_ids: List[int] = None  # List of group IDs for logging
    actual_total_tokens: int = 0  # Sum of (prompt_len + response_len) without padding

@dataclass
class GRPOStats:
    loss: torch.Tensor  # Scalar, total loss averaged over response tokens
    policy_loss: torch.Tensor  # Scalar, policy loss component
    kl_loss: torch.Tensor  # Scalar, KL divergence component
    ratios: torch.Tensor  # Scalar, mean exponentiated logprob ratio
    clipfrac: torch.Tensor  # Scalar, fraction of clipped ratios
    approx_policy_kls: torch.Tensor  # Scalar, approximate KL divergence
    metadata: Optional[Dict[str, Any]] = None  # Optional debug data, e.g., pi_logprobs (total_response_tokens,)
```

## Worker Modifications

### 1. `SyncLLMCollector`

- `_postprocess_for_queue`:
  1. Receives padded `data` from vLLM.
  2. Extracts unpadded sequence data, creates `UnscoredTrajectory` instances.
  3. Groups by `group_id`, creates `GroupedUnscoredTrajectories` with `total_tokens`.
  4. Enqueues into `rollout_queue`.

```python
def _postprocess_for_queue(self, data):
    """Postprocesses vLLM output into grouped trajectories for the rollout queue.
    
    Args:
        data: TensorDict with vLLM output (tokens, log_probs, etc.).
    
    Returns:
        int: Total generated tokens across all sequences.
    
    Example:
        total_tokens = collector._postprocess_for_queue(data)
    """
    batch_size = data['tokens'].shape[0]
    sequence_ids = [f"worker{self.worker_id}_{self._sequence_counter + i}" for i in range(batch_size)]
    self._sequence_counter += batch_size
    
    trajectories = []
    for i in range(batch_size):
        prompt_tokens = data['tokens'][i]
        response_tokens = data['tokens_response'][i]
        logprobs = data['log_probs'][i]
        answer = data['answers'][i]
        policy_version = data['policy_version']
        
        prompt_len = (prompt_tokens != self._tokenizer.pad_id).sum().item()
        response_len = (response_tokens != self._tokenizer.pad_id).sum().item()
        prompt_tokens = prompt_tokens[:prompt_len]
        response_tokens = response_tokens[:response_len]
        logprobs = logprobs[:response_len]
        
        group_id = self.worker_id * self.cfg.vllm.batch_size + (i // self.grpo_samples)
        
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
        trajectories.append(traj)
    
    grouped_trajectories = {}
    for traj in trajectories:
        grouped_trajectories.setdefault(traj.group_id, []).append(traj)
    
    for group_id, traj_list in grouped_trajectories.items():
        total_tokens = sum(traj.prompt_len + traj.response_len for traj in traj_list)
        grouped_traj = GroupedUnscoredTrajectories(
            trajectories=traj_list,
            group_id=group_id,
            total_tokens=total_tokens
        )
        self.rollout_queue.put(grouped_traj)
    
    return sum(traj.response_len for traj in trajectories)
```

### 2. `RefActor`

- **Processing Logic:**
  - Consumes `GroupedUnscoredTrajectories`, packs into `PackedTrajectory`, processes batches, computes advantages per group, stores `ScoredTrajectory`.

```python
def run(self):
    """Main loop for processing grouped trajectories from the rollout queue.
    
    Args:
        None
    
    Returns:
        None
    
    Example:
        ref_actor.run()  # Runs indefinitely, processing queue items
    """
    import time
    
    idx = 0
    while idx < self.cfg.num_steps:
        time_step_start = time.perf_counter()
        
        grouped_traj = self.rollout_queue.get(timeout=0.5)
        trajectories = grouped_traj.trajectories
        
        time_waiting_buffer = time.perf_counter() - time_step_start
        
        packed_trajectory = pack_sequences(
            sequences=trajectories,
            device=self._device,
            pad_token_id=self._tokenizer.pad_id,
            target_tokens_per_batch=self.cfg.target_tokens_per_batch
        )
        
        time_model_start = time.perf_counter()
        with torch.no_grad():
            ref_logits = self._ref_model(
                packed_trajectory.tokens,
                mask=packed_trajectory.attention_mask
            )
        ref_logprobs_packed = compute_logprobs(ref_logits, packed_trajectory.tokens)
        ref_logprobs_list = unpack_response_logprobs(packed_trajectory, ref_logprobs_packed)
        time_model_running = time.perf_counter() - time_model_start
        
        responses = [traj.response_tokens for traj in trajectories]
        answers = [traj.answer for traj in trajectories]
        rewards_by_fn, successes_by_fn, reward_metadata = batched_rewards(
            self._tokenizer, responses, answers, device=self._device
        )
        
        group_rewards = rewards_by_fn.sum(-1)
        mean_reward = group_rewards.mean()
        std_reward = group_rewards.std() + 1e-4
        advantages = (group_rewards - mean_reward) / std_reward
        
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
            self.replay_buffer.add(scored_traj)
        
        time_total_ref_step = time.perf_counter() - time_step_start
        if self._is_actor_zero:
            rewards_mean = rewards_by_fn.mean()
            successes_mean = successes_by_fn.mean()
            self._log_metrics(
                step_idx=idx,
                grouped_traj=grouped_traj,
                time_total_ref_step=time_total_ref_step,
                time_model_running=time_model_running,
                time_waiting_buffer=time_waiting_buffer,
                rollout_queue_size=self.rollout_queue.qsize(),
                rewards_mean=rewards_mean,
                successes_mean=successes_mean,
                rewards_mean_per_func=rewards_by_fn.mean(dim=0),
                successes_mean_per_func=successes_by_fn.mean(dim=0),
                reward_metadata=reward_metadata
            )
        
        idx += 1
```

- **Updated** `_log_metrics`**:**

```python
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
    reward_metadata: Dict
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
    
    log_dict.update({
        "ref_actor_performance/total_rollout_time (s)": time_total_ref_step,
        "ref_actor_performance/time_model_running (s)": time_model_running,
        "ref_actor_performance/pct_time_model_running (%)": (
            time_model_running / time_total_ref_step * 100 if time_total_ref_step > 0 else 0
        ),
        "ref_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
        "ref_actor_performance/pct_time_waiting_buffer (%)": (
            time_waiting_buffer / time_total_ref_step * 100 if time_total_ref_step > 0 else 0
        ),
        "ref_actor_performance/total_tokens_processed": grouped_traj.total_tokens,
        "queues/rollout_queue_size": rollout_queue_size,
        "ref_actor_rewards/rewards_mean": rewards_mean.item(),
        "ref_actor_rewards/successes_mean": successes_mean.item(),
    })
    
    for func_name, r_mean, s_mean in zip(
        reward_metadata["func_names"], rewards_mean_per_func, successes_mean_per_func
    ):
        log_dict[f"ref_actor_rewards/rewards_func_{func_name}_mean"] = r_mean.item()
        log_dict[f"ref_actor_rewards/successes_func_{func_name}_mean"] = s_mean.item()
    
    ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))
```

### 3. `PyTorchActorModel`

- **Updated** `grpo_step`**:**

```python
def grpo_step(self, packed_trajectory: PackedTrajectory) -> GRPOStats:
    """Perform a single GRPO optimization step on a packed trajectory.
    
    Args:
        packed_trajectory (PackedTrajectory): The packed batch of sequences containing tokens, masks, and training data.
    
    Returns:
        GRPOStats: Statistics from the GRPO step including loss and metrics.
    
    Example:
        stats = actor.grpo_step(packed_trajectory)
    """
    import torch.nn.functional as F
    
    with self.activations_handling_ctx:
        pi_logits = self._model(
            packed_trajectory.tokens,
            mask=packed_trajectory.attention_mask,
        )
    
    # Compute log probabilities for all positions up to the second-to-last token
    pi_logprobs_all = F.log_softmax(pi_logits[:-1], dim=-1).gather(
        1, packed_trajectory.tokens[1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # Select logprobs where the next token is a response token
    response_mask_shifted = packed_trajectory.response_mask[1:]
    pi_logprobs_response = pi_logprobs_all[response_mask_shifted]
    
    # Extract reference logprobs and advantages
    ref_logprobs = packed_trajectory.ref_logprobs
    advantages = packed_trajectory.advantages
    
    # Compute loss using a modified loss function that accepts precomputed logprobs
    loss, policy_loss, kl_loss = self._loss_fn.compute_loss_from_logprobs(
        pi_logprobs_response,
        ref_logprobs,
        advantages,
    )
    
    # Compute additional statistics
    with torch.no_grad():
        ratios = torch.exp(pi_logprobs_response - pi_logprobs_response.detach())
        clipfrac = (ratios > 1.0).float().mean()
        approx_policy_kls = 0.5 * ((pi_logprobs_response - ref_logprobs).pow(2)).mean()
    
    stats = GRPOStats(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        ratios=ratios.mean(),
        clipfrac=clipfrac,
        approx_policy_kls=approx_policy_kls,
        metadata={"pi_logprobs": pi_logprobs_response.detach()} if self.debug_logging_enabled else None,
    )
    
    loss.backward()
    return stats
```

- **Updated** `_log_metrics`**:**

```python
def _log_metrics(
    self,
    step_idx: int,
    trajectory: List[ScoredTrajectory],
    packed_trajectory: PackedTrajectory,
    grpo_stats: List[GRPOStats],
    total_step_time: float,
    time_grpo_steps: float,
    time_waiting_buffer: float,
    time_weight_sync: float,
    time_weight_gather: float,
    train_replay_buffer_size: int,
) -> None:
    """Log training metrics, only on rank zero.
    
    Args:
        step_idx (int): Current training step index.
        trajectory (List[ScoredTrajectory]): Original list of scored trajectories.
        packed_trajectory (PackedTrajectory): Packed trajectory used for training.
        grpo_stats (List[GRPOStats]): List of GRPO statistics from PPO epochs.
        total_step_time (float): Total time for the training step.
        time_grpo_steps (float): Time spent on GRPO steps.
        time_waiting_buffer (float): Time spent waiting for buffer.
        time_weight_sync (float): Time spent syncing weights.
        time_weight_gather (float): Time spent gathering weights.
        train_replay_buffer_size (int): Size of the replay buffer.
    
    Returns:
        None
    
    Example:
        actor._log_metrics(1, traj_list, packed_traj, stats, 2.0, 1.0, 0.5, 0.3, 0.2, 100)
    """
    if not self._is_rank_zero:
        return
    
    # Compute token metrics
    actual_total_tokens = sum(traj.prompt_len + traj.response_len for traj in trajectory)
    number_of_tokens = sum(traj.response_len for traj in trajectory)
    padded_tokens = packed_trajectory.packed_seq_len - actual_total_tokens
    padded_tokens_percentage = (
        (padded_tokens / packed_trajectory.packed_seq_len * 100)
        if packed_trajectory.packed_seq_len > 0 else 0
    )
    
    # Aggregate stats across PPO epochs
    grpo_stats_stacked = GRPOStats(
        loss=torch.stack([s.loss for s in grpo_stats]).mean(),
        policy_loss=torch.stack([s.policy_loss for s in grpo_stats]).mean(),
        kl_loss=torch.stack([s.kl_loss for s in grpo_stats]).mean(),
        ratios=torch.stack([s.ratios for s in grpo_stats]).mean(),
        clipfrac=torch.stack([s.clipfrac for s in grpo_stats]).mean(),
        approx_policy_kls=torch.stack([s.approx_policy_kls for s in grpo_stats]).mean(),
    )
    
    log_dict = {}
    if self._log_peak_memory_stats:
        memory_stats = training.get_memory_stats(device=self._device)
        log_dict.update(
            {f"train_actor_performance/memory/{k}": v for k, v in memory_stats.items()}
        )
    
    log_dict.update({
        "train_actor_training/loss": grpo_stats_stacked.loss.item(),
        "train_actor_training/policy_loss": grpo_stats_stacked.policy_loss.item(),
        "train_actor_training/kl_loss": grpo_stats_stacked.kl_loss.item(),
        "train_actor_training/ratios": grpo_stats_stacked.ratios.item(),
        "train_actor_training/clipfrac": grpo_stats_stacked.clipfrac.item(),
        "train_actor_training/approx_policy_kls": grpo_stats_stacked.approx_policy_kls.item(),
        "train_actor_training/response_lengths": torch.tensor(
            [traj.response_len for traj in trajectory]
        ).float().mean().item(),
    })
    
    log_dict.update({
        "train_actor_performance/total_step_time (s)": total_step_time,
        "train_actor_performance/time_grpo_steps (s)": time_grpo_steps,
        "train_actor_performance/pct_time_grpo_steps (%)": (
            time_grpo_steps / total_step_time * 100 if total_step_time > 0 else 0
        ),
        "train_actor_performance/tokens_per_second": (
            number_of_tokens / total_step_time if total_step_time > 0 else 0
        ),
        "train_actor_performance/time_weight_sync (s)": time_weight_sync,
        "train_actor_performance/pct_time_weight_sync (%)": (
            time_weight_sync / total_step_time * 100 if total_step_time > 0 else 0
        ),
        "train_actor_performance/padded_tokens_percentage (%)": padded_tokens_percentage,
        "train_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
        "train_actor_performance/pct_time_waiting_buffer (%)": (
            time_waiting_buffer / total_step_time * 100 if total_step_time > 0 else 0
        ),
        "train_actor_performance/time_weight_gather (s)": time_weight_gather,
        "train_actor_performance/pct_time_weight_gather (%)": (
            time_weight_gather / total_step_time * 100 if total_step_time > 0 else 0
        ),
        "queues/train_replay_buffer_size": train_replay_buffer_size,
    })
    
    ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))
```

- **Updated** `_log_debug_table`**:**

```python
def _log_debug_table(
    self,
    trajectory: List[ScoredTrajectory],
    packed_trajectory: PackedTrajectory,
    grpo_stats: GRPOStats,
) -> None:
    """Log debugging tables with per-token and per-sample features.
    
    Args:
        trajectory (List[ScoredTrajectory]): Original list of scored trajectories.
        packed_trajectory (PackedTrajectory): Packed trajectory used for training.
        grpo_stats (GRPOStats): GRPO statistics from the step.
    
    Returns:
        None
    
    Example:
        actor._log_debug_table(traj_list, packed_traj, stats)
    """
    if not self._is_rank_zero or not self.debug_logging_enabled:
        return
    
    pi_logprobs_response = grpo_stats.metadata.get("pi_logprobs")
    if pi_logprobs_response is None:
        return
    
    response_lens = [traj.response_len for traj in trajectory]
    pi_logprobs_list = torch.split(pi_logprobs_response, response_lens)
    
    per_sample_table_data = []
    per_token_table_data = []
    
    for idx, traj in enumerate(trajectory[:self.debug_num_samples_per_step]):
        seq_id = traj.sequence_id
        prompt = self._tokenizer.decode(traj.prompt_tokens.tolist(), skip_special_tokens=False)
        response = self._tokenizer.decode(traj.response_tokens.tolist(), skip_special_tokens=False)
        ref_logprobs = traj.ref_logprobs.tolist()
        logprobs = traj.logprobs.tolist()
        pi_logprobs = pi_logprobs_list[idx].tolist()
        response_tokens = traj.response_tokens.tolist()
        decoded_tokens = [
            self._tokenizer.decode([token], skip_special_tokens=False)
            for token in response_tokens
        ]
        
        # Per-sample data
        per_sample = {
            "Sequence ID": seq_id,
            "Prompt": prompt,
            "Response": response,
            "Advantages": traj.advantages.item(),
            "Response Length": traj.response_len,
            "Step": self._steps_run,
        }
        for fname, reward, success in zip(
            traj.reward_metadata["func_names"], traj.rewards, traj.successes
        ):
            per_sample[f"Reward_{fname}"] = reward.item()
            per_sample[f"Success_{fname}"] = success.item()
        per_sample.update({
            attr: getattr(grpo_stats, attr).item()
            for attr in ["loss", "policy_loss", "kl_loss", "ratios", "clipfrac", "approx_policy_kls"]
        })
        per_sample_table_data.append(per_sample)
        
        # Per-token data
        for pos in range(traj.response_len):
            per_token = {
                "Sequence ID": seq_id,
                "Token Position": pos,
                "Token ID": response_tokens[pos],
                "Decoded Token": decoded_tokens[pos],
                "Generated Logprob": logprobs[pos],
                "Ref Logprob": ref_logprobs[pos],
                "Pi Logprob": pi_logprobs[pos],
                "Abs Diff Pi-Ref": abs(pi_logprobs[pos] - ref_logprobs[pos]),
                "Abs Diff Pi-Generated": abs(pi_logprobs[pos] - logprobs[pos]),
                "Step": self._steps_run,
            }
            per_token_table_data.append(per_token)
    
    if per_sample_table_data:
        ray.get(self._metric_logger.log_table.remote(
            [list(row.values()) for row in per_sample_table_data],
            list(per_sample_table_data[0].keys()),
            "per_sample_debug_table",
            step=self._steps_run
        ))
    if per_token_table_data:
        ray.get(self._metric_logger.log_table.remote(
            [list(row.values()) for row in per_token_table_data],
            list(per_token_table_data[0].keys()),
            "per_token_debug_table",
            step=self._steps_run
        ))
```

## Helper Functions

```python
import torch
import torchtune.modules

def pack_sequences(
    sequences: List[Union[UnscoredTrajectory, ScoredTrajectory]],
    device: torch.device,
    pad_token_id: int,
    ignore_index: int = -100,
    target_tokens_per_batch: Optional[int] = None
) -> PackedTrajectory:
    """Packs sequences into a single PackedTrajectory, padding the last response if needed.
    
    Args:
        sequences (List[Union[UnscoredTrajectory, ScoredTrajectory]]): Sequences to pack.
        device (torch.device): Device for packed tensors.
        pad_token_id (int): Token ID for padding.
        ignore_index (int): Index for masking targets in loss.
        target_tokens_per_batch (Optional[int]): Target token count for padding.
    
    Returns:
        PackedTrajectory: Packed batch of sequences.
    
    Example:
        packed = pack_sequences(sequences, 'cuda', 0, target_tokens_per_batch=4096)
    """
    if not sequences:
        raise ValueError("Cannot pack an empty list of sequences.")
    
    num_sequences = len(sequences)
    is_scored = isinstance(sequences[0], ScoredTrajectory)
    
    prompt_lens = torch.tensor([s.prompt_len for s in sequences], device=device, dtype=torch.long)
    response_lens = torch.tensor([s.response_len for s in sequences], device=device, dtype=torch.long)
    total_lens = prompt_lens + response_lens
    packed_seq_len = total_lens.sum().item()
    actual_total_tokens = packed_seq_len
    
    # Concatenate Tokens
    all_tokens_list = [torch.cat([s.prompt_tokens, s.response_tokens]) for s in sequences[:-1]]
    last_seq = sequences[-1]
    last_tokens = torch.cat([last_seq.prompt_tokens, last_seq.response_tokens])
    
    # Pad last response if necessary
    if target_tokens_per_batch and packed_seq_len < target_tokens_per_batch:
        pad_count = target_tokens_per_batch - packed_seq_len
        padding = torch.full((pad_count,), pad_token_id, device=device, dtype=torch.long)
        last_tokens = torch.cat([last_tokens, padding])
        response_lens[-1] += pad_count
        total_lens[-1] += pad_count
        packed_seq_len = target_tokens_per_batch
    
    tokens = torch.cat(all_tokens_list + [last_tokens]).to(device=device, dtype=torch.long)
    
    # Sequence Map
    cumsum_lens = total_lens.cumsum(dim=0)
    sequence_map = torch.zeros((num_sequences, 2), device=device, dtype=torch.long)
    sequence_map[1:, 0] = cumsum_lens[:-1]
    sequence_map[:, 1] = cumsum_lens
    
    # Attention Mask
    attention_mask = torchtune.modules.create_packed_block_causal_mask(total_lens).to(device)
    
    # Response Mask
    response_mask = torch.zeros(packed_seq_len, dtype=torch.bool, device=device)
    for i, (start, end) in enumerate(sequence_map):
        response_mask[start + prompt_lens[i]:end] = True
    
    # Sequence Mask
    sequence_mask = torch.repeat_interleave(torch.arange(num_sequences, device=device), total_lens)
    
    # Optional Fields for Loss
    ref_logprobs_packed = advantages_packed = targets_packed = None
    if is_scored:
        total_response_tokens = response_lens.sum().item()
        ref_logprobs_packed = torch.cat([s.ref_logprobs for s in sequences]).to(device=device)
        advantages_packed = torch.cat([torch.full((s.response_len,), s.advantages, device=device) for s in sequences])
        targets_list = []
        for i, s in enumerate(sequences):
            start = sequence_map[i, 0].item()
            prompt_len = prompt_lens[i].item()
            response_len = s.response_len
            if response_len > 0:
                targets_list.append(tokens[start + prompt_len + 1:start + prompt_len + response_len])
        targets_packed = torch.cat(targets_list).to(device=device)
    
    sequence_ids = [s.sequence_id for s in sequences]
    group_ids = [s.group_id for s in sequences]
    
    return PackedTrajectory(
        tokens=tokens,
        attention_mask=attention_mask,
        response_mask=response_mask,
        sequence_mask=sequence_mask,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        sequence_map=sequence_map,
        packed_seq_len=packed_seq_len,
        ref_logprobs=ref_logprobs_packed,
        advantages=advantages_packed,
        targets=targets_packed,
        sequence_ids=sequence_ids,
        group_ids=group_ids,
        actual_total_tokens=actual_total_tokens
    )

def unpack_response_logprobs(
    packed_trajectory: PackedTrajectory,
    packed_logprobs: torch.Tensor
) -> List[torch.Tensor]:
    """Unpacks per-sequence response logprobs from a packed tensor using response lengths.
    
    Args:
        packed_trajectory (PackedTrajectory): Packed trajectory with response_lens.
        packed_logprobs (torch.Tensor): Packed logprobs, shape: (total_response_tokens,).
    
    Returns:
        List[torch.Tensor]: Per-sequence response logprobs, each of shape (response_len,).
    
    Example:
        logprobs_list = unpack_response_logprobs(packed_traj, packed_logprobs)
    """
    response_lens = packed_trajectory.response_lens.tolist()
    return torch.split(packed_logprobs, response_lens)
```

## Open Questions / TODOs

- **Multi-GPU/Multi-Node:** Ensure compatibility with distributed PyTorch utilities.
- **Performance Optimization:** Profile packing/unpacking, optimize if needed.
- **Testing:** Add unit and integration tests for the pipeline.
- **Vectorized Unpacking:** Explore using masks and scatter for efficient unpacking of logprobs.
- **Timing Concatenation:** Time efficiency of concatenating `ref_logprobs` vs. other methods.
- **Loss Function Adaptation:** Modify `GRPOWithChunkedOutputLoss` to include `compute_loss_from_logprobs` method as used in `grpo_step`.