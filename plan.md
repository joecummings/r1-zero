# Implementing Packing for GRPO Pipeline

## Goals

1. Implement dynamic sequence packing based on **target token count per batch** within `RefActor` and `PyTorchActorModel` using standard Python `@dataclass` objects, replacing static padding/concatenation for improved efficiency (memory and computation). **Packing involves concatenating sequences**, not padding them individually to a batch max length.
2. Refine data structures and data flow (`SyncLLMCollector` -> `RefActor` -> `ReplayBuffer` -> `PyTorchActorModel`) to support data packing, storing unpacked, processed sequence data (`ScoredTrajectory`) in the replay buffer. **One dataclass instance represents one sequence sample**, except for `GroupedUnscoredTrajectories` and `PackedTrajectory`.
3. Ensure correctness of reference logprob calculation, reward computation, and advantage mapping, considering that advantages for a group are computed only after all sequences in that group are processed by `RefActor`.
4. Maintain compatibility with GRPO loss calculation (e.g., `GRPOCKLDivLoss`), which operates on packed batches (`PackedTrajectory`) generated dynamically by `PyTorchActorModel`, potentially using chunked logits.
5. Utilize separate prompt and response tokens (`prompt_tokens`, `response_tokens`) in trajectory objects, relying on `response_len` for length information. Concatenation happens *during* packing.

## Key Considerations

- **Dataclasses over TensorClass:** Standard Python `@dataclass` will be used for data representation.
- **Sequence Representation:** Each `UnscoredTrajectory` or `ScoredTrajectory` instance represents a *single sequence* with separate prompt and response tokens.
- **Token-Based Batching:** `RefActor` and `PyTorchActorModel` pack sequences dynamically to meet a `target_tokens_per_batch` limit. Packing involves concatenating sequences into tensors like `PackedTrajectory.tokens`.
- **Packing Strategy: Unpad -> Concatenate -> Pad (if needed):**
  1. Individual sequences (prompt, response) are stored *unpadded*.
  2. During packing, the prompt and response of *each* sequence are concatenated (`[P1, R1]`, `[P2, R2]`, ...).
  3. These full sequences are then concatenated horizontally (`[P1, R1, P2, R2, ...]`).
  4. If the total length is less than `target_tokens_per_batch`, pad the *end* of the concatenated tensor.
- **Distributed Packing & Group Handling:**
  - `SyncLLMCollector`: Produces `GroupedUnscoredTrajectories` for the `rollout_queue`.
  - `RefActor`: Processes entire groups atomically. Stores individual `ScoredTrajectory` in the replay buffer.
  - `PyTorchActorModel`: Samples `ScoredTrajectory`, packs into `PackedTrajectory`, trains.
- **Data Flow:**
  - `SyncLLMCollector` -> `rollout_queue` \[`GroupedUnscoredTrajectories`\]
  - `RefActor` -> `replay_buffer` \[`ScoredTrajectory`\]
  - `PyTorchActorModel` (samples, packs, and trains).
- **Replay Buffer Content:** Stores individual, unpacked `ScoredTrajectory` instances with CPU tensors.
- **Masking is Key:** Masks (`response_mask`, `sequence_mask`, `attention_mask`) generated during packing are critical for operations on concatenated tensors.
- **Response Lengths:** Tracked per sequence via `response_len`, excluding padding.
- **Loss Function:** Uses `PackedTrajectory` with `response_mask` potentially applied implicitly via `targets`. `targets` should have shape `(total_response_tokens,)`.
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
from typing import List, Dict, Optional, Any, Union

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
    response_lens: torch.Tensor  # Length of each response (unpadded), shape: (num_sequences,)
    sequence_map: torch.Tensor  # Start/end indices per sequence in tokens, shape: (num_sequences, 2)
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
    ratios: torch.Tensor  # Scalar, mean exponentiated logprob ratio (pi_new / pi_old) - May be dummy in simplified loss
    clipfrac: torch.Tensor  # Scalar, fraction of clipped ratios - May be dummy in simplified loss
    approx_policy_kls: torch.Tensor  # Scalar, approximate KL divergence (pi_new || ref)
    metadata: Optional[Dict[str, Any]] = None  # Optional debug data, e.g., pi_logprobs (total_response_tokens,)
```

## GRPO Loss Function (Example Implementation)

This version handles chunked logits and calculates loss based on the TRL GRPO implementation, focusing on policy gradient and KL divergence from the reference model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class GRPOCKLDivLoss(nn.Module):
    """
    GRPO Loss calculation using KL divergence from a reference model and policy gradient.
    Handles chunked logits for memory efficiency.

    Args:
        kl_coeff (float): KL divergence coefficient (beta).
        num_output_chunks (int): Number of chunks the logits are split into.
        epsilon (float): Clipping parameter (currently unused in this simplified version).
    """
    def __init__(self, kl_coeff: float = 0.1, num_output_chunks: int = 1, epsilon: float = 0.1):
        super().__init__()
        self.kl_coeff = kl_coeff
        self.num_output_chunks = num_output_chunks
        self.epsilon = epsilon  # Unused for now

    def compute_per_token_quantities(
        self,
        pi_logits_chunk: torch.Tensor,  # [chunk_size, V]
        targets_chunk: torch.Tensor,    # [chunk_size]
        ref_logprobs_chunk: torch.Tensor,  # [chunk_size]
        advantages_chunk: torch.Tensor,    # [chunk_size]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes per-token loss components for a single chunk."""
        pi_logprobs_chunk = -F.cross_entropy(pi_logits_chunk.float(), targets_chunk, reduction="none")
        
        pi_logprobs_detached = pi_logprobs_chunk.detach()
        ref_logprobs_detached = ref_logprobs_chunk.detach()
        
        per_token_kl = (
            torch.exp(ref_logprobs_detached - pi_logprobs_chunk)
            - (ref_logprobs_detached - pi_logprobs_chunk)
            - 1
        )
        
        log_ratio = pi_logprobs_chunk - pi_logprobs_detached
        ratio = torch.exp(log_ratio)
        per_token_policy_loss = ratio * advantages_chunk
        
        per_token_loss = -(per_token_policy_loss - self.kl_coeff * per_token_kl)
        return per_token_loss, per_token_policy_loss, per_token_kl, pi_logprobs_chunk

    def forward(
        self,
        pi_logits: Union[torch.Tensor, List[torch.Tensor]],  # [total_response_tokens, V] or List[[chunk_size, V]]
        targets: torch.Tensor,              # [total_response_tokens]
        ref_logprobs: torch.Tensor,         # [total_response_tokens]
        advantages: torch.Tensor,           # [total_response_tokens]
        padding_masks: Optional[torch.Tensor] = None,  # Optional: [total_response_tokens]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the GRPO loss over potentially chunked logits. Expects pi_logits to correspond to response tokens only.

        Args:
            pi_logits: Logits for response tokens from the policy model.
            targets: Target token indices for response tokens.
            ref_logprobs: Reference log probabilities for the target tokens.
            advantages: Advantage values for the target tokens.
            padding_masks: Optional boolean mask (True for valid tokens). If None, all tokens are valid.

        Returns:
            Tuple containing:
                - loss: Final scalar loss.
                - policy_loss: Detached scalar policy loss component.
                - kl_loss: Detached scalar KL loss component.
                - ratios: Detached scalar ratio (dummy 1.0).
                - clipfrac: Detached scalar clip fraction (dummy 0.0).
                - pi_logprobs: Log probabilities of the policy for the target tokens.
        """
        if isinstance(pi_logits, torch.Tensor):
            if self.num_output_chunks != 1:
                log.warning(f"num_output_chunks is {self.num_output_chunks} but pi_logits is a single tensor.")
            pi_logits = [pi_logits]

        num_chunks = len(pi_logits)
        device = targets.device
        
        targets_chunks = targets.chunk(num_chunks)
        ref_logprobs_chunks = ref_logprobs.chunk(num_chunks)
        advantages_chunks = advantages.chunk(num_chunks)
        padding_masks_chunks = padding_masks.chunk(num_chunks) if padding_masks is not None else [None] * num_chunks
        
        total_loss_sum = torch.tensor(0.0, device=device)
        total_policy_sum = torch.tensor(0.0, device=device)
        total_kl_sum = torch.tensor(0.0, device=device)
        total_valid_tokens = torch.tensor(0.0, device=device)
        pi_logprobs_list = []
        
        for chunk_idx in range(num_chunks):
            (
                per_token_loss_chunk,
                per_token_policy_loss_chunk,
                per_token_kl_chunk,
                pi_logprobs_chunk,
            ) = self.compute_per_token_quantities(
                pi_logits[chunk_idx],
                targets_chunks[chunk_idx],
                ref_logprobs_chunks[chunk_idx],
                advantages_chunks[chunk_idx],
            )
            
            padding_mask_chunk = padding_masks_chunks[chunk_idx]
            if padding_mask_chunk is not None:
                num_valid_in_chunk = padding_mask_chunk.sum()
                total_loss_sum += (per_token_loss_chunk * padding_mask_chunk).sum()
                total_policy_sum += (per_token_policy_loss_chunk * padding_mask_chunk).sum()
                total_kl_sum += (per_token_kl_chunk * padding_mask_chunk).sum()
            else:
                num_valid_in_chunk = per_token_loss_chunk.numel()
                total_loss_sum += per_token_loss_chunk.sum()
                total_policy_sum += per_token_policy_loss_chunk.sum()
                total_kl_sum += per_token_kl_chunk.sum()
            
            total_valid_tokens += num_valid_in_chunk
            pi_logprobs_list.append(pi_logprobs_chunk)
        
        pi_logprobs = torch.cat(pi_logprobs_list)
        mean_loss = total_loss_sum / total_valid_tokens.clamp(min=1)
        mean_policy_loss = total_policy_sum / total_valid_tokens.clamp(min=1)
        mean_kl_loss = total_kl_sum / total_valid_tokens.clamp(min=1)
        
        ratios = torch.tensor(1.0, device=device)
        clipfrac = torch.tensor(0.0, device=device)
        
        return mean_loss, mean_policy_loss.detach(), mean_kl_loss.detach(), ratios, clipfrac, pi_logprobs.detach()
```

## Worker Modifications

### 1. `SyncLLMCollector`

- `_postprocess_for_queue`:
  1. Receives padded `data` from vLLM.
  2. Extracts unpadded sequence data, creates `UnscoredTrajectory` instances with CPU tensors.
  3. Groups by `group_id`, creates `GroupedUnscoredTrajectories` with `total_tokens`.
  4. Enqueues into `rollout_queue`.

```python
def _postprocess_for_queue(self, data):
    """Postprocesses vLLM output into grouped trajectories for the rollout queue.

    Args:
        data: TensorDict with vLLM output (tokens, log_probs, etc.).

    Returns:
        tuple: (postprocessed_results, total_generated_tokens)
            - postprocessed_results: List of GroupedUnscoredTrajectories enqueued.
            - total_generated_tokens: Total number of response tokens generated.

    Example:
        grouped_traj_list, total_tokens = collector._postprocess_for_queue(data)
    """
```

### 2. `RefActor`

- **Processing Logic:**
  - Consumes `GroupedUnscoredTrajectories`, packs into `PackedTrajectory`, processes batches, computes advantages per group, stores `ScoredTrajectory`.

```python
def _move_to_device(self, grouped_traj: GroupedUnscoredTrajectories) -> GroupedUnscoredTrajectories:
    """Move GroupedUnscoredTrajectories to the actor's device.

    Args:
        grouped_traj (GroupedUnscoredTrajectories): Grouped trajectories from queue with CPU tensors.

    Returns:
        GroupedUnscoredTrajectories: Same object with tensors on the actor's device.
    """
    for traj in grouped_traj.trajectories:
        traj.prompt_tokens = traj.prompt_tokens.to(self._device)
        traj.response_tokens = traj.response_tokens.to(self._device)
        traj.logprobs = traj.logprobs.to(self._device)
    return grouped_traj

def run(self):
    """Main loop for processing grouped trajectories from the rollout queue.

    Args:
        None

    Returns:
        None

    Example:
        ref_actor.run()
    """

```

- **Updated `_log_metrics`:**

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

    tokens_per_second = number_of_tokens / time_model_running if time_model_running > 0 else 0

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
```

### 3. `PyTorchActorModel`

- **Updated `_prepare_trajectory`:**

```python
def _prepare_trajectory(self, sampled_trajectories: List[ScoredTrajectory]) -> PackedTrajectory:
    """Prepares a packed trajectory for training from sampled trajectories.

    Args:
        sampled_trajectories (List[ScoredTrajectory]): List of trajectories sampled from replay buffer with CPU tensors.

    Returns:
        PackedTrajectory: Packed trajectory ready for model training on the actor's device.

    Example:
        packed_traj = actor._prepare_trajectory(trajectories)
    """
    target_tokens_per_batch = self.cfg.target_tokens_per_batch

    packed_trajectory = pack_sequences(
        sequences=sampled_trajectories,
        device=self._device,
        pad_token_id=self._tokenizer.pad_id,
        target_tokens_per_batch=target_tokens_per_batch
    )

    return packed_trajectory
```

- **Updated `grpo_step`:**

```python
def grpo_step(self, packed_trajectory: PackedTrajectory) -> GRPOStats:
    """Perform a single GRPO optimization step on a packed trajectory.

    Args:
        packed_trajectory (PackedTrajectory): Packed batch of sequences with training data.

    Returns:
        GRPOStats: Statistics from the GRPO step including loss and metrics.

    Example:
        stats = actor.grpo_step(packed_trajectory)
    """
    with self.activations_handling_ctx:
        pi_logits = self._model(
            packed_trajectory.tokens,
            mask=packed_trajectory.attention_mask,
        )  # Shape: [packed_seq_len, V]

    # Extract logits for response tokens
    relevant_logits = []
    num_sequences = len(packed_trajectory.sequence_map)
    for i in range(num_sequences):
        start, end = packed_trajectory.sequence_map[i]
        prompt_len = packed_trajectory.prompt_lens[i]
        response_len = packed_trajectory.response_lens[i]
        if response_len > 0:
            logits_start = start + prompt_len - 1
            logits_end = start + prompt_len + response_len - 1
            relevant_logits.append(pi_logits[logits_start:logits_end])
    if relevant_logits:
        relevant_logits = torch.cat(relevant_logits)  # Shape: [total_response_tokens, V]
    else:
        relevant_logits = torch.empty(0, pi_logits.size(-1), device=pi_logits.device)

    loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs_response = self._loss_fn(
        pi_logits=relevant_logits,
        targets=packed_trajectory.targets,
        ref_logprobs=packed_trajectory.ref_logprobs,
        advantages=packed_trajectory.advantages,
    )

    with torch.no_grad():
        approx_policy_kls = (
            0.5 * ((pi_logprobs_response - packed_trajectory.ref_logprobs).pow(2))
        ).mean()

    stats = GRPOStats(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        ratios=ratios,
        clipfrac=clipfrac,
        approx_policy_kls=approx_policy_kls,
        metadata={"pi_logprobs": pi_logprobs_response.detach()} if self.debug_logging_enabled else None,
    )

    loss.backward()
    return stats
```

- **Updated `_log_metrics`:**

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
```

- **Updated `_log_debug_table`:**

```python
def _log_debug_table(
    self,
    trajectory: List[ScoredTrajectory],
    packed_trajectory: PackedTrajectory,
    grpo_stats: GRPOStats,
) -> None:

```

## Helper Functions

```python
import torch
import torchtune.modules
from typing import List, Union, Optional
import torch.nn.functional as F

def pack_sequences(
    sequences: List[Union[UnscoredTrajectory, ScoredTrajectory]],
    device: torch.device,
    pad_token_id: int,
    target_tokens_per_batch: Optional[int] = None,
    ignore_index: int = -100
) -> PackedTrajectory:
    """Packs Trajectories into a single PackedTrajectory without padding in between sequences.

    Args:
        sequences (List[Union[UnscoredTrajectory, ScoredTrajectory]]): Sequences to pack with CPU tensors.
        device (torch.device): Target device for packed tensors.
        pad_token_id (int): Token ID for padding.
        target_tokens_per_batch (Optional[int]): Target token count for padding; if None, no padding added.
        ignore_index (int): Used when generating targets

    Returns:
        PackedTrajectory: Packed batch of sequences on the specified device.

    Example:
        packed = pack_sequences(sequences, 'cuda', 0, target_tokens_per_batch=4096)
    """
    if not sequences:
        raise ValueError("Cannot pack an empty list of sequences.")

    num_sequences = len(sequences)
    is_scored = isinstance(sequences[0], ScoredTrajectory)
    
    # Calculate lengths
    prompt_lens = torch.tensor([s.prompt_len for s in sequences], dtype=torch.long)
    response_lens = torch.tensor([s.response_len for s in sequences], dtype=torch.long)
    total_lens = prompt_lens + response_lens
    packed_seq_len = total_lens.sum().item()
    actual_total_tokens = packed_seq_len
    
    // Start of Selection
    # Aggregate all prompts and responses in order: P1, R1, P2, R2, ...
    tokens_list = []
    for s in sequences:
        tokens_list.append(s.prompt_tokens)
        tokens_list.append(s.response_tokens)
    # Pop and pad the last response if necessary
    if target_tokens_per_batch and packed_seq_len < target_tokens_per_batch:
        pad_count = target_tokens_per_batch - packed_seq_len
        padding = torch.full((pad_count,), pad_token_id, dtype=torch.long)
        tokens_list.append(padding)

        #TODO: this may be wrong
        response_lens[-1] += pad_count
        total_lens[-1] += pad_count
        packed_seq_len = target_tokens_per_batch

    # Concatenate all pieces in a single call
    tokens = torch.cat(tokens_list, dim=0).to(device=device, dtype=torch.long)
    
    # Create sequence map
    cumsum_lens = total_lens.cumsum(dim=0)
    sequence_map = torch.zeros((num_sequences, 2), dtype=torch.long)
    sequence_map[1:, 0] = cumsum_lens[:-1]
    sequence_map[:, 1] = cumsum_lens
    sequence_map = sequence_map.to(device)
    
    # Create masks
    attention_mask = torchtune.modules.create_packed_block_causal_mask(total_lens)
    sequence_mask = torch.repeat_interleave(torch.arange(num_sequences, device=device), total_lens)
    response_mask = torch.zeros(packed_seq_len, dtype=torch.bool, device=device)
    for i, (start, end) in enumerate(sequence_map):
        response_mask[start + prompt_lens[i]:end] = True

    # create position ids
    position_ids = #TODO: efficienlty create torch.arange 0 to total_lens to every sample

    # Pack additional fields for scored trajectories
    ref_logprobs = advantages = targets = None
    if is_scored:
        ref_logprobs = torch.cat([s.ref_logprobs for s in sequences]).to(device)
        adv_values = torch.stack([s.advantages for s in sequences]).to(device=device)
        repeats = response_lens.to(device=device)
        advantages = torch.repeat_interleave(adv_values, repeats)

        # -- Create targets --
        # Extract response tokens and shift them to create targets
        targets = tokens[response_mask]
        targets[:-1] = targets[1:]  # Shift: token N predicts token N+1

        # Create mask to replace the last token of each sequence with ignore_index
        last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=device)
        for i, (start, end) in enumerate(sequence_map):
            if response_lens[i] > 0:
                last_token_mask[end - 1] = True
        
        # Replace
        targets[last_tokens_indices] = ignore_index 

    return PackedTrajectory(
        tokens=tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        response_mask=response_mask,
        sequence_mask=sequence_mask,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        sequence_map=sequence_map,
        packed_seq_len=packed_seq_len,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        targets=targets,
        sequence_ids=[s.sequence_id for s in sequences],
        group_ids=[s.group_id for s in sequences],
        actual_total_tokens=actual_total_tokens
    )

def unpack_response_logprobs(
    packed_trajectory: PackedTrajectory,
    packed_logprobs: torch.Tensor
) -> List[torch.Tensor]:
    """Unpacks per-sequence response logprobs from a packed tensor using response lengths.

    Args:
        packed_trajectory (PackedTrajectory): Packed trajectory with response_lens.
        packed_logprobs (torch.Tensor): Packed logprobs for response tokens, shape: (total_response_tokens,).

    Returns:
        List[torch.Tensor]: Per-sequence response logprobs, each of shape (response_len,).

    Example:
        logprobs_list = unpack_response_logprobs(packed_traj, packed_response_logprobs)
    """
    response_lens_list = packed_trajectory.response_lens.tolist()
    return torch.split(packed_logprobs, response_lens_list)

def compute_response_logprobs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    sequence_map: torch.Tensor,
    prompt_lens: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Computes log probabilities for the response tokens within a packed sequence tensor.

    Args:
        logits (torch.Tensor): Raw logits output from the model, shape: (packed_seq_len, V).
        tokens (torch.Tensor): Packed token sequence tensor, shape: (packed_seq_len,).
        sequence_map (torch.Tensor): Mapping sequence index to start/end positions, shape: (num_sequences, 2).
        prompt_lens (torch.Tensor): Length of each prompt, shape: (num_sequences,).
        temperature (float): Temperature for logsoftmax scaling.

    Returns:
        torch.Tensor: Flat tensor of log probabilities for response tokens, shape: (total_response_tokens,).

    Example:
        ref_logprobs = compute_response_logprobs(logits, tokens, sequence_map, prompt_lens)
    """
    logprobs_list = []
    for i in range(sequence_map.shape[0]):
        start_idx, end_idx = sequence_map[i]
        prompt_len = prompt_lens[i].item()
        response_start_idx = start_idx + prompt_len

        if response_start_idx >= end_idx:
            continue

        relevant_logits = logits[response_start_idx - 1:end_idx - 1]
        targets = tokens[response_start_idx:end_idx]

        if relevant_logits.shape[0] == 0:
            continue

        if temperature != 1.0:
            relevant_logits = relevant_logits / temperature

        log_probs = F.log_softmax(relevant_logits, dim=-1)
        sequence_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        logprobs_list.append(sequence_logprobs)

    if not logprobs_list:
        return torch.tensor([], device=logits.device, dtype=logits.dtype)

    return torch.cat(logprobs_list)
```

## Optimization Note

To reduce GPU data transfers, `pack_sequences` now packs sequences on CPU and moves the resulting packed tensors to GPU once, rather than moving individual tensors beforehand. This minimizes memory allocations and transfers, improving efficiency, especially with many sequences. The current implementation is functional, so this optimization is optional but recommended.

## Open Questions / TODOs

- **Multi-GPU/Multi-Node:** Ensure compatibility with distributed PyTorch utilities (e.g., buffer sampling, gradient reduction).
- **Performance Optimization:** Profile packing/unpacking, logprob calculation, and loss calculation; optimize if needed (e.g., vectorized gather/scatter for logprob extraction).
- **Testing:** Add unit and integration tests for packing, logprob calculation, loss, and the pipeline.
- **Error Handling:** Enhance error handling in worker `run` loops.
- **Replay Buffer:** Verify the replay buffer correctly handles storing/sampling Python dataclass objects and CPU/GPU transitions.