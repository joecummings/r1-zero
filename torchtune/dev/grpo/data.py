# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, TypedDict, Union

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.modules import packed_block_causal_mask
from torchtune.dev.rl.datatypes.trajectory import PackedTrajectory, UnscoredTrajectory, ScoredTrajectory
BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)


class ReasoningProblem(TypedDict):
    question: str
    cot: str
    answer: str


class RLDataset(Dataset):
    """
    Base class for datasets used in reinforcement learning,
    which provide a reference answer that can be verified to compute rewards.
    """

    def __init__(
        self,
        *,
        source: str,
        problem_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._problem_transform = problem_transform
        self._tokenizer = tokenizer

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._problem_transform(
            sample
        )  # keys "question" and "answer"

        question = BASE_PROMPT % transformed_sample["question"]

        q_tokens = self._tokenizer.encode(question, add_eos=False)
        mask = [1 for _ in q_tokens]
        answer = transformed_sample["answer"]

        return {
            "question": question,
            "tokens": q_tokens,
            "mask": mask,
            "answer": answer,
        }


def padded_collate_rl(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Answers are simply concatenated into a list.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing tokens.
        padding_idx (int): Padding index for input ids. Defaults to 0.

    Returns:
        Dict[str, Union[torch.Tensor, List[str]]]: Collated input tensors and string answers.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "answer": "15"},
        >>>    {"tokens": [7,], "answer": "bromance"},
        >>> ]
        >>> collated = padded_collate_rl(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["answers"]
        >>> ["15", "bromance"]
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    answers = [x["answer"] for x in batch]
    text = [x["question"] for x in batch]

    return {"tokens": input_ids.long(), "answers": answers, "text": text}

class TrajectorySource(Iterator):
    @abstractmethod
    def __next__(self) -> Union['GroupedUnscoredTrajectories', 'ScoredTrajectory']:
        pass

class QueueSource(TrajectorySource):
    def __init__(self, queue):
        self.queue = queue

    def __next__(self):
        while True:
            try:
                return self.queue.get(block=True, timeout=1.0)
            except Empty:
                time.sleep(1.0)

class ReplayBufferSource(TrajectorySource):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __next__(self):
        while len(self.replay_buffer) == 0:
            time.sleep(1.0)
        return self.replay_buffer.sample(1)[0]  # Sample with bsz=1

#TODO: review this
class RLPackedDataset(Iterator):
    def __init__(self, source: TrajectorySource, target_tokens_per_batch: int, pad_token_id: int, device: torch.device):
        self.source = source
        self.target_tokens_per_batch = target_tokens_per_batch
        self.pad_token_id = pad_token_id
        self.device = device
        self.accumulated_trajectories = []
        self.accumulated_tokens = 0

    def __next__(self):
        while True:
            item = next(self.source)
            if isinstance(item, GroupedUnscoredTrajectories):
                trajectories = item.trajectories
                tokens_in_group = item.total_tokens  # Use total_tokens for group
            else:  # ScoredTrajectory
                trajectories = [item]
                tokens_in_group = item.prompt_len + item.response_len

            # Check if adding this group exceeds target_tokens_per_batch
            if self.accumulated_tokens + tokens_in_group > self.target_tokens_per_batch and self.accumulated_trajectories:
                # Pack accumulated trajectories without the new item
                packed = pack_sequences(
                    sequences=self.accumulated_trajectories,
                    device=self.device,
                    pad_token_id=self.pad_token_id,
                    target_tokens_per_batch=self.target_tokens_per_batch
                )
                # Keep the new item for the next batch
                self.accumulated_trajectories = trajectories
                self.accumulated_tokens = tokens_in_group
                return packed
            else:
                # Accumulate trajectories
                self.accumulated_trajectories.extend(trajectories)
                self.accumulated_tokens += tokens_in_group

            # Pack entire groups together, yield when token threshold is met
            if self.accumulated_tokens >= self.target_tokens_per_batch or self.accumulated_trajectories:
                packed = pack_sequences(
                    sequences=self.accumulated_trajectories,
                    device=self.device,
                    pad_token_id=self.pad_token_id,
                    target_tokens_per_batch=self.target_tokens_per_batch
                )
                self.accumulated_trajectories = []
                self.accumulated_tokens = 0
                return packed
            
def pack_sequences(
    sequences: List[Union[UnscoredTrajectory, ScoredTrajectory]],
    device: torch.device,
    pad_token_id: int,
    target_tokens_per_batch: Optional[int] = None,
    ignore_index: int = -100
) -> PackedTrajectory:
    """Packs Trajectories into a single PackedTrajectory without padding in between sequences.

    For example, for 3 sequences with tokens (Prompt + Response)
    sequences = [
        [P1, P2, P3, P4, P5, R1, R2], 
        [P6, R3],  
        [P7, P8, R4, R5, R6,]
    ]

    Instead of padding to match the longest sequence, we concatenate all sequences
    and add padding tokens to reach target_tokens_per_batch:
    tokens = [P1, P2, P3, P4, P5, R1, R2, P6, R3, P7, P8, R4, R5, R6, PAD, PAD, PAD]

    Additional masks and tensors are also computed to facilitate efficient processing. For details,
    check out the docstring of `PackedTrajectory`.

    Args:
        sequences (List[Union[UnscoredTrajectory, ScoredTrajectory]]): Sequences to pack.
        device (torch.device): Device to place tensors on.
        pad_token_id (int): Token ID for padding.
        target_tokens_per_batch (Optional[int]): Target token count for padding; if None, no padding added.
        ignore_index (int): Used when generating targets

    Returns:
        PackedTrajectory: Packed batch of sequences.

    Example:
        packed = pack_sequences(sequences, torch.device('cuda'), 0, target_tokens_per_batch=4096)
    """
    if not sequences:
        raise ValueError("Cannot pack an empty list of sequences.")

    num_sequences = len(sequences)
    is_scored = isinstance(sequences[0], ScoredTrajectory)
    
    # Calculate lengths
    prompt_lens = torch.tensor([s.prompt_len for s in sequences], dtype=torch.long, device=device)
    response_lens = torch.tensor([s.response_len for s in sequences], dtype=torch.long, device=device)
    total_lens = prompt_lens + response_lens
    packed_seq_len_no_padding = total_lens.sum().item()
    
    # Aggregate all prompts and responses in order: P1, R1, P2, R2, ...
    tokens_list = []
    for s in sequences:
        tokens_list.append(s.prompt_tokens)
        tokens_list.append(s.response_tokens)

    # Pad the last response to target_tokens_per_batch
    packed_seq_len_padded = packed_seq_len_no_padding
    pad_count = 0
    total_lens_padded = total_lens.clone()
    if target_tokens_per_batch and packed_seq_len_no_padding < target_tokens_per_batch:
        pad_count = target_tokens_per_batch - packed_seq_len_no_padding
        padding = torch.full((pad_count,), pad_token_id, dtype=torch.long, device=device)
        tokens_list.append(padding)
        # Pad total length so generate masks with the correct shape
        total_lens_padded[-1] += pad_count
        packed_seq_len_padded = target_tokens_per_batch

    # Concatenate all pieces in a single call
    tokens = torch.cat(tokens_list, dim=0).to(dtype=torch.long)
    
    # Sequence map
    cumsum_lens = total_lens.cumsum(dim=0)
    sequence_map = torch.zeros((num_sequences, 2), dtype=torch.long, device=device)
    sequence_map[1:, 0] = cumsum_lens[:-1]
    sequence_map[:, 1] = cumsum_lens
    sequence_map = sequence_map
    
    # Attention Mask
    attention_mask = packed_block_causal_mask(total_lens_padded).to(device=device)

    # Sequence Mask
    sequence_mask = torch.repeat_interleave(torch.arange(num_sequences, device=device), total_lens_padded)

    # Response Mask
    # E.g., sequences [P1, R1, P2, P2, R2, P3, R3, PAD]
    # E.g., sequence_map[:, 0]=[0,2,5], prompt_lens=[1,2,1], response_start=[1,4,6]
    response_start = sequence_map[:, 0] + prompt_lens
    # E.g., sequence_mask=[0,0,1,1,1,2,2], response_start_per_token=[1,1,4,4,4,6,6]
    response_start_per_token = response_start[sequence_mask]
    # E.g., arange=[0,1,2,3,4,5,6,7], response_mask=[F,T,F,F,T,F,T,F]
    response_mask = torch.arange(packed_seq_len_padded, device=device) >= response_start_per_token
    # E.g., pad_count=1, response_mask unchanged=[F,T,F,F,T,F,T,F]
    response_mask[-pad_count:] = False

    # Group Mask
    group_mask = torch.zeros(len(sequences), device=device)
    prev_group_id = sequences[0].group_id
    int_group_id = 0
    for i, seq in enumerate(sequences):
        if seq.group_id != prev_group_id:
            int_group_id += 1
            group_mask[i] = int_group_id
            prev_group_id = seq.group_id
        else:
            group_mask[i] = int_group_id
   
    # create position ids
    start_per_token = sequence_map[sequence_mask, 0]
    position_ids = torch.arange(packed_seq_len_padded, device=device) - start_per_token

    # Pack additional fields for scored trajectories
    ref_logprobs = advantages = targets = None
    if is_scored:
        ref_logprobs = torch.cat([s.ref_logprobs for s in sequences])
        adv_values = torch.stack([s.advantages for s in sequences])
        advantages = torch.repeat_interleave(adv_values, response_lens)

        # -- Create targets --
        targets = tokens[response_mask]
        targets[:-1] = targets[1:]
        
        # Find the indices of the last response token in each sequence
        last_response_indices = response_lens.cumsum(dim=0) - 1 # shift by 1
        # Set the last token of each sequence to ignore_index
        targets[last_response_indices] = ignore_index

    # Assertions for shapes
    assert tokens.shape == (packed_seq_len_padded,), f"Tokens shape mismatch. Expected: {packed_seq_len_padded}, Actual: {tokens.shape}"
    assert attention_mask.shape == (packed_seq_len_padded, packed_seq_len_padded), f"Attention mask shape mismatch. Expected: {packed_seq_len_padded}, Actual: {attention_mask.shape}"
    assert position_ids.shape == (packed_seq_len_padded,), f"Position IDs shape mismatch. Expected: {packed_seq_len_padded}, Actual: {position_ids.shape}"
    assert response_mask.shape == (packed_seq_len_padded,), f"Response mask shape mismatch. Expected: {packed_seq_len_padded}, Actual: {response_mask.shape}"
    assert sequence_mask.shape == (packed_seq_len_padded,), f"Sequence mask shape mismatch. Expected: {packed_seq_len_padded}, Actual: {sequence_mask.shape}"
    assert group_mask.shape == (packed_seq_len_padded,), f"Group mask shape mismatch. Expected: {packed_seq_len_padded}, Actual: {group_mask.shape}"

    if is_scored:
        total_response_tokens = response_lens.sum().item()
        assert ref_logprobs.shape == (total_response_tokens,), f"Ref logprobs shape mismatch. Expected: {total_response_tokens}, Actual: {ref_logprobs.shape}"
        assert advantages.shape == (total_response_tokens,), f"Advantages shape mismatch. Expected: {total_response_tokens}, Actual: {advantages.shape}"
        assert targets.shape == (total_response_tokens,), f"Targets shape mismatch. Expected: {total_response_tokens}, Actual: {targets.shape}"

    return PackedTrajectory(
        tokens=tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        response_mask=response_mask,
        sequence_mask=sequence_mask,
        group_mask=group_mask,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        sequence_map=sequence_map,
        packed_seq_len=packed_seq_len_padded,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        targets=targets,
        sequence_ids=[s.sequence_id for s in sequences],
        group_ids=[s.group_id for s in sequences],
        pad_count=pad_count
    )