from typing import Dict, List

import torch
from tensordict import TensorClass
from torchtune.dev.grpo.rewards import RewardOutput

class Trajectory(TensorClass["nocast"]):
    query_responses: torch.Tensor
    responses: torch.Tensor 
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    query_response_padding_masks: torch.Tensor
    seq_lens: torch.Tensor
    answers: torch.Tensor
    policy_version: int
    advantages: torch.Tensor
    reward_outputs: List[RewardOutput]
    sequence_ids: List[str]
