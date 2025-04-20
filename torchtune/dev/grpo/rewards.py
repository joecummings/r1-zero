# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict, List, Optional, Protocol, Tuple, Union
from xml.etree import ElementTree as ET

import torch
from dataclasses import dataclass, field

from torchtune.modules.transforms.tokenizers import ModelTokenizer

@dataclass
class RewardOutput:
    reward_base_name: str
    total_reward: torch.Tensor
    successes: torch.Tensor
    # optional
    rewards: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)

    def log(self, prefix: str = "") -> Dict[str, float]:
        log_dict = {}
        prefix = f"{prefix}/{self.reward_base_name}" if prefix else self.reward_base_name
        for reward_name, reward in self.rewards.items():
            log_dict[f"{prefix}/{reward_name}"] = reward.mean().item()
        log_dict[f"{prefix}"] = self.total_reward.mean().item()
        log_dict[f"{prefix}/successes"] = self.successes.mean().item()
        return log_dict


class RewardBase(Protocol):
    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
        device: torch.device,
    ) -> RewardOutput:
        pass


class FormattedMathCorrectnessReward(RewardBase):
    """
    This reward encourages the model to correctly answer a math problem, and requires
    the model to repond in an XML-style format to extract answers. 

    Args:
        answer_tag: the tag for the answer section. The answer will be extracted from <{answer_tag}>{answer}</{answer_tag}>
        positive_reward: the reward provided for correctly formatted completions
        negative_reward: the reward provided for incorrectly formatted completions
    """
    def __init__(self, answer_tag: str, positive_reward: float, negative_reward: float = 0.0):
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
        device: torch.device,
    ) -> RewardOutput:
        rewards = []
        answer_pattern = rf"<{self.answer_tag}>(.*?)</{self.answer_tag}>"
        for completion, answer in zip(completions, answers):
            match = re.search(answer_pattern, completion, re.S)
            if match:
                answer = match.group(1).strip()
                reward = self.positive_reward if answer == answer else self.negative_reward
                rewards.append(reward)
            else:
                rewards.append(self.negative_reward)
            
        rewards = torch.tensor(rewards)
        return RewardOutput(
            reward_base_name="math_correctness",
            total_reward=rewards,
            successes=(rewards == self.positive_reward).float()
        )

class ThinkingAnswerFormattingReward(RewardBase):
    """
    This reward encourages the model to respond in a reasoning-style format. It applies
    both a soft and strict formatting reward.

    The "soft" formatting reward rewards the model for using the tags, even if the tags do not
    have newlines.

    The "strict" formatting reward rewards the model for using the tags, and having newlines.

    Taken from https://github.com/huggingface/open-r1/blob/06bdd503341f5375bf93c3720df13f8588d47712/src/open_r1/rewards.py

    Args:
        think_tag: the tag for the think section. The tag will be XML-formatted as <{think_tag}>...</{think_tag}>
        answer_tag: the tag for the answer section. The tag will be XML-formatted as <{answer_tag}>...</{answer_tag}>
        positive_reward: the reward provided for correctly formatted completions
        negative_reward: the reward provided for incorrectly formatted completions
    """
    def __init__(self, think_tag: str, answer_tag: str, positive_reward: float, negative_reward: float = 0.0):
        self.think_tag = think_tag
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
        device: torch.device,
    ) -> RewardOutput:
        # soft format reward pattern
        think_pattern = rf"<{self.think_tag}>.*?</{self.think_tag}>"
        answer_pattern = rf"<{self.answer_tag}>.*?</{self.answer_tag}>"

        # strict format reward pattern
        strict_pattern = rf"^<{self.think_tag}>\n.*?\n</{self.think_tag}>\n<{self.answer_tag}>\n.*?\n</{self.answer_tag}>\n$"
        soft_format_rewards = []
        strict_format_rewards = []
        for completion in completions:
            strict_format_rewards.append(self.positive_reward if re.match(strict_pattern, completion, re.DOTALL | re.MULTILINE) else self.negative_reward)

            think_matches = re.findall(think_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(think_matches) == 1 and len(answer_matches) == 1:
                think_index = completion.find(think_matches[0])
                answer_index = completion.find(answer_matches[0])
                if think_index < answer_index:
                    soft_format_rewards.append(self.positive_reward)
                    continue
            soft_format_rewards.append(self.negative_reward)

        soft_format_rewards = torch.tensor(soft_format_rewards)
        strict_format_rewards = torch.tensor(strict_format_rewards)
        rewards = soft_format_rewards + strict_format_rewards
        successes = (rewards >= self.positive_reward).float()
        return RewardOutput(
            reward_base_name="formatting",
            total_reward=rewards,
            rewards={
                "soft_format_reward": soft_format_rewards,
                "strict_format_reward": strict_format_rewards,
            },
            successes=successes
        )

def extract_tags(text: str) -> Tuple[str, str]:
    """
    Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
    The values are lists of strings, with each string being the content of a tag.
    """
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    cot = think_match.group(1).strip() if think_match else ""
    potential_answer = answer_match.group(1).strip() if answer_match else ""
    return cot, potential_answer


def at_least_one_space_between_think_tags(
    cot: str, answer: str, potential_answer: str
) -> Tuple[int, int]:
    """Did the model at least try to think?"""
    if len(cot) > 0:
        return 1.0, 1.0  # (reward, success)
    else:
        return 0.0, 0.0


def math_response_correct(
    cot: str, answer: str, potential_answer: str
) -> Tuple[int, int]:
    """Did it get the right answer?"""
    if potential_answer is None:
        return 0.0, 0.0  # (reward, success)
    if answer == potential_answer:
        return 100.0, 1.0
    if answer in potential_answer:
        return 50.0, 0.0
    if len(potential_answer) > 0:
        return 1.0, 0.0
    return 0.0, 0.0


def batched_rewards(
    tokenizer: ModelTokenizer,
    completion_ids: torch.Tensor,
    completions: str,
    answers: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:

    reward_funcs = [
        at_least_one_space_between_think_tags,
        math_response_correct,
    ]
    num_reward_funcs = len(reward_funcs)
    batch_size, grpo_size, _ = completions.shape

    # TODO: should this be bfloat16?
    rewards_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )
    successes_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )
    metadata = {"func_names": [f.__name__ for f in reward_funcs]}
    for b in range(batch_size):
        for g in range(grpo_size):
            answer = answers[b][g]
            text_completion = tokenizer.decode(completions[b, g].tolist())
            cot, potential_answer = extract_tags(f"<think>{text_completion}")
            for rw_idx, reward_func in enumerate(reward_funcs):
                reward, success = reward_func(cot, answer, potential_answer)
                rewards_tensor[b, g, rw_idx] += reward
                successes_tensor[b, g, rw_idx] += success

    return rewards_tensor, successes_tensor, metadata


def shaped_correctness_reward(answer: str, completion: str) -> tuple[float, float]:
    """
    Reward function for verifiable rewards with some mild shaping.

    Args:
        answer (str): ground-truth answer to the current problem
        completion (str): model's completion, starting immediately after "Assistant: <think>"
    Returns:
        reward: (float) a shaped reward indicating the correct answer and the correct format
        success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
    """
    reward = 0.0
    success = 0.0

    try:
        tags = extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
    except ET.ParseError:
        tags = {"think": [], "answer": []}

    if len(tags["answer"]) == 1:
        reward += 5.0

    if len(tags["think"]) == 1:
        reward += 5.0

    if any(attempt == answer for attempt in tags["answer"]):
        # One of the answer tags has the right answer
        reward += 20.0

    if any((answer in attempt) for attempt in tags["answer"]):
        # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
        reward += 10.0

    if len(tags["answer"]) > 0 and tags["answer"][-1] == answer:
        reward = 100.0
        success = 1

    return reward, success


def batch_shaped_correctness_reward(
    tokenizer: ModelTokenizer, completions: torch.Tensor, answers: list[str]
) -> [torch.Tensor, torch.Tensor]:
    """Utility function to apply the shaped reward function to a GRPO-style batch of completions."""

    batch_size, grpo_size, *_ = completions.shape
    rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    # completions :: [B, G, L]
    for b in range(batch_size):
        for g in range(grpo_size):
            text_completion = tokenizer.decode(
                completions[b, g].tolist()
            )  # skips special tokens, stops at eos
            reward, success = shaped_correctness_reward(
                answer=answers[b], completion=text_completion
            )
            rewards[b, g] = reward
            successes[b, g] = success

    return rewards, successes


# class ShapedCorrectnessReward(Transform):
#     def __init__(self, tokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer

#     def _step(
#         self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
#     ) -> TensorDictBase:
#         # Get the completion
#         responses = next_tensordict["responses"]  # batch_size, grpo_size, L
#         answers = next_tensordict["answers"]  # batch_size, grpo_size
#         if responses.ndim  == 3:
#             batch_size, grpo_size, _ = responses.shape
#         # decode
#         text_completion = self.tokenizer.decode(
#             responses.flatten(0, 1).tolist()
#         )
#         # Decomposed reward
#         tds = [self.single_shaped_correctness_reward(answer, compl) for answer, compl in zip(answers.view(-1), text_completion)]
#         tds = torch.stack(tds)
#         if responses.ndim  == 3:
#             tds = tds.reshape(batch_size, grpo_size)
#         tds = tds.apply(lambda t: t.unsqueeze(-1))
#         return tds

#     def transform_reward_spec(self, reward_spec: Composite) -> Composite:
#         shape = reward_spec.shape + (1,)
#         reward_spec.update(Composite(
#             reward_answer=Unbounded(shape),
#             reward_think=Unbounded(shape),
#             reward_right=Unbounded(shape),
#             reward_contained=Unbounded(shape),
#             reward=Unbounded(shape),
#             success=Unbounded(shape, dtype=torch.bool),
#         ))
#         return reward_spec

#     @classmethod
#     def single_shaped_correctness_reward(cls, answer: str, completion: str) -> tuple[float, float]:
#         """
#         Reward function for verifiable rewards with some mild shaping.

#         Args:
#             answer (str): ground-truth answer to the current problem
#             completion (str): model's completion, starting immediately after "Assistant: <think>"
#         Returns:
#             reward: (float) a shaped reward indicating the correct answer and the correct format
#             success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
#         """

#         try:
#             tags = extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
#         except ET.ParseError:
#             tags = {"think": [], "answer": []}

#         reward_answer = 5.0 * (len(tags["answer"]) == 1)

#         reward_think = 5.0 * (len(tags["think"]) == 1)

#         # One of the answer tags has the right answer
#         reward_right = 20.0 * (any(attempt == answer for attempt in tags["answer"]))

#         # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
#         reward_contained = 10.0 * (any((answer in attempt) for attempt in tags["answer"]))

#         success = len(tags["answer"]) > 0 and tags["answer"][-1] == answer
#         # Compose the rewards
#         reward = 100.0 * float(success) + (reward_answer + reward_think + reward_contained + reward_right) * (1- float(success))

#         rewards = TensorDict(
#             reward_answer=reward_answer,
#             reward_think=reward_think,
#             reward_right=reward_right,
#             reward_contained=reward_contained,
#             reward=reward,
#             success=success,
#         )
#         return rewards
