# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
import importlib.util

import random
import string

import pytest

import torch

from tensordict import NonTensorData, NonTensorStack, set_capture_non_tensor_stack
from torchrl.envs import StepCounter
from torchtune.dev.grpo.envs import as_padded_tensor, DataLoadingPrimer, LLMEnv


_has_transformers = importlib.util.find_spec("transformers") is not None


class DummyStrDataLoader:
    def __init__(self, batch_size=0):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size

    def generate_random_string(self, length=10):
        """Generate a random string of a given length."""
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            return {"text": self.generate_random_string()}
        else:
            return {
                "text": [self.generate_random_string() for _ in range(self.batch_size)]
            }


class DummyTensorDataLoader:
    def __init__(self, batch_size=0, max_length=10, padding=False):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding

    def generate_random_tensor(self):
        """Generate a tensor of random int64 values."""
        length = random.randint(1, self.max_length)
        rt = torch.randint(1, 10000, (length,))
        return rt

    def pad_tensor(self, tensor):
        """Pad a tensor to the maximum length."""
        padding_length = self.max_length - len(tensor)
        return torch.cat((torch.zeros(padding_length, dtype=torch.int64), tensor))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            tensor = self.generate_random_tensor()
            tokens = self.pad_tensor(tensor) if self.padding else tensor
        else:
            tensors = [self.generate_random_tensor() for _ in range(self.batch_size)]
            if self.padding:
                tensors = [self.pad_tensor(tensor) for tensor in tensors]
                tokens = torch.stack(tensors)
            else:
                tokens = tensors
        return {"tokens": tokens, "attention_mask": tokens != 0}


class TestLLMEnv:
    @pytest.fixture(scope="class", autouse=True)
    def set_capture(self):
        with set_capture_non_tensor_stack(False):
            yield None
        return

    @pytest.mark.skipif(not _has_transformers, reason="test requires transformers")
    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            # TODO: a bit experimental, fails with check_env_specs
            # [False, "as_nested_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_llm_env(
        self, from_text, stack_method, device, dl_batch_size, env_batch_size
    ):
        if from_text:
            primer = DataLoadingPrimer(
                dataloader=DummyStrDataLoader(batch_size=dl_batch_size),
                batch_size=env_batch_size,
            )
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            primer = DataLoadingPrimer(
                dataloader=DummyTensorDataLoader(
                    batch_size=dl_batch_size, padding=True
                ),
                stack_method=stack_method,
                batch_size=env_batch_size,
            )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv(
                from_text=from_text,
                device=device,
                batch_size=primer.batch_size,
            )
        env = env.append_transform(primer)
        if env_batch_size is None:
            assert env.batch_size == torch.Size((dl_batch_size,))
        else:
            if not isinstance(env_batch_size, tuple):
                env_batch_size = (
                    torch.Size(())
                    if env_batch_size == 0
                    else torch.Size((env_batch_size,))
                )
            assert env.batch_size == env_batch_size

        env.check_env_specs(break_when_any_done="both")

    @pytest.mark.skipif(not _has_transformers, reason="test requires transformers")
    @pytest.mark.parametrize("tokenizer", [True, False])
    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    def test_llm_from_dataloader(
        self,
        from_text,
        stack_method,
        device,
        dl_batch_size,
        env_batch_size,
        tokenizer,
    ):
        from transformers import AutoTokenizer

        if tokenizer and from_text:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            tokenizer = None
        if from_text:
            kwargs = {
                "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
            }
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            kwargs = {
                "dataloader": DummyTensorDataLoader(
                    padding=True, batch_size=dl_batch_size
                ),
                "stack_method": stack_method,
            }
        kwargs.update(
            {
                "batch_size": env_batch_size,
                "from_text": from_text,
                "device": device,
                "has_attention": False,
                "tokenizer": tokenizer,
            }
        )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv.from_dataloader(**kwargs)
        if env_batch_size is None:
            assert env.batch_size == torch.Size((dl_batch_size,))
        else:
            if not isinstance(env_batch_size, tuple):
                env_batch_size = (
                    torch.Size(())
                    if env_batch_size == 0
                    else torch.Size((env_batch_size,))
                )
            assert env.batch_size == env_batch_size
        env.check_env_specs(break_when_any_done="both")

        def policy(td):
            if from_text and tokenizer is None:
                if not td.shape:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorData(
                        "<nothing>", device=device
                    )
                else:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorStack(
                        *[
                            NonTensorData("<nothing>", device=device)
                            for _ in range(td.shape[0])
                        ]
                    )
            else:
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (1,), dtype=torch.int64
                )
            return td

        r = env.rollout(10, policy)
        if env.batch_size == ():
            assert r.ndim == 1
            r = r.unsqueeze(0)
        else:
            assert r.ndim == 2
        if from_text and tokenizer is None:
            assert isinstance(r[0, 0][LLMEnv._DEFAULT_STR_KEY], str)
            assert isinstance(r[0, 1][LLMEnv._DEFAULT_STR_KEY], str)
            assert (
                r[0, 0][LLMEnv._DEFAULT_STR_KEY]
                == r[0, 1][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[0, 0][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            ), (
                r[0, 0][LLMEnv._DEFAULT_STR_KEY],
                r[0, 0][LLMEnv._DEFAULT_ACTION_STR_KEY],
                r[0, 0]["next", LLMEnv._DEFAULT_STR_KEY],
                r[0, 1][LLMEnv._DEFAULT_STR_KEY],
            )
            assert (
                r[0, 1][LLMEnv._DEFAULT_STR_KEY]
                == r[0, 2][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[0, 1][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
            assert (
                r[-1, 0][LLMEnv._DEFAULT_STR_KEY]
                == r[-1, 1][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[-1, 0][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
            assert (
                r[-1, 1][LLMEnv._DEFAULT_STR_KEY]
                == r[-1, 2][LLMEnv._DEFAULT_STR_KEY][
                    : -len(r[-1, 1][LLMEnv._DEFAULT_ACTION_STR_KEY])
                ]
            )
        elif tokenizer is None:
            assert (
                r[0, 0][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[0, 1][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[0, 1][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[0, 2][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[-1, 0][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[-1, 1][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()
            assert (
                r[-1, 1][LLMEnv._DEFAULT_TOKEN_KEY]
                == r[-1, 2][LLMEnv._DEFAULT_TOKEN_KEY][:-1]
            ).all()

    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
            # TODO: a bit experimental, fails with check_env_specs
            # [False, "as_nested_tensor"],
            [False, None],
        ],
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("repeats", [3])
    def test_llm_from_dataloader_repeats(
        self, from_text, stack_method, device, env_batch_size, dl_batch_size, repeats
    ):
        if from_text:
            kwargs = {
                "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
                "repeats": repeats,
            }
        else:
            if stack_method is None:
                stack_method = as_padded_tensor
            kwargs = {
                "dataloader": DummyTensorDataLoader(
                    padding=True, batch_size=dl_batch_size
                ),
                "stack_method": stack_method,
                "repeats": repeats,
            }
        kwargs.update(
            {
                "batch_size": env_batch_size,
                "from_text": from_text,
                "device": device,
                "has_attention": False,
            }
        )
        with pytest.warns(UserWarning, match="eos_token_id"):
            env = LLMEnv.from_dataloader(**kwargs)
        assert env.transform.repeats == repeats

        max_steps = 3
        env.append_transform(StepCounter(max_steps=max_steps))

        def policy(td):
            if from_text:
                if not td.shape:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = "<nothing>"
                else:
                    td[LLMEnv._DEFAULT_ACTION_STR_KEY] = NonTensorStack(
                        *["<nothing>" for _ in range(td.shape[0])]
                    )
            else:
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (1,), dtype=torch.int64
                )
            return td

        r = env.rollout(100, policy, break_when_any_done=False)
        # check that r at reset is always the same
        r_reset = r[..., ::max_steps]
        if from_text:
            all_strings = r_reset.view(-1)[LLMEnv._DEFAULT_STR_KEY]
            assert sum(s == all_strings[0] for s in all_strings) == repeats
            assert sum(s == all_strings[repeats] for s in all_strings) == repeats
            assert sum(s == all_strings[repeats * 2] for s in all_strings) == repeats
        else:
            all_tokens = r_reset.view(-1)[LLMEnv._DEFAULT_TOKEN_KEY]
            assert sum((s == all_tokens[0]).all() for s in all_tokens) == repeats
            assert sum((s == all_tokens[repeats]).all() for s in all_tokens) == repeats
            assert (
                sum((s == all_tokens[repeats * 2]).all() for s in all_tokens) == repeats
            )

    @pytest.mark.parametrize(
        "from_text,stack_method",
        [
            [True, None],
            [False, "as_padded_tensor"],
        ],
    )
    @pytest.mark.parametrize("device", [None])
    @pytest.mark.parametrize("dl_batch_size", [1, 4])
    @pytest.mark.parametrize("env_batch_size", [None, 0, (), 4])
    @pytest.mark.parametrize("repeats", [3])
    @pytest.mark.parametrize(
        "assign_reward,assign_done", [[True, False], [True, True], [False, True]]
    )
    def test_done_and_reward(
        self,
        from_text,
        stack_method,
        device,
        env_batch_size,
        dl_batch_size,
        repeats,
        assign_reward,
        assign_done,
    ):
        with pytest.raises(
            ValueError, match="from_text"
        ) if from_text else contextlib.nullcontext():
            if from_text:
                kwargs = {
                    "dataloader": DummyStrDataLoader(batch_size=dl_batch_size),
                    "repeats": repeats,
                    "assign_reward": assign_reward,
                    "assign_done": assign_done,
                }
            else:
                if stack_method is None:
                    stack_method = as_padded_tensor
                kwargs = {
                    "dataloader": DummyTensorDataLoader(
                        padding=True, batch_size=dl_batch_size
                    ),
                    "stack_method": stack_method,
                    "repeats": repeats,
                    "assign_reward": assign_reward,
                    "assign_done": assign_done,
                }
            kwargs.update(
                {
                    "batch_size": env_batch_size,
                    "from_text": from_text,
                    "device": device,
                    "has_attention": False,
                }
            )
            with pytest.warns(UserWarning, match="eos_token_id"):
                env = LLMEnv.from_dataloader(**kwargs)
            # We want to make sure that transforms that rely on the done state work appropriately
            env.append_transform(StepCounter(max_steps=10))

            def policy(td):
                td[LLMEnv._DEFAULT_ACTION_TOKENS_KEY] = torch.ones(
                    td.shape + (torch.randint(10, (1,)).item(),), dtype=torch.int64
                )
                return td

            r = env.rollout(100, policy, break_when_any_done=False)
            if assign_done:
                assert "terminated" in r
                assert "done" in r


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
