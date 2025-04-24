# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations, annotations

import warnings
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Literal

import torch
from tensordict import (
    LazyStackedTensorDict, NestedKey, TensorDict, TensorDictBase, is_leaf_nontensor, lazy_stack, unravel_key,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict
from torch.utils.data import DataLoader
from torchrl._utils import _replace_last
from torchrl.data.tensor_specs import Bounded, Composite, NonTensor, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import TensorDictPrimer, Transform
from torchrl.envs.utils import make_composite_from_td
from torchrl.modules.utils.utils import _unpad_tensors


def as_nested_tensor(list_of_tensordicts: list[TensorDictBase]) -> TensorDictBase:
    """Stacks a list of tensordicts into a single tensordict with nested tensors.

    Args:
        list_of_tensordicts (list[TensorDictBase]): A list of tensordicts to stack.

    Returns:
        TensorDictBase: A tensordict with nested tensors.

    """

    def _as_nested_tensor(*list_of_tensors):
        return torch.nested.as_nested_tensor(list_of_tensors, layout=torch.jagged)

    batch_size = list(list_of_tensordicts[0].shape)
    batch_size.insert(0, len(list_of_tensordicts))
    return list_of_tensordicts[0].apply(
        _as_nested_tensor, *list_of_tensordicts[1:], batch_size=batch_size
    )


def as_padded_tensor(
        list_of_tensordicts: list[[TensorDictBase]], dim=0, stack_dim: int = 0
) -> TensorDictBase:
    """Stacks a list of tensordicts into a single tensordict with padded tensors.

    Args:
        list_of_tensordicts (list[[TensorDictBase]]): A list of tensordicts to stack.
        dim (int, optional): The dimension along which to pad. Defaults to 0.
        stack_dim (int, optional): The dimension along which to stack. Defaults to 0.

    Returns:
        TensorDictBase: A tensordict with padded tensors.
    """

    def _stack_tensors(*list_of_tensors):
        if dim < 0:
            raise ValueError("dim must be >= 0")
        max_length = max([t.size(dim) for t in list_of_tensors])

        def pad_tensor(tensor):
            padding_length = max_length - tensor.size(dim)
            shape = [s if i != dim else padding_length for i, s in enumerate(tensor.shape)]
            return torch.cat((tensor.new_zeros(shape), tensor), dim=dim)

        return torch.stack([pad_tensor(t) for t in list_of_tensors], dim=stack_dim)

    batch_size = list(list_of_tensordicts[0].shape)
    batch_size.insert(dim, len(list_of_tensordicts))
    result = list_of_tensordicts[0].apply(
        _stack_tensors, *list_of_tensordicts[1:], batch_size=batch_size
    )
    return result


class DataLoadingPrimer(TensorDictPrimer):
    """A primer that loads data from a dataloader and converts it into a tensordict using ``stack_method``.

    Args:
        dataloader (Iterable[Dict[str, Any]]): The dataloader to load data from.
            During collection, we will attempt to convert it into a tensordict using :func:`~tensordict.from_dict` or a
            similar function.
            It is assumed that the elements retrieved from the dataloader come in batches along the first dimension
            of every tensor, unless `dataloader.batch_size=0`.
            The dataloader must yield mappable data structures (e.g., dictionaries).

    Keyword Args:
        primers (Composite | None, optional): The primers to use for each key in the dataloader. Defaults to None.
        stack_method (Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"], optional): The method to
            use for stacking the data. Defaults to ``maybe_dense_stack``.
        repeats (int, optional): How many times the same sample needs to appear successively. This can be useful in
            situations like GRPO where a single prompt is used multiple times to estimate the advantage using Monte-Carlo
            samples (rather than an advantage module).
        batch_size (int, torch.Size or None): the batch-size of the data delivered by the transform.
            This is somewhat unrelated to the batch-size of the dataloader, in the sense that this number may or may
            not match the DL's batch size.
            If left empty, the batch-size is inferred from `dataloader.batch_size` if that attribute exists. If not,
            an empty batch-size will be used (`torch.Size([])`).

            .. note:: The batch-size of the Primer must match the batch-size of the parent environment (typically a
                wrapper around :class:`~torchrl.envs.LLMEnv`).

        group_repeats (bool, optional): if ``True``, the batch-size is multiplied by the number of repeats such that
            all repeats are grouped in a single batch collected from the buffer. Defaults to ``False``.

    Attributes:
        dataloader (Iterable[Any]): The dataloader to load data from.
        endless_dataloader (Iterable[Any]): An endless iterator over the dataloader.
        stack_method (Callable[[Any], Any]): The method to use for stacking the data.

    .. seealso:: :class:`~torchrl.envs.LLMEnv` and :class:`~torchrl.envs.LLMEnv.from_dataloader`.

    Example of a dataloader yielding strings:
        >>> import random
        >>> import string
        >>> import tensordict as td
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded
        >>> td.set_capture_non_tensor_stack(False).set()
        >>> class DummyDataLoader:
        ...     '''A dummy dataloader that generates random strings.'''
        ...     def __init__(self, batch_size: int = 0):
        ...         self.batch_size = batch_size
        ...     def generate_random_string(self, length: int = 10) -. str:
        ...         '''Generate a random string of a given length.'''
        ...         return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        ...     def __iter__(self):
        ...         return self
        ...     def __next__(self):
        ...         if self.batch_size == 0:
        ...             return self.generate_random_string()
        ...         else:
        ...             return [self.generate_random_string() for _ in range(self.batch_size)]
        >>> # Create an LLM environment with string-to-string input/output.
        >>> env = LLMEnv(from_text=True)
        >>> # Append a DataLoadingPrimer to the environment.
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyDataLoader(),
        >>>         example_data="a string!",
        >>>     )
        >>> )
        >>> # Test the environment.
        >>> print(env.rand_action(TensorDict()))
        TensorDict(
            fields={
                action: NonTensorData(data=a string, batch_size=torch.Size([]), device=None)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.rollout(3))
        TensorDict(
            fields={
                action: NonTensorStack(
                    ['a string', 'a string', 'a string'],
                    batch_size=torch.Size([3]),
                    device=None),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: NonTensorStack(
                            ['zxwvupirska string', 'zxwvupirska stringa string...,
                            batch_size=torch.Size([3]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: NonTensorStack(
                    ['zxwvupirsk', 'zxwvupirska string', 'zxwvupirska ...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # Roll out the environment with a specific initial state.
        >>> init_state = env.reset(TensorDict(batch_size=[3]))
        >>> print(env.rollout(3, auto_reset=False, tensordict=init_state))
        TensorDict(
            fields={
                action: NonTensorStack(
                    [['a string', 'a string', 'a string'], ['a string'...,
                    batch_size=torch.Size([3, 3]),
                    device=None),
                done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: NonTensorStack(
                            [[array(['nngcmflsana string', 'vrrbnhzpmga string...,
                            batch_size=torch.Size([3, 3]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3, 3]),
                    device=None,
                    is_shared=False),
                observation: NonTensorStack(
                    [['nngcmflsan', array(['nngcmflsana string', 'vrrb...,
                    batch_size=torch.Size([3, 3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3, 3]),
            device=None,
            is_shared=False)

    Example of dataloader yielding tensors:
        >>> import random
        >>> import string
        >>>
        >>> import tensordict as td
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded
        >>>
        >>> td.set_capture_non_tensor_stack(False).set()
        >>>
        >>>
        >>> class DummyTensorDataLoader:
        ...     '''A dummy dataloader that generates tensors of random int64 values.'''
        ...
        ...     def __init__(self, batch_size: int = 0, max_length: int = 10, padding: bool = False):
        ...         '''
        ...         Args:
        ...             batch_size (int, optional): The batch size of the generated tensors. Defaults to 0.
        ...             max_length (int, optional): The maximum length of the generated tensors. Defaults to 10.
        ...             padding (bool, optional): Whether to pad the tensors to the maximum length. Defaults to False.
        ...         '''
        ...         self.batch_size = batch_size
        ...         self.max_length = max_length
        ...         self.padding = padding
        ...
        ...     def generate_random_tensor(self) -. torch.Tensor:
        ...         '''Generate a tensor of random int64 values.'''
        ...         length = random.randint(1, self.max_length)
        ...         return torch.tensor([random.randint(0, 100) for _ in range(length)], dtype=torch.int64)
        ...
        ...     def pad_tensor(self, tensor: torch.Tensor) -. torch.Tensor:
        ...         '''Pad a tensor to the maximum length.'''
        ...         padding_length = self.max_length - len(tensor)
        ...         return torch.cat((torch.zeros(padding_length, dtype=torch.int64), tensor))
        ...
        ...     def __iter__(self):
        ...         return self
        ...
        ...     def __next__(self):
        ...         if self.batch_size == 0:
        ...             tensor = self.generate_random_tensor()
        ...             return self.pad_tensor(tensor) if self.padding else tensor
        ...         else:
        ...             tensors = [self.generate_random_tensor() for _ in range(self.batch_size)]
        ...             if self.padding:
        ...                 tensors = [self.pad_tensor(tensor) for tensor in tensors]
        ...                 return torch.stack(tensors)
        ...             else:
        ...                 return tensors
        >>>
        >>> # Create an LLM environment with non-string input/output and append a DataLoadingPrimer.
        >>> env = LLMEnv(from_text=False)
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyTensorDataLoader(),
        >>>         data_specs=[Unbounded(shape=(-1,), dtype=torch.int64)],
        >>>     )
        >>> )
        >>> print(env.rand_action(TensorDict()))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.rollout(3))
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                observation: Tensor(shape=torch.Size([3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> # Create an LLM environment with padded tensor input/output and append a DataLoadingPrimer.
        >>> env = LLMEnv(from_text=False)
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyTensorDataLoader(padding=True),
        >>>         data_specs=[Unbounded(shape=(-1,), dtype=torch.int64)],
        >>>         stack_method="as_padded_tensor",
        >>>     )
        >>> )
        >>> print(env.rollout(3, auto_reset=False, tensordict=env.reset(TensorDict(batch_size=[3]))))
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([3, 3]),
                    device=None,
                    is_shared=False,
                    stack_dim=1),
                observation: Tensor(shape=torch.Size([3, 3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([3, 3]),
            device=None,
            is_shared=False,
            stack_dim=1)

    """

    def __init__(
            self,
            dataloader: Iterable[dict[str, Any]],
            *,
            primers: Composite | None = None,
            stack_method: Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"] | None = None,
            batch_size: int | torch.Size | None = None,
            repeats: int | None = None,
            device: torch.device | None = None,
            group_repeats: bool = False, ):
        self.dataloader = dataloader
        if repeats is None:
            repeats = 0
        self.repeats = repeats

        # Determine batch-size
        #  We must distinguish the batch-size of the DL and the batch size of the transform.
        #  We may want more or less elements than the DL and the logic is slightly different so we
        #  allow to recompose batches on the fly. If the DL has a batch-size, every element will be
        #  unbound and stored in a queue. Otherwise, we get as many elements from the DL to fulfill
        #  the required batch-size.
        #
        #  If the batch-size is passed, we will stack as many elements as necessary to fulfill this.
        #  If not, we try to get it from the dataloader. Contrary to the dataloader, we will always
        #  deliver the same batch-size (we create an infinite dataloader and reset when it's done),
        #  whereas DLs with drop_last=False may return batches of different sizes.
        #
        # If the batch size passed to the transform is empty (torch.Size(())) or 0, we will consider that
        #  the batch-size is determined on-the-fly.
        #
        # A batch-size of 0 in the dataloader means no batch-size.
        #
        # If needed, the various repeats can be grouped in a single batch through group_repeats.
        #
        # If auto_batch_size is on, we call auto_batch_size=True when doing TensorDict.from_dict:
        #  That way we get a tensordict of the right batch-size.
        # If the dataloader has no batch-size, we're not sure that we can determine the batch-size
        #  automatically so we will consider that each element in the DL has a batch-size of 0 (ie,
        #  a single non-batched element is returned at a time).

        if batch_size is None:
            batch_size = getattr(dataloader, "batch_size", torch.Size([]))
        if batch_size == 0:
            batch_size = torch.Size(())
        if not isinstance(batch_size, (list, tuple)):
            batch_size = (batch_size,)
        batch_size = torch.Size(batch_size)
        auto_batch_size = getattr(dataloader, "batch_size", 1) != 0

        if len(batch_size) > 1:
            raise ValueError(
                f"batch_size can only be 0 or 1D, got batch_size={batch_size}."
            )

        # We deliver all the repeats in the same batch
        if repeats and group_repeats:
            if batch_size == torch.Size([]):
                batch_size = torch.Size((repeats,))
            else:
                batch_size = torch.Size([batch_size[0] * repeats])

        self._queue = deque()
        self.auto_batch_size = auto_batch_size
        self.batch_size = batch_size
        self.endless_dataloader = self._endless_iter(self.dataloader)

        if stack_method is None:
            stack_method = lazy_stack
        elif stack_method == "as_nested_tensor":
            stack_method = as_nested_tensor
        elif stack_method == "as_padded_tensor":
            stack_method = as_padded_tensor
        elif not callable(stack_method):
            raise ValueError(f"Unknown stack_method={stack_method}")
        self.stack_method = stack_method

        if primers is None:
            # We can get the primer from the dataloader itself
            data = self._load_from_dataloader()
            primers = make_composite_from_td(data, dynamic_shape=True)
            if batch_size:
                primers = primers.expand(batch_size)
            self._queue.insert(0, data)
            self.data_keys = list(primers.keys(True, True))
        else:
            self.data_keys = list(primers.keys(True, True))

        super().__init__(
            primers=primers,
            default_value=self._load_from_dataloader,
            reset_key=None,
            expand_specs=None,
            single_default_value=True,
            call_before_env_reset=True,
            device=device, )
        self._reset_key = "_reset"

    @classmethod
    def _endless_iter(self, obj):
        while True:
            yield from obj

    def _load_from_dataloader(self, reset: torch.Tensor | None = None):
        """Loads a single element from the dataloader, or alternatively from the buffer.

        If `reset` is passed, then one element per reset will be loaded.
        """
        if reset is not None:
            if not reset.any():
                raise RuntimeError("reset must have at least one True value.")
            if reset.ndim > 0:
                loaded = [self._load_from_dataloader() for _ in range(reset.sum())]
                return self.stack_method(loaded)

        primers = getattr(self, "primers", None)
        if primers is not None:
            device = self.primers.device
        else:
            device = None

        if len(self._queue) > 0:
            result = self._queue.popleft()
            if result.device != device:
                result = result.to(device)
            return result

        data = next(self.endless_dataloader)
        # Some heuristic here:
        # if data is a map, assume its keys match the keys in spec
        # TODO: one could rename the keys too
        if isinstance(data, Mapping):
            out = TensorDict.from_dict(
                data,
                auto_batch_size=self.auto_batch_size,
                batch_dims=int(bool(self.auto_batch_size or self.batch_size)),
                device=device, )
        else:
            raise TypeError(
                "Data loader must return a mapping that can be automatically cast to a tensordict. Check that you have "
                "the appropriate collate_fn in your dataloader to do so."
            )
        if not out.ndim:
            out = out.unsqueeze(0)
        self._queue.extend(
            [d for d in out.unbind(0) for _ in range(max(1, self.repeats))]
        )
        out = self._queue.popleft()
        return out

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        # Check batch size
        parent = getattr(self, "parent", None)
        if (self.batch_size is not None and parent is not None and parent.batch_size != self.batch_size):
            warnings.warn(
                f"The parent env has a different batch size than the {type(self).__name__} transform."
            )
        return result

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(primers={self.primers}, dataloader={self.dataloader})"


class LLMEnv(EnvBase):
    """A text generation environment for language models.

    This environment is designed to work with language models, where the observation is a string or a tensor of
    integers representing a sequence of tokens. The action is also a string or a tensor of integers, which is
    concatenated to the previous observation to form the new observation.

    By default, this environment is meant to track history for a prompt. Users can append transforms to tailor
    this to their use case, such as Chain of Thought (CoT) reasoning or other custom processing.

    Users must append a transform to set the "done" condition, which would trigger the loading of the next prompt.
    Prompts to the language model can be loaded when the environment is ``reset`` if the environment is created via
    :meth:`~from_dataloader`.

    .. note:: The default arguments of the `LLMEnv` class are set to make it easy to run this environment with
        the vllm backend (:class:`~torchrl.modules.vLLMWrapper`).

    Keyword Args:
        token_key (NestedKey, optional): The key in the tensordict where the tokens are stored (when `from_text=False`).
            Defaults to ``"tokens"``.
        str_key (NestedKey, optional): The key in the tensordict where the string input is stored (when `from_text=True`).
            Defaults to ``"text"``.
        attention_key (NestedKey, optional): The key in the tensordict where the attention mask is stored.
            Defaults to ``"attention_mask"``.
        action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to
            ``"tokens_response"`` or ``"text_response"``.
        reward_key (NestedKey, optional): The key in the tensordict where the reward is stored if `assign_reward=True`.
            Defaults to  ``"reward"``.
        from_text (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``True``.
        device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
        vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
            unbounded vocabulary. Defaults to ``None``.
        has_attention (bool, optional): If ``True``, an attention mask is to be used under the key indicated by
            :attr:`attention_key`. Defaults to ``True``.
        assign_reward (bool, optional): If ``True``, a zero-valued reward of shape equal to the action shape
            is written during calls to `step()`. Defaults to ``False``.
        assign_done (bool, optional): If ``True``, a zero-valued done and terminated state of shape equal to the
            action shape is written during calls to `step()`. Defaults to ``False``.
            .. note:: Regardless of the value assigned to `assign_done`, a done state will be written at the root
                as it is a requirement for all TorchRL environments.
        batch_size (int or torch.Size, optional): Batch size of the environment.
            If left empty, an empty batch-size is assumed.
            The batch size can be null (`torch.Size([])`) or one-dimensional. Batchless environments are not supported.

            .. note:: When using a :class:`~torchrl.envs.DataLoadingPrimer` transform, the batch-size of the env
                and the transform should match.

        eos_token_id (int, optional): The token id of the end of the sequence. If passed, the `done` state
            is set to `True` when detected. Defaults to `None`.

    .. seealso:: :class:`~torchrl.envs.DataLoadingPrimer` for examples.

    Methods:
        from_dataloader: Creates an LLMEnv instance from a dataloader.

    """

    _DEFAULT_TOKEN_KEY = "tokens"
    _DEFAULT_STR_KEY = "text"
    _DEFAULT_ATTENTION_KEY = "attention_mask"
    _DEFAULT_ACTION_TOKENS_KEY = "tokens_response"
    _DEFAULT_ACTION_STR_KEY = "text_response"

    def __init__(
            self,
            *,
            token_key: NestedKey | None = None,
            str_key: NestedKey | None = None,
            attention_key: NestedKey | None = None,
            action_key: NestedKey | None = None,
            reward_key: NestedKey = "reward",
            from_text: bool = True,
            device: torch.device | None = None,
            vocab_size: int | None = None,
            assign_reward: bool = False,
            assign_done: bool = False,
            batch_size: int | torch.Size | None = None,
            has_attention: bool = True,
            # Experimental
            as_llm_data: bool = False,
            eos_token_id: int | None = None, ) -> None:
        self.as_llm_data = as_llm_data
        if token_key is None:
            token_key = self._DEFAULT_TOKEN_KEY
        if str_key is None:
            str_key = self._DEFAULT_STR_KEY
        if attention_key is None:
            attention_key = self._DEFAULT_ATTENTION_KEY
        if action_key is None:
            if from_text:
                action_key = self._DEFAULT_ACTION_STR_KEY
            else:
                action_key = self._DEFAULT_ACTION_TOKENS_KEY
        self._batch_locked = True
        if batch_size is None:
            batch_size = ()
        else:
            if not isinstance(batch_size, (tuple, list)):
                batch_size = (batch_size,)
            elif len(batch_size) > 1:
                raise TypeError(
                    f"batch-size of LLMEnv must be 0 or 1d. Got batch_size={batch_size}."
                )
        super().__init__(
            device=device, batch_size=batch_size, )
        self.has_attention = has_attention
        self.from_text = from_text
        self.vocab_size = vocab_size
        self.token_key = unravel_key(token_key)
        self.str_key = unravel_key(str_key)
        if attention_key is not None:
            attention_key = unravel_key(attention_key)
        self.attention_key = attention_key
        self.assign_reward = assign_reward
        self.assign_done = assign_done
        self.eos_token_id = eos_token_id
        if eos_token_id is None:
            warnings.warn(
                "eos_token_id is missing. This means that the environment will not be able to capture its "
                "done state automatically. This may lead to undefined behaviors when the generated text reaches "
                "an eos_token.", category=UserWarning, )

        # self.action_key = unravel_key(action_key)
        if from_text:
            self.full_observation_spec_unbatched = Composite(
                {
                    self.str_key: NonTensor(
                        example_data="a string", batched=True, shape=(), device=device, )
                }
            )
            self.full_action_spec_unbatched = Composite(
                {
                    action_key: NonTensor(
                        example_data="a string", batched=True, shape=(), device=device
                    )
                }
            )
        else:
            if vocab_size is None:
                observation_spec = {
                    token_key: Unbounded(shape=(-1,), dtype=torch.int64, device=device)
                }
                if self.has_attention:
                    observation_spec[attention_key] = Unbounded(
                        shape=(-1,), dtype=torch.int64, device=device
                    )
                self.full_observation_spec_unbatched = Composite(observation_spec)
                self.full_action_spec_unbatched = Composite(
                    {
                        action_key: Unbounded(
                            shape=(-1,), dtype=torch.int64, device=device
                        )
                    }
                )
            else:
                self.full_observation_spec_unbatched = Composite(
                    {
                        token_key: Bounded(
                            shape=(-1,), dtype=torch.int64, low=0, high=vocab_size, device=device, )
                    }
                )
                self.full_action_spec_unbatched = Composite(
                    {
                        action_key: Bounded(
                            shape=(-1,), dtype=torch.int64, low=0, high=vocab_size, device=device, )
                    }
                )
        STR2STR_ERR = ValueError(
            "from_text cannot be True when either of assign_reward / assign_done are True. "
            "Tokens are required to compute the reward shape."
        )
        if self.assign_reward:
            if self.from_text:
                raise STR2STR_ERR
            self.full_reward_spec_unbatched = Composite(
                {reward_key: Unbounded(shape=(-1,), device=device)}
            )
        else:
            self.full_reward_spec_unbatched = Composite(device=device)

        if not self.assign_done:
            # Use single done
            self.full_done_spec_unbatched = Composite(
                done=Unbounded(shape=(1,), dtype=torch.bool, device=device),
                terminated=Unbounded(shape=(1,), dtype=torch.bool, device=device), )
        elif self.from_text:
            raise STR2STR_ERR
        else:
            # Use single done
            self.full_done_spec_unbatched = Composite(
                tokens_data=Composite(
                    done=Unbounded(shape=(-1,), dtype=torch.bool, device=device),
                    terminated=Unbounded(shape=(-1,), dtype=torch.bool, device=device), ),
                done=Unbounded(shape=(1,), dtype=torch.bool, device=device),
                terminated=Unbounded(shape=(1,), dtype=torch.bool, device=device), )

    @classmethod
    def from_dataloader(
            cls,
            dataloader: DataLoader,
            *,
            tokenizer: transformers.PretrainedTokenizerBase | None = None,
            # noqa
            token_key: NestedKey | None = None,
            str_key: NestedKey | None = None,
            attention_key: NestedKey | None = None,
            action_key: NestedKey | None = None,
            reward_key: NestedKey = "reward",
            from_text: bool = True,
            device: torch.device | None = None,
            vocab_size: int | None = None,
            batch_size: int | torch.Size | None = None,
            has_attention: bool = True,
            assign_reward: bool = False,
            assign_done: bool = False,
            primers: Composite | None = None,
            example_data: Any = None,
            stack_method: Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"] = None,
            repeats: int | None = None,
            group_repeats: bool = True,
            eos_token_id: int | None = None, ) -> LLMEnv:
        """Creates an LLMEnv instance from a dataloader.

        This method creates an LLMEnv instance and appends a DataLoadingPrimer to it, which populates ``data_keys`` (by default ``observation_key``) with data from the provided dataloader when the environment is reset.

        Args:
            dataloader (DataLoader): The dataloader to load data from.

        Keyword Args:
            tokenizer (transformers.PretrainedTokenizerBase or str, optional): the tokenizer to use. If ``None``,
                "bert-base-uncased" will be used by default. If a string is provided, it should be the name of a
                pre-trained tokenizer.

                .. note:: Using the `tokenizer` will append a :class:`~torchrl.envs.Tokenizer` transform to the environment.
                    If `from_text` is set to `True`, the tokenizer will be called during every iteration and the rollout
                    will contain both tokens and text data.
                    If `from_text` is set to `False`, the tokenizer will be called during reset only, and the only
                    text data in the rollout will be the text sampled from the dataset.

            token_key (NestedKey, optional): The key in the tensordict where the tokens are stored (when `from_text=False`).
                Defaults to ``("tokens_in", "input_ids")``.
            str_key (NestedKey, optional): The key in the tensordict where the string input is stored (when `from_text=True`).
                Defaults to ``"test"``.
            attention_key (NestedKey, optional): The key in the tensordict where the attention mask is stored.
                Defaults to ``("tokens_in", "input_ids")``
            action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to
                ``("tokens_out", "sequences")``.
            reward_key (NestedKey, optional): The key in the tensordict where the reward is stored if `assign_reward=True`.
                Defaults to  ``"reward"``.
            from_text (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``True``.
            device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
            vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
                unbounded vocabulary. Defaults to ``None``.
            has_attention (bool, optional): if ``True``, an attention mask is to be used under the key indicated by
                :attr:`attention_key`. Defaults to ``True``.
            assign_reward (bool, optional): if ``True``, a zero-valued reward of shape equal to to the action shape
                is written during calls to `step()`. Defaults to ``False``.
            assign_done (bool, optional): if ``True``, a zero-valued done and terminated state of shape equal to to the
                action shape is written during calls to `step()`. Defaults to ``False``.

                .. note:: regardless of the value assigned to `assign_done`, a done state will be written at the root
                    as it is a requirement for all TorchRL environments.

            batch_size (int or torch.Size, optional): Batch size of the environment.
                If left empty, the batch size is inferred from `dataloader.batch_size` if that attribute exists, otherwise
                it is set to `()`.
                The batch size can be null (`torch.Size([])`) or one-dimensional. Batchless environments are not supported.

                .. note:: When using a :class:`~torchrl.envs.DataLoadingPrimer` transform, the batch-size of the env
                    and the transform should match.

            primers (Composite | None, optional): The primers to use for each key in the dataloader.
                Defaults to ``None`` (inferred automatically from the first batch of data).
            stack_method (Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"], optional): The
                method to use for stacking the data. Defaults to ``None``.
            repeats (int, optional): How many times the same sample needs to appear successively. This can be useful in
                situations like GRPO where a single prompt is used multiple times to estimate the advantage using Monte-Carlo
                samples (rather than an advantage module).
            group_repeats (bool, optional): if ``True``, the batch-size is multiplied by the number of repeats such that
                all repeats are grouped in a single batch collected from the buffer. Defaults to ``True``.
            eos_token_id (int, optional): The token id of the end of the sequence. If passed, the `done` state
                is set to `True` when detected. Defaults to `None`.

        Returns:
            LLMEnv: The created LLMEnv instance.
        """
        from smith.envs import DataLoadingPrimer, Tokenizer

        if str_key is None:
            str_key = LLMEnv._DEFAULT_STR_KEY
        if token_key is None:
            token_key = LLMEnv._DEFAULT_TOKEN_KEY
        if attention_key is None:
            attention_key = LLMEnv._DEFAULT_ATTENTION_KEY
        elif tokenizer is not None and attention_key != _replace_last(
                token_key, "attention_mask"
        ):
            raise ValueError(
                "When using the Tokenizer, attention key must match `(*token_key[:-1], 'attention_mask')` where "
                f"`token_key` is a tuple-typed nested key. Got attention_key={attention_key} while expecting "
                f"{_replace_last(token_key, 'attention_mask')}."
            )

        if tokenizer is not None:
            if from_text:
                # In this case, the tokenizer is appended to the env after each step
                if action_key is None:
                    action_key = cls._DEFAULT_ACTION_STR_KEY
                tokenizer_transform = Tokenizer(
                    tokenizer=tokenizer,
                    in_keys=[str_key],
                    out_keys=[token_key],
                    # Assume that the tokens are named according to _DEFAULT_ACTION_TOKENS_KEY
                    in_keys_inv=[action_key],
                    out_keys_inv=[cls._DEFAULT_ACTION_TOKENS_KEY],
                    call_before_reset=False,
                    # We should always see the required entries
                    missing_tolerance=False, )
            else:
                # FIXME: This is broken - do we need it anyway?
                raise RuntimeError(
                    "tokenizers can only be used whenever from_text is set to `True`."
                )

        primer = DataLoadingPrimer(
            dataloader=dataloader,
            primers=primers,
            stack_method=stack_method,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size, )
        env = LLMEnv(
            from_text=from_text,
            device=device,
            token_key=token_key,
            str_key=str_key,
            attention_key=attention_key,
            action_key=action_key,
            reward_key=reward_key,
            vocab_size=vocab_size,
            assign_reward=assign_reward,
            assign_done=assign_done,
            batch_size=primer.batch_size,
            has_attention=has_attention,
            eos_token_id=eos_token_id, )
        if tokenizer is not None:
            env = env.append_transform(tokenizer_transform)
        return env.append_transform(primer)

    @staticmethod
    def _check_obs_act_and_cat(obs, action, *, device):
        if not isinstance(obs, str):
            raise TypeError(f"Observation must be a string, got {type(obs)}.")
        if not isinstance(action, str):
            raise TypeError(f"Action must be a string, got {type(action)}.")
        return NonTensorData(obs + action, device=device)

    def _step(
            self, tensordict: TensorDictBase, ) -> TensorDictBase:
        next_td = tensordict.empty()
        self._make_next_obs(tensordict, next_td)
        self._maybe_make_reward(tensordict, next_td)
        self._maybe_make_done(tensordict, next_td)
        if self.as_llm_data:
            raise NotImplementedError()
        return next_td

    def _maybe_make_reward(
            self, tensordict: TensorDictBase, next_td: TensorDictBase
    ) -> TensorDictBase:
        if self.assign_reward:
            next_td.set(
                self.reward_key, torch.zeros_like(
                    tensordict.get(self.action_key), dtype=self.reward_spec.dtype
                ), )
        return next_td

    def _maybe_make_done(
            self, tensordict: TensorDictBase, next_td: TensorDictBase, resetting: bool = False, ) -> TensorDictBase:
        if self.assign_done:
            action = tensordict.get(self.action_key)
            if action is None:
                done = torch.zeros(
                    tensordict.shape + (1,), dtype=torch.bool, device=self.device
                )
            else:
                done = torch.zeros_like(action, dtype=torch.bool)
            next_td.set(("tokens_data", "terminated"), done)
            next_td.set(("tokens_data", "done"), done.clone())
            next_td.set(
                "done", next_td.get(("tokens_data", "done")).any(-1, keepdim=True)
            )
            next_td.set(
                "terminated", next_td.get(("tokens_data", "terminated")).any(-1, keepdim=True), )
        if not resetting and self.eos_token_id is not None:
            if self.from_text:
                token_action_key = self._DEFAULT_ACTION_TOKENS_KEY
            else:
                token_action_key = self.action_key
            action = tensordict.get(
                token_action_key, as_padded_tensor=True, padding_value=-1
            )
            mask = action == -1

            if action is None:
                raise RuntimeError(
                    f"Couldn't find the tokenized action with key {token_action_key} to set the done state in tensordict "
                    f"with keys {list(tensordict.keys(True))}."
                )
            full_done = action == self.eos_token_id
            done = full_done.any(-1, keepdim=True)
            next_td.set("done", done)
            next_td.set("terminated", done)
            if self.assign_done:
                full_done = _unpad_tensors(full_done, mask)
                next_td.set(("tokens_data", "terminated"), full_done)
                next_td.set(("tokens_data", "done"), full_done)
        return next_td

    def _make_next_obs(
            self, tensordict: TensorDictBase, nex_td: TensorDictBase
    ) -> TensorDictBase:
        # Cat action entry with prev obs
        if self.from_text:
            obs = tensordict[self.str_key]
            action = tensordict[self.action_key]
            if not tensordict.batch_size:
                if not isinstance(obs, str) or not isinstance(action, str):
                    raise TypeError(
                        "The tensordict is batchless, yet the action and/or observations are not "
                        f"strings but {type(action)} and {type(obs)}, respectivly."
                    )
                observation = self._check_obs_act_and_cat(
                    obs, action, device=self.device
                )
            else:
                observation = NonTensorStack(
                    *[self._check_obs_act_and_cat(_obs, _action, device=self.device) for (_obs, _action) in
                        _zip_strict(obs, action)]
                )
            return nex_td.set(self.str_key, observation)
        else:
            try:
                obs: torch.Tensor = tensordict.get(self.token_key)
                action = tensordict.get(self.action_key)
                if getattr(obs, "is_nested", False):
                    observation = torch.nested.as_nested_tensor(
                        [torch.cat([_obs, _action], -1) for _obs, _action in _zip_strict(
                            obs.unbind(0), action.unbind(0)
                        )], layout=obs.layout, )
                else:
                    observation = torch.cat([obs, action], -1)
                    if self.has_attention:
                        attention_mask = tensordict.get(self.attention_key)
                        attention_mask = torch.cat(
                            [attention_mask, attention_mask.new_ones(action.shape)], -1
                        )
                        nex_td.set(self.attention_key, attention_mask)
            except TypeError:
                raise TypeError(
                    "Failed to cat action and observation tensors. Check that from_text argument is correctly "
                    f"set in {type(self).__name__}."
                )
            return nex_td.set(self.token_key, observation)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # We should have an observation by this time, if not raise an exception
        def check_token():
            return not self.from_text and (self.token_key not in tensordict.keys(isinstance(self.token_key, tuple)))

        def check_str():
            return self.from_text and (self.str_key not in tensordict.keys(isinstance(self.str_key, tuple)))

        if tensordict is None or check_token() or check_str():
            raise KeyError(
                f"Observation key {self.token_key}/{self.str_key} is not defined in tensordict with keys "
                f"{list(tensordict.keys(True, True, is_leaf=is_leaf_nontensor))}. Make sure a TensorDictPrimer (eg, "
                f"torchrl.envs.DataLoadingPrimer) is appended to the env transforms."
            )
        if not isinstance(tensordict, LazyStackedTensorDict) and tensordict.ndim:
            tensordict = LazyStackedTensorDict(*tensordict.unbind(0))
        td_reset = tensordict.copy()
        if td_reset.device != self.device:
            if self.device is None:
                td_reset.clear_device_()
            else:
                td_reset = td_reset.to(self.device)
        tensordict = self._maybe_make_done(tensordict, td_reset, resetting=True)
        if self.as_llm_data:
            raise NotImplementedError()
        return tensordict

    def _set_seed(self, seed: int | None):
        return seed
