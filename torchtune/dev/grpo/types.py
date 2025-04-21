# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
import torch
from typing import Dict, Optional, Any

@dataclass
class GRPOStats:
    """
    Contains statistics computed during a GRPO (Generalized Reward Policy Optimization)
    training step.

    Attributes:
        loss (torch.Tensor): The total scalar loss for the GRPO step.
        policy_loss (torch.Tensor): The scalar component of the loss related to the
            policy objective.
        kl_loss (torch.Tensor): The scalar component of the loss penalizing KL divergence
            from the reference policy.
        ratios (torch.Tensor): The scalar mean of the importance sampling ratios
            (pi_theta / pi_ref).
        clipfrac (torch.Tensor): The scalar fraction of ratios that were clipped
            according to the PPO clipping mechanism.
        approx_policy_kls (torch.Tensor): A scalar estimate of the KL divergence between
            the policy before and after the optimization step.
        metadata (Optional[Dict[str, Any]]): A dictionary containing additional data for
            debugging or logging purposes (e.g., policy log probabilities `pi_logprobs`).
    """
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None
