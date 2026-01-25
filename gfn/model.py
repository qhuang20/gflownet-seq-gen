"""
GFlowNet Model

This module contains the Trajectory Balance (TB) model for GFlowNet.
"""

import torch
import torch.nn as nn

from .env import MAX_ACTIONS
from .utils import get_input_size


class TBModel(nn.Module):
    """
    Trajectory Balance Model for GFlowNet.
    
    Predicts forward policy P_F and optionally backward policy P_B.
    Also learns the log partition function logZ.
    """
    
    def __init__(
        self,
        n_hid: int = 32,
        uniform_backward: bool = True
    ):
        """
        Args:
            n_hid: Number of hidden units
            uniform_backward: If True, use uniform backward policy (simpler)
        """
        super().__init__()
        
        self.uniform_backward = uniform_backward
        input_size = get_input_size()
        
        # Output size: P_F only if uniform backward, otherwise P_F + P_B
        output_size = MAX_ACTIONS if uniform_backward else 2 * MAX_ACTIONS
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, output_size),
        )
        
        # Log partition function (learnable scalar)
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: State tensor from state_to_tensor()
        
        Returns:
            Tuple of (P_F_logits, P_B_logits)
            If uniform_backward=True, P_B is zeros (placeholder)
        """
        logits = self.mlp(x)
        
        if self.uniform_backward:
            P_F = logits
            P_B = torch.zeros_like(P_F)
        else:
            P_F = logits[..., :MAX_ACTIONS]
            P_B = logits[..., MAX_ACTIONS:]

        return P_F, P_B


def trajectory_balance_loss(
    logZ: torch.Tensor,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    reward: torch.Tensor
) -> torch.Tensor:
    """
    Compute Trajectory Balance loss.
    
    The TB objective is:
        Z * P_F(τ) = R(x) * P_B(τ)
    
    In log space:
        log Z + log P_F(τ) = log R(x) + log P_B(τ)
    
    Loss is the squared difference:
        (log Z + log P_F - log R - log P_B)²
    
    Args:
        logZ: Log partition function (learnable)
        log_P_F: Sum of log forward probabilities along trajectory
        log_P_B: Sum of log backward probabilities along trajectory
        reward: Terminal reward R(x)
    
    Returns:
        Scalar loss value
    """
    # Clip log(reward) to avoid log(0)
    log_reward = torch.log(reward).clamp(min=-20.0)
    loss = (logZ + log_P_F - log_reward - log_P_B).pow(2)
    return loss
