"""
GFlowNet Models

This module contains models for GFlowNet:
- TBModel: Trajectory Balance model (learns global logZ)
- DBModel: Detailed Balance model (learns F(s) per state)
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


class DBModel(nn.Module):
    """
    Detailed Balance Model for GFlowNet.
    
    Predicts forward policy P_F, backward policy P_B, and state flow F(s).
    Used for both DB and FL-DB objectives.
    """
    
    def __init__(
        self,
        n_hid: int = 32,
        uniform_backward: bool = False
    ):
        """
        Args:
            n_hid: Number of hidden units
            uniform_backward: If True, use uniform backward policy
        """
        super().__init__()
        
        self.uniform_backward = uniform_backward
        input_size = get_input_size()
        
        # Output: P_F + P_B + log F(s)
        # If uniform backward, we still output P_B slot but won't use it
        output_size = MAX_ACTIONS + MAX_ACTIONS + 1
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, output_size),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: State tensor from state_to_tensor()
        
        Returns:
            Tuple of (P_F_logits, P_B_logits, log_F)
            - P_F_logits: Forward policy logits
            - P_B_logits: Backward policy logits (zeros if uniform)
            - log_F: Log flow for this state
        """
        output = self.mlp(x)
        
        P_F = output[..., :MAX_ACTIONS]
        P_B_raw = output[..., MAX_ACTIONS:2*MAX_ACTIONS]
        log_F = output[..., -1]
        
        if self.uniform_backward:
            P_B = torch.zeros_like(P_B_raw)
        else:
            P_B = P_B_raw

        return P_F, P_B, log_F
    
    @property
    def logZ(self) -> torch.Tensor:
        """
        For compatibility with TB interface.
        Returns log F(s0) which equals log Z.
        Note: This requires running forward pass on initial state.
        """
        raise NotImplementedError(
            "DBModel doesn't have a global logZ. "
            "Use log_F from forward pass on initial state instead."
        )


# Legacy import support - loss function moved to losses.py
def trajectory_balance_loss(
    logZ: torch.Tensor,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    reward: torch.Tensor
) -> torch.Tensor:
    """
    Compute Trajectory Balance loss.
    
    Note: This function is kept for backward compatibility.
    Consider using gfn.losses.trajectory_balance_loss instead.
    """
    from .losses import trajectory_balance_loss as tb_loss
    return tb_loss(logZ, log_P_F, log_P_B, reward)
