"""GFlowNet models: TBModel (Trajectory Balance) and DBModel (Detailed Balance)."""

import torch
import torch.nn as nn
from typing import List, Union, Optional

from .env import get_env_config
from .utils import get_input_size


def build_mlp(input_size, output_size, hidden_layers, activation=nn.LeakyReLU):
    """Build an MLP with configurable hidden layers."""
    layers = []
    prev_size = input_size

    for hidden_size in hidden_layers:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(activation())
        prev_size = hidden_size

    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


class TBModel(nn.Module):
    """Trajectory Balance model. Learns P_F, optional P_B, and global logZ."""

    def __init__(self, n_hid=32, uniform_backward=True):
        super().__init__()
        self.uniform_backward = uniform_backward
        config = get_env_config()
        input_size = get_input_size()
        max_actions = config.max_actions

        if isinstance(n_hid, int):
            hidden_layers = [n_hid]
        else:
            hidden_layers = list(n_hid)
        self.hidden_layers = hidden_layers

        output_size = max_actions if uniform_backward else 2 * max_actions
        self.mlp = build_mlp(input_size, output_size, hidden_layers)
        self.logZ = nn.Parameter(torch.ones(1))
        self._max_actions = max_actions

    def forward(self, x):
        logits = self.mlp(x)
        if self.uniform_backward:
            P_F = logits
            P_B = torch.zeros_like(P_F)
        else:
            P_F = logits[..., :self._max_actions]
            P_B = logits[..., self._max_actions:]
        return P_F, P_B

    def __repr__(self):
        return f"TBModel(hidden_layers={self.hidden_layers}, uniform_backward={self.uniform_backward})"


class DBModel(nn.Module):
    """Detailed Balance model. Learns P_F, P_B, and per-state flow F(s)."""

    def __init__(self, n_hid=32, uniform_backward=False):
        super().__init__()
        self.uniform_backward = uniform_backward
        config = get_env_config()
        input_size = get_input_size()
        max_actions = config.max_actions

        if isinstance(n_hid, int):
            hidden_layers = [n_hid, n_hid]
        else:
            hidden_layers = list(n_hid)
        self.hidden_layers = hidden_layers

        output_size = max_actions + max_actions + 1  # P_F + P_B + log F(s)
        self.mlp = build_mlp(input_size, output_size, hidden_layers)
        self._max_actions = max_actions

    def forward(self, x):
        output = self.mlp(x)
        P_F = output[..., :self._max_actions]
        P_B_raw = output[..., self._max_actions:2*self._max_actions]
        log_F = output[..., -1]

        if self.uniform_backward:
            P_B = torch.zeros_like(P_B_raw)
        else:
            P_B = P_B_raw
        return P_F, P_B, log_F

    @property
    def logZ(self):
        raise NotImplementedError(
            "DBModel doesn't have a global logZ. "
            "Use log_F from forward pass on initial state instead."
        )

    def __repr__(self):
        return f"DBModel(hidden_layers={self.hidden_layers}, uniform_backward={self.uniform_backward})"


# Legacy import support
def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward):
    from .losses import trajectory_balance_loss as tb_loss
    return tb_loss(logZ, log_P_F, log_P_B, reward)
