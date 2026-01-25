"""
GFlowNet Utility Functions

This module contains utility functions for:
- Random seed setting
- State tensor conversion
"""

import random
import numpy as np
import torch

from .env import N_TIMESTEPS, MAX_LEN, VOCAB_SIZE, ALPHABET


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_to_tensor(state: list) -> torch.Tensor:
    """
    Convert state to tensor representation for neural network input.
    
    The state is encoded as:
    - One-hot encoding for timestep (size: N_TIMESTEPS)
    - One-hot encoding for each position in sequence (size: MAX_LEN * (VOCAB_SIZE + 1))
    
    Args:
        state: [timestep, sequence] where sequence is a list like ['A', 'B', 'ε', 'ε']
    
    Returns:
        Tensor of shape (N_TIMESTEPS + MAX_LEN * (VOCAB_SIZE + 1),)
    """
    timestep, seq = state
    
    # Create one-hot encoding for timestep
    time_tensor = torch.zeros(N_TIMESTEPS)
    time_tensor[timestep] = 1
    
    # Create sequence tensor of shape (MAX_LEN, VOCAB_SIZE + 1)
    # +1 for epsilon character
    seq_tensor = torch.zeros(MAX_LEN, VOCAB_SIZE + 1)
    
    # For each position in the sequence
    for i, char in enumerate(seq):
        if char == 'ε':
            # Last index is for epsilon
            seq_tensor[i, -1] = 1
        else:
            # Convert A->0, B->1, C->2
            char_idx = ord(char) - ord('A')
            seq_tensor[i, char_idx] = 1
            
    # Concatenate time and sequence tensors
    return torch.cat([time_tensor, seq_tensor.flatten()])


def get_input_size() -> int:
    """Return the size of the state tensor input."""
    return N_TIMESTEPS + MAX_LEN * (VOCAB_SIZE + 1)
