"""Utility functions: seed, tensor conversion, FASTA loading."""

import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

from .env import get_env_config


def load_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """Load sequences from a FASTA file. Returns list of (header, sequence)."""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper())

        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))

    return sequences


def load_fasta_sequences(fasta_path: str, as_rna: bool = True) -> List[str]:
    """Load sequences from FASTA (sequences only). Converts T->U if as_rna."""
    data = load_fasta(fasta_path)
    sequences = [seq for _, seq in data]
    if as_rna:
        sequences = [seq.replace('T', 'U') for seq in sequences]
    return sequences


def analyze_sequences(sequences: List[str]) -> Dict:
    """Analyze a list of sequences: lengths, unique counts, character frequencies."""
    lengths = [len(s) for s in sequences]
    unique_seqs = list(set(sequences))
    all_chars = ''.join(sequences)
    char_counts = {char: all_chars.count(char) for char in set(all_chars)}

    return {
        'n_sequences': len(sequences),
        'n_unique': len(unique_seqs),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'char_counts': char_counts,
        'unique_sequences': unique_seqs,
    }


def truncate_sequences(sequences: List[str], max_len: int, from_start: bool = True) -> List[str]:
    """Truncate sequences to max_len from start or end."""
    if from_start:
        return [seq[:max_len] for seq in sequences]
    else:
        return [seq[-max_len:] for seq in sequences]


def sequences_to_targets(sequences: List[str], max_len: int) -> List[List[str]]:
    """Convert sequence strings to target format with epsilon padding."""
    targets = []
    for seq in sequences:
        target = list(seq) + ['ε'] * (max_len - len(seq))
        targets.append(target)
    return targets


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_to_tensor(state: list) -> torch.Tensor:
    """Convert state to one-hot tensor for neural network input.

    Encoding: one-hot timestep + one-hot per position (vocab_size + 1 for epsilon).
    """
    config = get_env_config()
    timestep, seq = state

    time_tensor = torch.zeros(config.n_timesteps)
    time_tensor[timestep] = 1

    seq_tensor = torch.zeros(config.max_seq_len, config.vocab_size + 1)
    char_to_idx = {char: idx for idx, char in enumerate(config.alphabet)}

    for i, char in enumerate(seq):
        if char == 'ε':
            seq_tensor[i, -1] = 1
        elif char in char_to_idx:
            seq_tensor[i, char_to_idx[char]] = 1
        else:
            raise ValueError(f"Unknown character: {char}. Alphabet: {config.alphabet}")

    return torch.cat([time_tensor, seq_tensor.flatten()])


def get_input_size() -> int:
    """Return the input tensor size based on current config."""
    config = get_env_config()
    return config.n_timesteps + config.max_seq_len * (config.vocab_size + 1)
