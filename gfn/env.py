"""Sequence generation environment: state space, action space, and transition functions."""

import numpy as np
import torch
from typing import List, Tuple, Set
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the sequence environment."""
    alphabet: List[str]
    max_seq_len: int

    @property
    def n_timesteps(self) -> int:
        return self.max_seq_len + 1

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def max_actions(self) -> int:
        """insertions + deletions + mutations"""
        return (self.max_seq_len * self.vocab_size +
                self.max_seq_len +
                self.max_seq_len * self.vocab_size)


PRESETS = {
    'toy': EnvConfig(alphabet=['A', 'B', 'C'], max_seq_len=4),
    'rna': EnvConfig(alphabet=['A', 'U', 'G', 'C'], max_seq_len=8),
    'dna': EnvConfig(alphabet=['A', 'T', 'G', 'C'], max_seq_len=8),
    'rna_long': EnvConfig(alphabet=['A', 'U', 'G', 'C'], max_seq_len=12),
    'let7_pilot': EnvConfig(alphabet=['A', 'U', 'G', 'C'], max_seq_len=10),
    'let7_medium': EnvConfig(alphabet=['A', 'U', 'G', 'C'], max_seq_len=15),
    'let7_full': EnvConfig(alphabet=['A', 'U', 'G', 'C'], max_seq_len=22),
}

# Active configuration (global state)
_active_config: EnvConfig = PRESETS['toy']


def set_env_config(config: EnvConfig) -> None:
    """Set the active environment configuration."""
    global _active_config, ALPHABET, N_TIMESTEPS, VOCAB_SIZE, MAX_LEN, MAX_ACTIONS, ACTIONS_LIST
    _active_config = config
    ALPHABET = config.alphabet
    N_TIMESTEPS = config.n_timesteps
    VOCAB_SIZE = config.vocab_size
    MAX_LEN = config.max_seq_len
    MAX_ACTIONS = config.max_actions
    ACTIONS_LIST = build_actions_list()


def use_preset(name: str) -> EnvConfig:
    """Use a preset configuration."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    set_env_config(PRESETS[name])
    return _active_config


def get_env_config() -> EnvConfig:
    """Get the current environment configuration."""
    return _active_config


# Global constants (for backward compatibility)
ALPHABET = _active_config.alphabet
N_TIMESTEPS = _active_config.n_timesteps
VOCAB_SIZE = _active_config.vocab_size
MAX_LEN = _active_config.max_seq_len
MAX_ACTIONS = _active_config.max_actions


def build_actions_list() -> List[Tuple]:
    """Build the list of all possible actions based on current config."""
    config = _active_config
    actions = []
    for pos in range(config.max_seq_len):
        for char in config.alphabet:
            actions.append(('insert', pos, char))
    for pos in range(config.max_seq_len):
        actions.append(('delete', pos))
    for pos in range(config.max_seq_len):
        for char in config.alphabet:
            actions.append(('mutate', pos, char))
    return actions


ACTIONS_LIST = build_actions_list()


def get_next_states(curr_seq: str) -> List[str]:
    """Get all possible next states given current sequence (for enumerating state space)."""
    config = _active_config
    next_states = []

    for pos in range(len(curr_seq) + 1):
        for char in config.alphabet:
            new_seq = curr_seq[:pos] + char + curr_seq[pos:]
            next_states.append(new_seq)

    if curr_seq:
        for pos in range(len(curr_seq)):
            new_seq = curr_seq[:pos] + curr_seq[pos+1:]
            next_states.append(new_seq)

    for pos in range(len(curr_seq)):
        for char in config.alphabet:
            new_seq = curr_seq[:pos] + char + curr_seq[pos+1:]
            next_states.append(new_seq)

    return next_states


def perform_action(state: List, action_idx: int) -> List:
    """Perform an action on a state, return new state [timestep+1, new_sequence]."""
    config = _active_config
    timestep, sequence = state
    action = ACTIONS_LIST[action_idx]
    action_type = action[0]
    new_sequence = sequence.copy()

    if action_type == 'insert':
        _, insert_pos, char = action
        for i in range(len(sequence) - 1, insert_pos, -1):
            new_sequence[i] = new_sequence[i - 1]
        new_sequence[insert_pos] = char

    elif action_type == 'delete':
        _, del_pos = action
        for i in range(del_pos, len(sequence) - 1):
            new_sequence[i] = new_sequence[i + 1]
        new_sequence[-1] = 'ε'

    else:  # mutate
        _, mut_pos, char = action
        new_sequence[mut_pos] = char

    return [timestep + 1, new_sequence]


def infer_action_id(current_state: List, next_state: List) -> int:
    """Infer the action index that led from current_state to next_state."""
    config = _active_config
    for idx in range(config.max_actions):
        if perform_action(current_state, idx) == next_state:
            return idx
    raise ValueError("No valid action found between states")


def calculate_forward_mask(seq: List[str], insert_only: bool = False) -> torch.Tensor:
    """Calculate forward action mask to prevent invalid actions."""
    config = _active_config
    mask = np.zeros(config.max_actions)
    seq_len = len([x for x in seq if x != 'ε'])

    # Insertions
    if seq_len < config.max_seq_len:
        for pos in range(seq_len + 1):
            for char_idx, char in enumerate(config.alphabet):
                action_idx = pos * config.vocab_size + char_idx
                mask[action_idx] = 1

    if not insert_only:
        # Deletions
        deletion_offset = config.vocab_size * config.max_seq_len
        if seq_len > 0:
            for pos in range(seq_len):
                mask[deletion_offset + pos] = 1

        # Mutations
        mutation_offset = deletion_offset + config.max_seq_len
        for pos in range(seq_len):
            current_char = seq[pos]
            if current_char != 'ε':
                for char_idx, char in enumerate(config.alphabet):
                    action_idx = mutation_offset + pos * config.vocab_size + char_idx
                    mask[action_idx] = 1

    return torch.tensor(mask, dtype=torch.bool)


def calculate_backward_mask(timestep: int, seq: List[str]) -> torch.Tensor:
    """Calculate backward mask considering timestep and possible parent states."""
    config = _active_config
    seq_len = len([c for c in seq if c != 'ε'])
    mask = [0] * config.max_actions

    insertion_offset = 0
    deletion_offset = config.max_seq_len * config.vocab_size
    mutation_offset = deletion_offset + config.max_seq_len

    if timestep == 0:
        return torch.tensor(mask, dtype=torch.bool)

    max_prev_len = timestep - 1

    if timestep == 2 and seq_len == 1:
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(config.alphabet):
                    mask[mutation_offset + pos * config.vocab_size + char_idx] = 1
        return torch.tensor(mask, dtype=torch.bool)

    if seq_len > max_prev_len:
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1

    elif seq_len == max_prev_len:
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(config.alphabet):
                    mask[mutation_offset + pos * config.vocab_size + char_idx] = 1

    else:
        for pos in range(seq_len + 1):
            for char_idx in range(config.vocab_size):
                mask[insertion_offset + pos * config.vocab_size + char_idx] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(config.alphabet):
                    mask[mutation_offset + pos * config.vocab_size + char_idx] = 1

    return torch.tensor(mask, dtype=torch.bool)


def generate_all_states() -> List[List[str]]:
    """Generate all possible states at each timestep."""
    config = _active_config
    all_states = [set() for _ in range(config.n_timesteps)]
    all_states[0].add('')

    for t in range(config.n_timesteps - 1):
        for curr_seq in all_states[t]:
            next_states = get_next_states(curr_seq)
            all_states[t + 1].update(next_states)

    return [sorted(list(states)) for states in all_states]


def get_initial_state() -> List:
    """Return the initial (root) state."""
    config = _active_config
    return [0, ['ε'] * config.max_seq_len]


def state_to_string(state: List) -> str:
    """Convert state to unpadded string representation."""
    return ''.join(s for s in state[1] if s != 'ε')


def print_env_info() -> None:
    """Print current environment configuration."""
    config = _active_config
    print(f"Environment Configuration:")
    print(f"  Alphabet: {config.alphabet}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Timesteps: {config.n_timesteps}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max actions: {config.max_actions}")
