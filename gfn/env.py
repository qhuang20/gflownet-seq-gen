"""
GFlowNet Sequence Environment

This module defines the sequence generation environment including:
- State space (sequences of characters)
- Action space (insert, delete, mutate operations)
- Transition functions
- Action masks for valid actions
"""

import numpy as np
import torch
from typing import List, Tuple, Set

# =============================================================================
# Global Constants
# =============================================================================

ALPHABET = ['A', 'B', 'C']
N_TIMESTEPS = 5  # t=0 to t=4
VOCAB_SIZE = len(ALPHABET)
MAX_LEN = N_TIMESTEPS - 1
MAX_ACTIONS = MAX_LEN * VOCAB_SIZE + MAX_LEN + MAX_LEN * VOCAB_SIZE  # 28 actions

# =============================================================================
# Actions List
# =============================================================================

def build_actions_list() -> List[Tuple]:
    """Build the list of all possible actions."""
    actions = []
    # Insertions: (insert, position, character)
    for pos in range(MAX_LEN):
        for char in ALPHABET:
            actions.append(('insert', pos, char))
    # Deletions: (delete, position)
    for pos in range(MAX_LEN):
        actions.append(('delete', pos))
    # Mutations: (mutate, position, character)
    for pos in range(MAX_LEN):
        for char in ALPHABET:
            actions.append(('mutate', pos, char))
    return actions

ACTIONS_LIST = build_actions_list()

# =============================================================================
# State Transition Functions
# =============================================================================

def get_next_states(curr_seq: str) -> List[str]:
    """
    Get all possible next states given current sequence.
    Used for enumerating the full state space.
    
    Args:
        curr_seq: Current sequence as a string (e.g., 'ABC' or '' for empty)
    
    Returns:
        List of all possible next sequences
    """
    next_states = []
    
    # Insertions - can insert at any position including start and end
    for pos in range(len(curr_seq) + 1):
        for char in ALPHABET:
            new_seq = curr_seq[:pos] + char + curr_seq[pos:]
            next_states.append(new_seq)
    
    # Deletions - can delete any existing character
    if curr_seq:  # only if sequence is not empty
        for pos in range(len(curr_seq)):
            new_seq = curr_seq[:pos] + curr_seq[pos+1:]
            next_states.append(new_seq)
    
    # Mutations - can change any existing character
    for pos in range(len(curr_seq)):
        for char in ALPHABET:
            new_seq = curr_seq[:pos] + char + curr_seq[pos+1:]
            next_states.append(new_seq)
            
    return next_states


def perform_action(state: List, action_idx: int) -> List:
    """
    Perform an action on a state and return the new state.
    
    Args:
        state: [timestep, sequence] where sequence is a list like ['A', 'B', 'ε', 'ε']
        action_idx: Index into ACTIONS_LIST
    
    Returns:
        New state [timestep + 1, new_sequence]
    """
    timestep, sequence = state
    action = ACTIONS_LIST[action_idx]
    action_type = action[0]
    new_sequence = sequence.copy()
    
    if action_type == 'insert':
        _, insert_pos, char = action
        # Shift elements right starting from insert position
        for i in range(len(sequence) - 1, insert_pos, -1):
            new_sequence[i] = new_sequence[i - 1]
        new_sequence[insert_pos] = char
        
    elif action_type == 'delete':
        _, del_pos = action
        # Shift elements left starting from delete position
        for i in range(del_pos, len(sequence) - 1):
            new_sequence[i] = new_sequence[i + 1]
        new_sequence[-1] = 'ε'  # Fill last position with epsilon
        
    else:  # mutate
        _, mut_pos, char = action
        new_sequence[mut_pos] = char
        
    return [timestep + 1, new_sequence]


def infer_action_id(current_state: List, next_state: List) -> int:
    """
    Infer the action index that led from current_state to next_state.
    
    Args:
        current_state: Starting state
        next_state: Resulting state
    
    Returns:
        Action index
    
    Raises:
        ValueError: If no valid action found
    """
    for idx in range(MAX_ACTIONS):
        if perform_action(current_state, idx) == next_state:
            return idx
    raise ValueError("No valid action found between states")


# =============================================================================
# Action Masks
# =============================================================================

def calculate_forward_mask(seq: List[str]) -> torch.Tensor:
    """
    Calculate forward action mask to prevent invalid actions.
    
    Args:
        seq: Current sequence as list (e.g., ['A', 'B', 'ε', 'ε'])
    
    Returns:
        Boolean tensor of shape (MAX_ACTIONS,) where True = valid action
    """
    mask = np.zeros(MAX_ACTIONS)
    
    # Get current sequence length (excluding epsilon)
    seq_len = len([x for x in seq if x != 'ε'])
    
    # Handle insertions
    if seq_len < MAX_LEN:
        for pos in range(seq_len + 1):
            for char_idx, char in enumerate(ALPHABET):
                action_idx = pos * VOCAB_SIZE + char_idx
                mask[action_idx] = 1
                
    # Handle deletions
    deletion_offset = VOCAB_SIZE * MAX_LEN
    if seq_len > 0:
        for pos in range(seq_len):
            mask[deletion_offset + pos] = 1
            
    # Handle mutations
    mutation_offset = deletion_offset + MAX_LEN
    for pos in range(seq_len):
        current_char = seq[pos]
        if current_char != 'ε':
            for char_idx, char in enumerate(ALPHABET):
                action_idx = mutation_offset + pos * VOCAB_SIZE + char_idx
                mask[action_idx] = 1

    return torch.tensor(mask, dtype=torch.bool)


def calculate_backward_mask(timestep: int, seq: List[str]) -> torch.Tensor:
    """
    Calculate backward mask considering the timestep and possible parent states.
    
    Args:
        timestep: Current timestep (0-based)
        seq: Current sequence as list
    
    Returns:
        Boolean tensor of shape (MAX_ACTIONS,) where True = valid backward action
    """
    seq_len = len([c for c in seq if c != 'ε'])
    mask = [0] * MAX_ACTIONS
    
    # Calculate offsets
    insertion_offset = 0
    deletion_offset = MAX_LEN * VOCAB_SIZE
    mutation_offset = deletion_offset + MAX_LEN

    # At t=0, no backward actions possible (root state)
    if timestep == 0:
        return torch.tensor(mask, dtype=torch.bool)
    
    # Max length at previous timestep
    max_prev_len = timestep - 1
    
    # Special case: at t=2 with one character sequence, only mutations allowed
    if timestep == 2 and seq_len == 1:
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(ALPHABET):
                    mask[mutation_offset + pos * VOCAB_SIZE + char_idx] = 1
        return torch.tensor(mask, dtype=torch.bool)
    
    # If current sequence length > max_prev_len, only deletions possible
    if seq_len > max_prev_len:
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
                
    # If current sequence length == max_prev_len, mutations and deletions possible
    elif seq_len == max_prev_len:
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(ALPHABET):
                    mask[mutation_offset + pos * VOCAB_SIZE + char_idx] = 1
                    
    # If current sequence length < max_prev_len, all actions possible
    else:
        for pos in range(seq_len + 1):
            for char_idx in range(VOCAB_SIZE):
                mask[insertion_offset + pos * VOCAB_SIZE + char_idx] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(ALPHABET):
                    mask[mutation_offset + pos * VOCAB_SIZE + char_idx] = 1

    return torch.tensor(mask, dtype=torch.bool)


# =============================================================================
# State Space Generation
# =============================================================================

def generate_all_states() -> List[List[str]]:
    """
    Generate all possible states at each timestep.
    
    Returns:
        List of lists, where all_states[t] contains all possible sequences at time t
    """
    all_states = [set() for _ in range(N_TIMESTEPS)]
    all_states[0].add('')  # Start with empty sequence

    for t in range(N_TIMESTEPS - 1):
        for curr_seq in all_states[t]:
            next_states = get_next_states(curr_seq)
            all_states[t + 1].update(next_states)

    # Convert sets to sorted lists for consistent ordering
    return [sorted(list(states)) for states in all_states]


def get_initial_state() -> List:
    """Return the initial (root) state."""
    return [0, ['ε'] * MAX_LEN]


def state_to_string(state: List) -> str:
    """Convert state to unpadded string representation."""
    return ''.join(s for s in state[1] if s != 'ε')
