"""
GFlowNet Training

This module contains the training loop and related functions.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from .env import (
    MAX_LEN, N_TIMESTEPS,
    ACTIONS_LIST,
    perform_action,
    calculate_forward_mask,
    calculate_backward_mask,
    get_initial_state,
)
from .utils import state_to_tensor, set_seed
from .model import TBModel, trajectory_balance_loss


@dataclass
class TrainingConfig:
    """Configuration for GFlowNet training."""
    
    seed: int = 42
    n_hid_units: int = 32
    n_episodes: int = 20_000
    learning_rate: float = 3e-3
    update_freq: int = 4
    uniform_backward: bool = True
    replay_freq: float = 0.0  # Probability of using replay buffer
    
    # Derived
    n_action_steps: int = field(init=False)
    
    def __post_init__(self):
        self.n_action_steps = N_TIMESTEPS - 1


@dataclass
class TrainingResult:
    """Results from training."""
    
    model: TBModel
    losses: List[float]
    logZs: List[float]
    sampled_states: List[List]
    
    @property
    def final_Z(self) -> float:
        """Get the final partition function estimate."""
        return np.exp(self.logZs[-1])


def sample_trajectory(
    model: TBModel,
    reward_fn: Callable,
    config: TrainingConfig,
    replay_buffer: Optional[List] = None,
) -> Tuple[List, float, float, float]:
    """
    Sample a single trajectory using the current policy.
    
    Args:
        model: The GFlowNet model
        reward_fn: Reward function that takes a sequence
        config: Training configuration
        replay_buffer: Optional replay buffer for off-policy learning
    
    Returns:
        Tuple of (final_state, total_log_P_F, total_log_P_B, reward)
    """
    state = get_initial_state()
    P_F, _ = model(state_to_tensor(state))
    total_log_P_F, total_log_P_B = 0.0, 0.0
    
    # Decide whether to use replay buffer
    use_replay = replay_buffer and random.random() < config.replay_freq
    if use_replay:
        traj = random.choice(replay_buffer)
    
    for t in range(config.n_action_steps):
        # Forward policy
        mask = calculate_forward_mask(state[1])
        P_F_masked = torch.where(mask, P_F, torch.tensor(-100.0))
        categorical = Categorical(logits=P_F_masked)
        
        if use_replay:
            action_idx = torch.tensor(traj[t][-1])
        else:
            action_idx = categorical.sample()
        
        total_log_P_F += categorical.log_prob(action_idx)
        
        # Transition to new state
        new_state = perform_action(state, action_idx.item())
        P_F, _ = model(state_to_tensor(new_state))
        
        # Backward policy
        if config.uniform_backward:
            mask = calculate_backward_mask(new_state[0], new_state[1])
            valid_actions = mask.sum()
            total_log_P_B += -torch.log(valid_actions.float())
        else:
            mask = calculate_backward_mask(new_state[0], new_state[1])
            _, P_B = model(state_to_tensor(new_state))
            P_B_masked = torch.where(mask, P_B, torch.tensor(-100.0))
            total_log_P_B += Categorical(logits=P_B_masked).log_prob(action_idx)
        
        state = new_state
    
    # Calculate reward at terminal state
    reward = reward_fn(state[1])
    
    return state, total_log_P_F, total_log_P_B, reward


def train(
    reward_fn: Callable,
    config: Optional[TrainingConfig] = None,
    replay_buffer: Optional[List] = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train a GFlowNet model.
    
    Args:
        reward_fn: Reward function that takes a sequence and returns a float
        config: Training configuration (uses defaults if None)
        replay_buffer: Optional replay buffer for off-policy learning
        verbose: Whether to show progress bar
    
    Returns:
        TrainingResult containing trained model and metrics
    """
    if config is None:
        config = TrainingConfig()
    
    set_seed(config.seed)
    
    model = TBModel(config.n_hid_units, config.uniform_backward)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    
    losses, logZs, sampled_states = [], [], []
    minibatch_loss = torch.tensor(0.0)
    
    iterator = range(config.n_episodes)
    if verbose:
        iterator = tqdm(iterator, ncols=60, desc="Training")
    
    for episode in iterator:
        # Sample trajectory
        state, log_P_F, log_P_B, reward = sample_trajectory(
            model, reward_fn, config, replay_buffer
        )
        
        # Compute loss
        loss = trajectory_balance_loss(
            model.logZ,
            log_P_F,
            log_P_B,
            torch.tensor(reward).float()
        )
        minibatch_loss = minibatch_loss + loss
        
        sampled_states.append(state)
        
        # Update at specified frequency
        if episode % config.update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = torch.tensor(0.0)
    
    return TrainingResult(
        model=model,
        losses=losses,
        logZs=logZs,
        sampled_states=sampled_states
    )


def get_policy_probs(
    model: TBModel,
    state: List,
) -> torch.Tensor:
    """
    Get forward policy probabilities for a given state.
    
    Args:
        model: Trained GFlowNet model
        state: Current state [timestep, sequence]
    
    Returns:
        Tensor of action probabilities
    """
    with torch.no_grad():
        P_F, _ = model(state_to_tensor(state))
        mask = calculate_forward_mask(state[1])
        P_F = torch.where(mask, P_F, torch.tensor(-100.0))
        probs = Categorical(logits=P_F).probs
    return probs


def generate_greedy_trajectory(model: TBModel) -> List[List]:
    """
    Generate the most likely trajectory by greedily selecting highest probability actions.
    
    Args:
        model: Trained GFlowNet model
    
    Returns:
        List of states forming the trajectory
    """
    state = get_initial_state()
    trajectory = [state]
    
    for _ in range(MAX_LEN):
        probs = get_policy_probs(model, state)
        best_action = torch.argmax(probs).item()
        state = perform_action(state, best_action)
        trajectory.append(state)
    
    return trajectory
