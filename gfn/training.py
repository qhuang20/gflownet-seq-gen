"""
GFlowNet Training

This module contains training functions for different GFlowNet objectives:
- TB (Trajectory Balance)
- DB (Detailed Balance)
- FL-DB (Forward-Looking Detailed Balance)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Union

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
from .model import TBModel, DBModel
from .losses import (
    trajectory_balance_loss,
    detailed_balance_loss,
    forward_looking_db_loss,
)


@dataclass
class TrainingConfig:
    """Configuration for GFlowNet training."""
    
    seed: int = 42
    n_hid_units: int = 32
    n_episodes: int = 20_000
    learning_rate: float = 3e-3
    update_freq: int = 4
    uniform_backward: bool = True
    replay_freq: float = 0.0
    
    # Objective: "TB", "DB", or "FLDB"
    objective: str = "TB"
    
    # Derived
    n_action_steps: int = field(init=False)
    
    def __post_init__(self):
        self.n_action_steps = N_TIMESTEPS - 1
        assert self.objective in ["TB", "DB", "FLDB"], \
            f"Unknown objective: {self.objective}. Use 'TB', 'DB', or 'FLDB'"


@dataclass
class TrainingResult:
    """Results from training."""
    
    model: Union[TBModel, DBModel]
    losses: List[float]
    logZs: List[float]
    sampled_states: List[List]
    objective: str = "TB"
    
    @property
    def final_Z(self) -> float:
        """Get the final partition function estimate."""
        return np.exp(self.logZs[-1])


# =============================================================================
# TB Training (Original)
# =============================================================================

def sample_trajectory_tb(
    model: TBModel,
    reward_fn: Callable,
    config: TrainingConfig,
    replay_buffer: Optional[List] = None,
) -> Tuple[List, float, float, float]:
    """Sample trajectory for TB objective."""
    state = get_initial_state()
    P_F, _ = model(state_to_tensor(state))
    total_log_P_F, total_log_P_B = 0.0, 0.0
    
    use_replay = replay_buffer and random.random() < config.replay_freq
    if use_replay:
        traj = random.choice(replay_buffer)
    
    for t in range(config.n_action_steps):
        mask = calculate_forward_mask(state[1])
        P_F_masked = torch.where(mask, P_F, torch.tensor(-100.0))
        categorical = Categorical(logits=P_F_masked)
        
        if use_replay:
            action_idx = torch.tensor(traj[t][-1])
        else:
            action_idx = categorical.sample()
        
        total_log_P_F += categorical.log_prob(action_idx)
        
        new_state = perform_action(state, action_idx.item())
        P_F, _ = model(state_to_tensor(new_state))
        
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
    
    reward = reward_fn(state[1])
    return state, total_log_P_F, total_log_P_B, reward


def train_tb(
    reward_fn: Callable,
    config: TrainingConfig,
    replay_buffer: Optional[List] = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train using Trajectory Balance objective."""
    set_seed(config.seed)
    
    model = TBModel(config.n_hid_units, config.uniform_backward)
    # logZ is already in model.parameters(), no need to add separately
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    
    losses, logZs, sampled_states = [], [], []
    minibatch_loss = torch.tensor(0.0)
    
    iterator = range(config.n_episodes)
    if verbose:
        iterator = tqdm(iterator, ncols=60, desc="TB Training")
    
    for episode in iterator:
        state, log_P_F, log_P_B, reward = sample_trajectory_tb(
            model, reward_fn, config, replay_buffer
        )
        
        loss = trajectory_balance_loss(
            model.logZ, log_P_F, log_P_B, torch.tensor(reward).float()
        )
        minibatch_loss = minibatch_loss + loss
        sampled_states.append(state)
        
        if episode % config.update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = torch.tensor(0.0)
    
    return TrainingResult(
        model=model, losses=losses, logZs=logZs,
        sampled_states=sampled_states, objective="TB"
    )


# =============================================================================
# DB / FL-DB Training
# =============================================================================

def sample_trajectory_db(
    model: DBModel,
    reward_fn: Callable,
    config: TrainingConfig,
    use_fldb: bool = False,
) -> Tuple[List, List, List, List, List, float]:
    """
    Sample trajectory for DB/FL-DB objective.
    
    Returns:
        Tuple of (final_state, log_flows, log_P_Fs, log_P_Bs, trajectory_rewards, terminal_reward)
    """
    state = get_initial_state()
    
    log_flows = []
    log_P_Fs = []
    log_P_Bs = []
    trajectory_rewards = []
    
    for t in range(config.n_action_steps):
        state_tensor = state_to_tensor(state)
        P_F, P_B, log_F = model(state_tensor)
        log_flows.append(log_F)
        
        # Forward action
        mask = calculate_forward_mask(state[1])
        P_F_masked = torch.where(mask, P_F, torch.tensor(-100.0))
        categorical = Categorical(logits=P_F_masked)
        action_idx = categorical.sample()
        log_P_Fs.append(categorical.log_prob(action_idx))
        
        # Transition
        new_state = perform_action(state, action_idx.item())
        
        # Backward probability
        if config.uniform_backward:
            mask = calculate_backward_mask(new_state[0], new_state[1])
            valid_actions = mask.sum()
            log_P_Bs.append(-torch.log(valid_actions.float()))
        else:
            _, P_B_new, _ = model(state_to_tensor(new_state))
            mask = calculate_backward_mask(new_state[0], new_state[1])
            P_B_masked = torch.where(mask, P_B_new, torch.tensor(-100.0))
            log_P_Bs.append(Categorical(logits=P_B_masked).log_prob(action_idx))
        
        # Trajectory reward for FL-DB (using terminal reward for all steps as default)
        if use_fldb:
            # For FL-DB, we use uniform intermediate reward of 1.0 (log = 0)
            # The real reward comes at terminal state
            trajectory_rewards.append(torch.tensor(0.0))  # log(1.0) = 0
        
        state = new_state
    
    # Get flow at terminal state (will be replaced by reward in loss)
    _, _, log_F_terminal = model(state_to_tensor(state))
    log_flows.append(log_F_terminal)
    
    terminal_reward = reward_fn(state[1])
    
    return state, log_flows, log_P_Fs, log_P_Bs, trajectory_rewards, terminal_reward


def train_db(
    reward_fn: Callable,
    config: TrainingConfig,
    use_fldb: bool = False,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train using Detailed Balance or Forward-Looking DB objective.
    
    Args:
        reward_fn: Reward function
        config: Training configuration
        use_fldb: If True, use FL-DB; otherwise use standard DB
        verbose: Show progress bar
    """
    set_seed(config.seed)
    
    model = DBModel(config.n_hid_units, config.uniform_backward)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    
    losses, logZs, sampled_states = [], [], []
    minibatch_loss = torch.tensor(0.0)
    
    objective_name = "FL-DB" if use_fldb else "DB"
    iterator = range(config.n_episodes)
    if verbose:
        iterator = tqdm(iterator, ncols=60, desc=f"{objective_name} Training")
    
    for episode in iterator:
        state, log_flows, log_P_Fs, log_P_Bs, traj_rewards, terminal_reward = \
            sample_trajectory_db(model, reward_fn, config, use_fldb)
        
        # Compute DB/FL-DB loss for each transition
        episode_loss = torch.tensor(0.0)
        log_terminal_reward = torch.log(torch.tensor(terminal_reward).float()).clamp(min=-20.0)
        
        for t in range(config.n_action_steps):
            log_F_s = log_flows[t]
            log_P_F = log_P_Fs[t]
            log_P_B = log_P_Bs[t]
            
            # For last transition, use terminal reward
            if t == config.n_action_steps - 1:
                log_F_s_next = log_terminal_reward
            else:
                log_F_s_next = log_flows[t + 1]
            
            if use_fldb:
                # FL-DB: include trajectory reward (0 for intermediate, terminal for last)
                log_traj_r = traj_rewards[t] if t < len(traj_rewards) else torch.tensor(0.0)
                step_loss = forward_looking_db_loss(
                    log_F_s, log_P_F, log_P_B, log_F_s_next, log_traj_r
                )
            else:
                step_loss = detailed_balance_loss(
                    log_F_s, log_P_F, log_P_B, log_F_s_next
                )
            
            episode_loss = episode_loss + step_loss
        
        minibatch_loss = minibatch_loss + episode_loss
        sampled_states.append(state)
        
        if episode % config.update_freq == 0:
            losses.append(minibatch_loss.item())
            # log Z = log F(s0)
            with torch.no_grad():
                init_state = get_initial_state()
                _, _, log_F_init = model(state_to_tensor(init_state))
                logZs.append(log_F_init.item())
            
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = torch.tensor(0.0)
    
    return TrainingResult(
        model=model, losses=losses, logZs=logZs,
        sampled_states=sampled_states, objective=objective_name
    )


# =============================================================================
# Unified Training Interface
# =============================================================================

def train(
    reward_fn: Callable,
    config: Optional[TrainingConfig] = None,
    replay_buffer: Optional[List] = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train a GFlowNet model using specified objective.
    
    Args:
        reward_fn: Reward function that takes a sequence and returns a float
        config: Training configuration (uses defaults if None)
        replay_buffer: Optional replay buffer (only for TB)
        verbose: Whether to show progress bar
    
    Returns:
        TrainingResult containing trained model and metrics
    """
    if config is None:
        config = TrainingConfig()
    
    if config.objective == "TB":
        return train_tb(reward_fn, config, replay_buffer, verbose)
    elif config.objective == "DB":
        return train_db(reward_fn, config, use_fldb=False, verbose=verbose)
    elif config.objective == "FLDB":
        return train_db(reward_fn, config, use_fldb=True, verbose=verbose)
    else:
        raise ValueError(f"Unknown objective: {config.objective}")


# =============================================================================
# Utility Functions
# =============================================================================

def get_policy_probs(
    model: Union[TBModel, DBModel],
    state: List,
) -> torch.Tensor:
    """Get forward policy probabilities for a given state."""
    with torch.no_grad():
        output = model(state_to_tensor(state))
        P_F = output[0]  # Works for both TBModel and DBModel
        mask = calculate_forward_mask(state[1])
        P_F = torch.where(mask, P_F, torch.tensor(-100.0))
        probs = Categorical(logits=P_F).probs
    return probs


def generate_greedy_trajectory(model: Union[TBModel, DBModel]) -> List[List]:
    """Generate the most likely trajectory by greedily selecting highest probability actions."""
    state = get_initial_state()
    trajectory = [state]
    
    for _ in range(MAX_LEN):
        probs = get_policy_probs(model, state)
        best_action = torch.argmax(probs).item()
        state = perform_action(state, best_action)
        trajectory.append(state)
    
    return trajectory


# Legacy alias for backward compatibility
sample_trajectory = sample_trajectory_tb
