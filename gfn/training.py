"""GFlowNet training: TB, DB, and FL-DB objectives."""

import random
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Union, Dict, Any

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
    objective: str = "TB"  # "TB", "DB", or "FLDB"

    n_action_steps: int = field(init=False)

    def __post_init__(self):
        self.n_action_steps = N_TIMESTEPS - 1
        assert self.objective in ["TB", "DB", "FLDB"], \
            f"Unknown objective: {self.objective}. Use 'TB', 'DB', or 'FLDB'"


@dataclass
class HitTrajectory:
    """Record of a trajectory that hit a target sequence."""

    sequence: List[str]
    iteration: int
    reward: float
    hit_count: int = 1
    batch_index: int = 0

    actions: Optional[List[int]] = None
    log_P_Fs: Optional[List[float]] = None
    log_P_Bs: Optional[List[float]] = None
    log_flows: Optional[List[float]] = None
    intermediate_rewards: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'sequence': self.sequence,
            'sequence_str': ''.join(c for c in self.sequence if c != 'ε'),
            'iteration': self.iteration,
            'reward': self.reward,
            'hit_count': self.hit_count,
            'batch_index': self.batch_index,
        }
        if self.actions is not None:
            result['actions'] = self.actions
        if self.log_P_Fs is not None:
            result['log_P_Fs'] = self.log_P_Fs
        if self.log_P_Bs is not None:
            result['log_P_Bs'] = self.log_P_Bs
        if self.log_flows is not None:
            result['log_flows'] = self.log_flows
        if self.intermediate_rewards is not None:
            result['intermediate_rewards'] = self.intermediate_rewards
        return result

    @classmethod
    def from_dict(cls, d):
        return cls(
            sequence=d['sequence'],
            iteration=d['iteration'],
            reward=d['reward'],
            hit_count=d.get('hit_count', 1),
            batch_index=d.get('batch_index', 0),
            actions=d.get('actions'),
            log_P_Fs=d.get('log_P_Fs'),
            log_P_Bs=d.get('log_P_Bs'),
            log_flows=d.get('log_flows'),
            intermediate_rewards=d.get('intermediate_rewards'),
        )


@dataclass
class TrainingResult:
    """Results from training."""

    model: Union[TBModel, DBModel]
    losses: List[float]
    logZs: List[float]
    sampled_states: List[List]
    objective: str = "TB"
    hit_rates: Optional[List[float]] = None
    target_coverages: Optional[List[float]] = None
    n_targets: int = 0
    hit_trajectories: Optional[List[HitTrajectory]] = None

    @property
    def final_Z(self):
        return np.exp(self.logZs[-1])

    @property
    def final_hit_rate(self):
        return self.hit_rates[-1] if self.hit_rates else None

    @property
    def final_target_coverage(self):
        return self.target_coverages[-1] if self.target_coverages else None

    @property
    def n_unique_targets_hit(self):
        if self.target_coverages and self.n_targets > 0:
            return int(self.target_coverages[-1] * self.n_targets)
        return None

    @property
    def total_hits(self):
        return len(self.hit_trajectories) if self.hit_trajectories else 0

    def get_hit_stats(self):
        if not self.hit_trajectories:
            return {'total_hits': 0, 'unique_targets': 0, 'hits_per_target': {}}

        hits_per_target = {}
        for hit in self.hit_trajectories:
            seq_str = ''.join(c for c in hit.sequence if c != 'ε')
            hits_per_target[seq_str] = hits_per_target.get(seq_str, 0) + 1

        return {
            'total_hits': len(self.hit_trajectories),
            'unique_targets': len(hits_per_target),
            'hits_per_target': hits_per_target,
            'first_hit_iteration': min(h.iteration for h in self.hit_trajectories),
            'last_hit_iteration': max(h.iteration for h in self.hit_trajectories),
        }

    def save(self, path, save_model=True):
        """Save training result to disk."""
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        if save_model and self.model is not None:
            model_path = base_path.with_suffix('.pt')
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

        if self.hit_trajectories:
            hits_path = base_path.with_name(base_path.name + '_hits').with_suffix('.json')
            hits_data = {
                'metadata': {
                    'objective': self.objective,
                    'n_targets': self.n_targets,
                    'total_hits': self.total_hits,
                    'saved_at': datetime.now().isoformat(),
                },
                'stats': self.get_hit_stats(),
                'hit_trajectories': [h.to_dict() for h in self.hit_trajectories],
            }
            with open(hits_path, 'w') as f:
                json.dump(hits_data, f, indent=2)
            print(f"Hit trajectories saved to: {hits_path}")

        metrics_path = base_path.with_name(base_path.name + '_metrics').with_suffix('.json')
        metrics_data = {
            'objective': self.objective,
            'n_targets': self.n_targets,
            'losses': self.losses,
            'logZs': self.logZs,
            'hit_rates': self.hit_rates,
            'target_coverages': self.target_coverages,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

        pickle_path = base_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Full result saved to: {pickle_path}")

    @classmethod
    def load(cls, path):
        """Load training result from disk."""
        pickle_path = Path(path)
        if not pickle_path.suffix:
            pickle_path = pickle_path.with_suffix('.pkl')
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
        print(f"Loaded from: {pickle_path}")
        return result

    @classmethod
    def load_hits_only(cls, path):
        """Load only hit trajectories from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return [HitTrajectory.from_dict(h) for h in data['hit_trajectories']]


def sample_trajectory_tb(model, reward_fn, config, replay_buffer=None):
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


def train_tb(reward_fn, config, replay_buffer=None, verbose=True):
    """Train using Trajectory Balance objective."""
    set_seed(config.seed)
    model = TBModel(config.n_hid_units, config.uniform_backward)
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


def sample_trajectory_db(model, reward_fn, config, use_fldb=False):
    """Sample trajectory for DB/FL-DB objective."""
    state = get_initial_state()
    log_flows, log_P_Fs, log_P_Bs, trajectory_rewards = [], [], [], []

    for t in range(config.n_action_steps):
        state_tensor = state_to_tensor(state)
        P_F, P_B, log_F = model(state_tensor)
        log_flows.append(log_F)

        mask = calculate_forward_mask(state[1])
        P_F_masked = torch.where(mask, P_F, torch.tensor(-100.0))
        categorical = Categorical(logits=P_F_masked)
        action_idx = categorical.sample()
        log_P_Fs.append(categorical.log_prob(action_idx))

        new_state = perform_action(state, action_idx.item())

        if config.uniform_backward:
            mask = calculate_backward_mask(new_state[0], new_state[1])
            valid_actions = mask.sum()
            log_P_Bs.append(-torch.log(valid_actions.float()))
        else:
            _, P_B_new, _ = model(state_to_tensor(new_state))
            mask = calculate_backward_mask(new_state[0], new_state[1])
            P_B_masked = torch.where(mask, P_B_new, torch.tensor(-100.0))
            log_P_Bs.append(Categorical(logits=P_B_masked).log_prob(action_idx))

        # FL-DB: use R(s') after executing action
        if use_fldb:
            intermediate_reward = reward_fn(new_state[1])
            log_intermediate_reward = torch.log(torch.tensor(intermediate_reward).float()).clamp(min=-20.0)
            trajectory_rewards.append(log_intermediate_reward)

        state = new_state

    _, _, log_F_terminal = model(state_to_tensor(state))
    log_flows.append(log_F_terminal)
    terminal_reward = reward_fn(state[1])

    return state, log_flows, log_P_Fs, log_P_Bs, trajectory_rewards, terminal_reward


def train_db(reward_fn, config, use_fldb=False, verbose=True):
    """Train using DB or FL-DB objective."""
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

        episode_loss = torch.tensor(0.0)
        log_terminal_reward = torch.log(torch.tensor(terminal_reward).float()).clamp(min=-20.0)

        for t in range(config.n_action_steps):
            log_F_s = log_flows[t]
            log_P_F = log_P_Fs[t]
            log_P_B = log_P_Bs[t]

            if t == config.n_action_steps - 1:
                log_F_s_next = log_terminal_reward
            else:
                log_F_s_next = log_flows[t + 1]

            if use_fldb:
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


def train(reward_fn, config=None, replay_buffer=None, verbose=True):
    """Train a GFlowNet model using the specified objective."""
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


def get_policy_probs(model, state, insert_only=False):
    """Get forward policy probabilities for a given state."""
    with torch.no_grad():
        output = model(state_to_tensor(state))
        P_F = output[0]
        mask = calculate_forward_mask(state[1], insert_only=insert_only)
        P_F = torch.where(mask, P_F, torch.tensor(-100.0))
        probs = Categorical(logits=P_F).probs
    return probs


def generate_greedy_trajectory(model, insert_only=False):
    """Generate the most likely trajectory by greedy action selection."""
    state = get_initial_state()
    trajectory = [state]

    for _ in range(MAX_LEN):
        probs = get_policy_probs(model, state, insert_only=insert_only)
        best_action = torch.argmax(probs).item()
        state = perform_action(state, best_action)
        trajectory.append(state)

    return trajectory


# Legacy alias
sample_trajectory = sample_trajectory_tb
