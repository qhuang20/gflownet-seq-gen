"""Flow network visualization and plotting functions."""

from typing import List, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
import torch

from .env import (
    N_TIMESTEPS, MAX_LEN, ACTIONS_LIST,
    generate_all_states, get_next_states,
    perform_action, calculate_forward_mask,
    get_initial_state, state_to_string,
)
from .utils import state_to_tensor
from .model import TBModel
from .training import TrainingResult


def plot_training_curves(result, figsize=(12, 8)):
    """Plot training loss and partition function estimates."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    axes[0].plot(result.losses, color="black")
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    Z_values = np.exp(result.logZs)
    axes[1].plot(Z_values, color="black")
    axes[1].set_ylabel('Estimated Z')
    axes[1].set_xlabel('Update Step')
    axes[1].set_title(f'Partition Function (final Z = {result.final_Z:.2f})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reward_distribution(result, reward_fn):
    """Print reward distribution of sampled states."""
    rewards = [reward_fn(state[1]) for state in result.sampled_states]
    unique_rewards, counts = np.unique(rewards, return_counts=True)
    percentages = counts / len(rewards) * 100

    print(f"\nReward Distribution ({len(result.sampled_states)} samples):")
    print("-" * 40)
    for reward, pct in zip(unique_rewards, percentages):
        print(f"  Reward {reward:.2f}: {pct:.1f}%")


def plot_state_space(trajectory=None, target_sequences=None, figsize=(18, 10)):
    """Plot the state space evolution over timesteps."""
    all_states = generate_all_states()

    target_set = set()
    if target_sequences:
        for seq in target_sequences:
            target_set.add(''.join(s for s in seq if s != 'ε'))

    fig = plt.figure(figsize=figsize)
    state_coords = {}

    for t, states in enumerate(all_states):
        y = N_TIMESTEPS - 1 - t
        x_positions = np.linspace(0, 1, len(states))
        is_terminal = (t == len(all_states) - 1)

        if is_terminal and target_set:
            target_x, target_states = [], []
            other_x = []
            for x, state in zip(x_positions, states):
                if state in target_set:
                    target_x.append(x)
                    target_states.append(state)
                else:
                    other_x.append(x)

            if other_x:
                plt.scatter(other_x, [y] * len(other_x), alpha=0.3, c='gray', s=20)
            if target_x:
                plt.scatter(target_x, [y] * len(target_x), alpha=0.9, c='red',
                           s=100, edgecolors='darkred', linewidths=2, zorder=10)
                for x, state in zip(target_x, target_states):
                    plt.annotate(state, (x, y), xytext=(0, -15),
                                textcoords='offset points',
                                rotation=90, ha='center', va='top',
                                fontsize=9, fontweight='bold', color='red')
                    state_coords[(t, state)] = (x, y)

            for x, state in zip(x_positions, states):
                label = state if state else 'ε'
                state_coords[(t, label)] = (x, y)
        else:
            plt.scatter(x_positions, [y] * len(states), alpha=0.5)
            for x, state in zip(x_positions, states):
                label = state if state else 'ε'
                plt.annotate(label, (x, y), xytext=(0, -15),
                            textcoords='offset points',
                            rotation=90, ha='center', va='top',
                            fontsize=10)
                state_coords[(t, label)] = (x, y)

    if trajectory:
        traj_coords = []
        for t, state in enumerate(trajectory):
            state_str = state_to_string(state) or 'ε'
            if (t, state_str) in state_coords:
                traj_coords.append(state_coords[(t, state_str)])

        if traj_coords:
            traj_coords = np.array(traj_coords)
            plt.plot(traj_coords[:, 0], traj_coords[:, 1],
                    'r-', linewidth=2, alpha=0.7)

    plt.yticks(range(N_TIMESTEPS - 1, -1, -1),
               [f't={t}' for t in range(N_TIMESTEPS)], fontsize=13)
    plt.xlabel('States', fontsize=14)
    plt.ylabel('Timestep', fontsize=14)
    plt.title('State Space Evolution', fontsize=15)
    plt.grid(True, axis='y')
    plt.margins(y=0.15)

    return fig


def compute_edge_flows(model, prob_threshold=0.0001):
    """Compute flow through each edge in the state graph."""
    edges = []
    root_state = get_initial_state()
    states_by_time = {0: [root_state]}

    for t in range(MAX_LEN):
        next_states = {}

        for curr_state in states_by_time[t]:
            with torch.no_grad():
                output = model(state_to_tensor(curr_state))
                P_F = output[0]
                mask = calculate_forward_mask(curr_state[1])
                P_F = torch.where(mask, P_F, torch.tensor(-100.0))
                probs = Categorical(logits=P_F).probs

            next_state_probs = {}
            for action_idx, (action, prob) in enumerate(zip(ACTIONS_LIST, probs)):
                if prob > prob_threshold:
                    next_state = perform_action(curr_state, action_idx)
                    next_state_str = str(next_state)

                    unpadded_curr = state_to_string(curr_state)
                    unpadded_next = state_to_string(next_state)

                    transition_key = (
                        tuple([curr_state[0], unpadded_curr]),
                        tuple([next_state[0], unpadded_next])
                    )

                    if transition_key not in next_state_probs:
                        next_state_probs[transition_key] = {
                            'prob': float(prob), 'state': next_state
                        }
                    else:
                        next_state_probs[transition_key]['prob'] += float(prob)

                    if next_state_str not in next_states:
                        next_states[next_state_str] = next_state

            for transition_key, info in next_state_probs.items():
                edges.append([
                    list(transition_key[0]),
                    list(transition_key[1]),
                    info['prob']
                ])

        states_by_time[t + 1] = list(next_states.values())

    return edges


def compute_max_flow_trajectories(model, target_sequences, initial_flow=100.0):
    """Compute greedy traceback trajectories from each target to root."""
    edges = compute_edge_flows(model)

    target_unpadded = [
        ''.join(s for s in seq if s != 'ε')
        for seq in target_sequences
    ]

    # Calculate node flows
    node_flows = {(0, ''): initial_flow}
    for source, target, prob in edges:
        t1, state1 = source
        key1 = (t1, state1 if state1 else '')
        if key1 in node_flows:
            current_flow = node_flows[key1]
            edge_flow = current_flow * prob
            t2, state2 = target
            key2 = (t2, state2 if state2 else '')
            if key2 not in node_flows:
                node_flows[key2] = 0
            node_flows[key2] += edge_flow

    trajectories = []
    for target_seq in target_unpadded:
        trajectory = [(N_TIMESTEPS - 1, target_seq)]
        current_time = N_TIMESTEPS - 1
        current_state = target_seq

        while current_time > 0:
            max_flow = 0
            best_prev_state = None
            for source, target, prob in edges:
                t1, state1 = source
                t2, state2 = target
                if t2 == current_time and state2 == current_state:
                    source_flow = node_flows.get((t1, state1 if state1 else ''), 0)
                    edge_flow = source_flow * prob
                    if edge_flow > max_flow:
                        max_flow = edge_flow
                        best_prev_state = state1
            if best_prev_state is not None:
                current_time -= 1
                current_state = best_prev_state
                trajectory.append((current_time, current_state))
            else:
                break
        trajectories.append(trajectory[::-1])

    return trajectories


def plot_flow_network(
    model, target_sequences=None, initial_flow=100.0,
    edge_flow_threshold=0.5, show_edge_labels=False,
    show_target_trajectory=True, show_nontarget_terminal_labels=True,
    figsize=(18, 10),
):
    """Visualize the flow network with greedy traceback trajectories in red."""
    edges = compute_edge_flows(model)
    all_states = generate_all_states()

    target_unpadded = []
    if target_sequences:
        target_unpadded = [
            ''.join(s for s in seq if s != 'ε')
            for seq in target_sequences
        ]

    # Calculate node flows
    node_flows = {(0, ''): initial_flow}
    for source, target, prob in edges:
        t1, state1 = source
        key1 = (t1, state1 if state1 else '')
        if key1 in node_flows:
            current_flow = node_flows[key1]
            edge_flow = current_flow * prob
            t2, state2 = target
            key2 = (t2, state2 if state2 else '')
            if key2 not in node_flows:
                node_flows[key2] = 0
            node_flows[key2] += edge_flow

    # Compute greedy traceback trajectories
    trajectories = []
    if show_target_trajectory and target_unpadded:
        for target_seq in target_unpadded:
            trajectory = [(N_TIMESTEPS - 1, target_seq)]
            current_time = N_TIMESTEPS - 1
            current_state = target_seq

            while current_time > 0:
                max_flow = 0
                best_prev_state = None
                for source, target, prob in edges:
                    t1, state1 = source
                    t2, state2 = target
                    if t2 == current_time and state2 == current_state:
                        source_flow = node_flows.get((t1, state1 if state1 else ''), 0)
                        edge_flow = source_flow * prob
                        if edge_flow > max_flow:
                            max_flow = edge_flow
                            best_prev_state = state1
                if best_prev_state is not None:
                    current_time -= 1
                    current_state = best_prev_state
                    trajectory.append((current_time, current_state))
                else:
                    break
            trajectories.append(trajectory[::-1])

    fig = plt.figure(figsize=figsize)

    # Plot nodes
    for t, states in enumerate(all_states):
        y = N_TIMESTEPS - 1 - t
        x_positions = np.linspace(0, 1, len(states))
        is_terminal = (t == N_TIMESTEPS - 1)

        if is_terminal and target_unpadded:
            terminal_with_flow = set()
            if show_nontarget_terminal_labels:
                for source, target, prob in edges:
                    t1, state1 = source
                    t2, state2 = target
                    if t2 == N_TIMESTEPS - 1:
                        source_flow = node_flows.get((t1, state1 if state1 else ''), 0)
                        edge_flow = source_flow * prob
                        if edge_flow > edge_flow_threshold:
                            terminal_with_flow.add(state2)

            target_x, target_states, target_sizes = [], [], []
            flow_x, flow_states, flow_sizes = [], [], []
            noflow_x = []

            for x, state in zip(x_positions, states):
                flow = node_flows.get((t, state if state else ''), 0)
                size = 10 + np.sqrt(flow / initial_flow) * 290
                if state in target_unpadded:
                    target_x.append(x)
                    target_states.append(state)
                    target_sizes.append(size)
                elif state in terminal_with_flow:
                    flow_x.append(x)
                    flow_states.append(state)
                    flow_sizes.append(size)
                else:
                    noflow_x.append(x)

            if noflow_x:
                plt.scatter(noflow_x, [y] * len(noflow_x), alpha=0.3, c='gray', s=20)
            if flow_x:
                plt.scatter(flow_x, [y] * len(flow_x), alpha=0.7, c='black', s=flow_sizes)
                for x, state in zip(flow_x, flow_states):
                    plt.annotate(state, (x, y), xytext=(0, -15),
                                textcoords='offset points',
                                rotation=90, ha='center', va='top',
                                fontsize=8, color='black')
            if target_x:
                plt.scatter(target_x, [y] * len(target_x), alpha=0.9, c='red',
                           s=100, edgecolors='darkred', linewidths=2, zorder=10)
                for x, state in zip(target_x, target_states):
                    plt.annotate(state, (x, y), xytext=(0, -15),
                                textcoords='offset points',
                                rotation=90, ha='center', va='top',
                                fontsize=9, fontweight='bold', color='red')
        else:
            sizes = []
            colors = []
            for state in states:
                flow = node_flows.get((t, state if state else ''), 0)
                size = 10 + np.sqrt(flow / initial_flow) * 290
                sizes.append(size)
                intensity = np.cbrt(flow / initial_flow)
                intensity = min(1, max(0.1, intensity))
                colors.append((0, 0, 0, intensity))

            plt.scatter(x_positions, [y] * len(states), c=colors, s=sizes)
            for x, state in zip(x_positions, states):
                label = state if state else 'ε'
                plt.annotate(label, (x, y), xytext=(0, -15),
                            textcoords='offset points',
                            rotation=90, ha='center', va='top', color='black')

    # Plot edges
    for source, target, prob in edges:
        t1, state1 = source
        t2, state2 = target

        source_flow = node_flows.get((t1, state1 if state1 else ''), 0)
        edge_flow = source_flow * prob

        if edge_flow > edge_flow_threshold:
            y1 = N_TIMESTEPS - 1 - t1
            y2 = N_TIMESTEPS - 1 - t2

            try:
                x1_idx = all_states[t1].index(state1 if state1 else '')
                x2_idx = all_states[t2].index(state2 if state2 else '')

                x1 = np.linspace(0, 1, len(all_states[t1]))[x1_idx]
                x2 = np.linspace(0, 1, len(all_states[t2]))[x2_idx]

                intensity = np.cbrt(edge_flow / initial_flow)
                intensity = min(1, max(0.1, intensity))

                is_trajectory_edge = False
                if show_target_trajectory:
                    for trajectory in trajectories:
                        for i in range(len(trajectory) - 1):
                            if (t1, state1) == trajectory[i] and (t2, state2) == trajectory[i + 1]:
                                is_trajectory_edge = True
                                break
                        if is_trajectory_edge:
                            break

                edge_color = (1, 0, 0, intensity) if is_trajectory_edge else (0, 0, 0, intensity)
                width = 1 + 3 * np.log1p(100 * edge_flow / initial_flow)
                plt.plot([x1, x2], [y1, y2], color=edge_color, linewidth=width)

                if show_edge_labels and edge_flow > edge_flow_threshold:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    plt.annotate(f'{prob:.2f}', (mid_x, mid_y),
                                xytext=(-5, 5), textcoords='offset points',
                                fontsize=6, color='red')
                    plt.annotate(f'{edge_flow:.1f}', (mid_x, mid_y),
                                xytext=(-5, -5), textcoords='offset points',
                                fontsize=6, color='blue')
            except ValueError:
                pass

    title = f'Flow Network (edges with flow > {edge_flow_threshold})'
    if show_target_trajectory and trajectories:
        title += ' with Max Flow Trajectories'

    plt.yticks(range(N_TIMESTEPS - 1, -1, -1),
               [f't={t}' for t in range(N_TIMESTEPS)], fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.xlabel('States', fontsize=18)
    plt.ylabel('Timestep', fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(True, axis='y')
    plt.margins(y=0.15)

    if show_target_trajectory and trajectories:
        print("\nMax flow trajectories (greedy traceback):")
        for i, trajectory in enumerate(trajectories):
            print(f"\nTrajectory for target '{target_unpadded[i]}':")
            for t, state in trajectory:
                print(f"  t={t}: {state if state else 'ε'}")

    return fig


def print_policy(model, state):
    """Print action probabilities for a given state."""
    with torch.no_grad():
        output = model(state_to_tensor(state))
        P_F = output[0]
        mask = calculate_forward_mask(state[1])
        P_F = torch.where(mask, P_F, torch.tensor(-100.0))
        probs = Categorical(logits=P_F).probs

    print(f"\nPolicy for state {state}:")
    print("-" * 50)
    for action, prob in zip(ACTIONS_LIST, probs):
        if prob > 0.01:
            print(f"  {action}: {prob:.3f}")
