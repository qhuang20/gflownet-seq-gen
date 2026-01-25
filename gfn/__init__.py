"""
GFlowNet for Sequence Generation

A modular implementation of GFlowNet (Generative Flow Network) for learning
to generate sequences with rewards proportional to their quality.

Example usage:
    from gfn import train, TrainingConfig
    from gfn.reward import TargetMatchReward, DEFAULT_TARGETS
    from gfn.visualization import plot_training_curves, plot_flow_network
    
    # Create reward function
    reward_fn = TargetMatchReward(DEFAULT_TARGETS, r_min=0.1)
    
    # Train model
    config = TrainingConfig(n_episodes=20_000, learning_rate=3e-3)
    result = train(reward_fn, config)
    
    # Visualize
    plot_training_curves(result)
    plot_flow_network(result.model)
"""

# Environment
from .env import (
    # Constants
    ALPHABET,
    N_TIMESTEPS,
    VOCAB_SIZE,
    MAX_LEN,
    MAX_ACTIONS,
    ACTIONS_LIST,
    # Functions
    get_next_states,
    perform_action,
    infer_action_id,
    calculate_forward_mask,
    calculate_backward_mask,
    generate_all_states,
    get_initial_state,
    state_to_string,
)

# Utilities
from .utils import (
    set_seed,
    state_to_tensor,
    get_input_size,
)

# Reward functions
from .reward import (
    RewardFunction,
    TargetMatchReward,
    CountReward,
    AlignmentReward,
    create_target_reward,
    DEFAULT_TARGETS,
)

# Model
from .model import (
    TBModel,
    trajectory_balance_loss,
)

# Training
from .training import (
    TrainingConfig,
    TrainingResult,
    train,
    sample_trajectory,
    get_policy_probs,
    generate_greedy_trajectory,
)

# Visualization
from .visualization import (
    plot_training_curves,
    plot_reward_distribution,
    plot_state_space,
    compute_edge_flows,
    compute_max_flow_trajectories,
    plot_flow_network,
    print_policy,
)

__version__ = "0.1.0"
__all__ = [
    # Constants
    "ALPHABET",
    "N_TIMESTEPS", 
    "VOCAB_SIZE",
    "MAX_LEN",
    "MAX_ACTIONS",
    "ACTIONS_LIST",
    # Environment
    "get_next_states",
    "perform_action",
    "infer_action_id",
    "calculate_forward_mask",
    "calculate_backward_mask",
    "generate_all_states",
    "get_initial_state",
    "state_to_string",
    # Utilities
    "set_seed",
    "state_to_tensor",
    "get_input_size",
    # Reward
    "RewardFunction",
    "TargetMatchReward",
    "CountReward",
    "AlignmentReward",
    "create_target_reward",
    "DEFAULT_TARGETS",
    # Model
    "TBModel",
    "trajectory_balance_loss",
    # Training
    "TrainingConfig",
    "TrainingResult",
    "train",
    "sample_trajectory",
    "get_policy_probs",
    "generate_greedy_trajectory",
    # Visualization
    "plot_training_curves",
    "plot_reward_distribution",
    "plot_state_space",
    "compute_edge_flows",
    "compute_max_flow_trajectories",
    "plot_flow_network",
    "print_policy",
]
