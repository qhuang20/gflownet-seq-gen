"""
GFlowNet for Sequence Generation

A modular implementation of GFlowNet (Generative Flow Network) for learning
to generate sequences with rewards proportional to their quality.

Supports three training objectives:
- TB (Trajectory Balance): Global log Z parameter
- DB (Detailed Balance): Per-state flow F(s)
- FL-DB (Forward-Looking DB): DB with intermediate rewards

Example usage:
    from gfn import train, TrainingConfig
    from gfn.reward import TargetMatchReward, DEFAULT_TARGETS
    from gfn.visualization import plot_training_curves, plot_flow_network
    
    # Create reward function
    reward_fn = TargetMatchReward(DEFAULT_TARGETS, r_min=0.1)
    
    # Train with TB (default)
    config = TrainingConfig(n_episodes=20_000, objective="TB")
    result = train(reward_fn, config)
    
    # Or train with DB
    config_db = TrainingConfig(n_episodes=20_000, objective="DB")
    result_db = train(reward_fn, config_db)
    
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
    DBModel,
    trajectory_balance_loss,
)

# Loss functions
from .losses import (
    trajectory_balance_loss,
    detailed_balance_loss,
    forward_looking_db_loss,
    compute_db_trajectory_loss,
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
    "DBModel",
    # Losses
    "trajectory_balance_loss",
    "detailed_balance_loss",
    "forward_looking_db_loss",
    "compute_db_trajectory_loss",
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
