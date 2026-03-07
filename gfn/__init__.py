"""
GFlowNet for Sequence Generation

Supports TB, DB, and FL-DB training objectives.
"""

from .env import (
    EnvConfig,
    PRESETS,
    set_env_config,
    use_preset,
    get_env_config,
    print_env_info,
    ALPHABET,
    N_TIMESTEPS,
    VOCAB_SIZE,
    MAX_LEN,
    MAX_ACTIONS,
    ACTIONS_LIST,
    get_next_states,
    perform_action,
    infer_action_id,
    calculate_forward_mask,
    calculate_backward_mask,
    generate_all_states,
    get_initial_state,
    state_to_string,
)

from .utils import (
    set_seed,
    state_to_tensor,
    get_input_size,
    load_fasta,
    load_fasta_sequences,
    analyze_sequences,
    truncate_sequences,
    sequences_to_targets,
)

from .reward import (
    RewardFunction,
    TargetMatchReward,
    CountReward,
    AlignmentReward,
    HammingReward,
    EntropyWeightedHammingReward,
    AdaptiveHammingReward,
    ProgressiveHammingReward,
    ConservationWeightedHammingReward,
    create_target_reward,
    DEFAULT_TARGETS,
)

from .model import (
    TBModel,
    DBModel,
    trajectory_balance_loss,
)

from .losses import (
    trajectory_balance_loss,
    detailed_balance_loss,
    forward_looking_db_loss,
    compute_db_trajectory_loss,
)

from .training import (
    TrainingConfig,
    TrainingResult,
    HitTrajectory,
    train,
    sample_trajectory,
    get_policy_probs,
    generate_greedy_trajectory,
)

from .training_fast import (
    FastTrainingConfig,
    train_fast,
)

from .visualization import (
    plot_training_curves,
    plot_reward_distribution,
    plot_state_space,
    compute_edge_flows,
    compute_max_flow_trajectories,
    plot_flow_network,
    print_policy,
)

__version__ = "0.2.0"
