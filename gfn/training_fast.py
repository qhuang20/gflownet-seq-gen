"""GPU-accelerated GFlowNet training with batch trajectory sampling."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from .env import get_env_config, N_TIMESTEPS, MAX_LEN, VOCAB_SIZE, MAX_ACTIONS
from .model import TBModel, DBModel
from .losses import trajectory_balance_loss
from .training import TrainingResult, HitTrajectory


@dataclass
class FastTrainingConfig:
    """Configuration for fast batch training.

    Environment configuration is integrated - just set alphabet and max_seq_len.
    """

    # Environment
    alphabet: List[str] = field(default_factory=lambda: ['A', 'U', 'G', 'C'])
    max_seq_len: int = 10

    # Training
    seed: int = 42
    hidden_layers: Union[int, List[int]] = 32
    learning_rate: float = 3e-3
    auto_scale_lr: bool = True
    batch_size: int = 256
    n_iterations: int = 5000  # gradient updates, not episodes

    # Policy
    uniform_backward: bool = True
    explore_ratio: float = 0.1
    temperature: float = 1.0

    # Action space
    insert_only: bool = False

    # Device and objective
    device: str = "cuda"
    objective: str = "TB"  # "TB", "DB", or "FLDB"
    n_reward_workers: int = 4

    # Target sequences for hit rate tracking
    target_sequences: Optional[List[List[str]]] = None

    # Derived
    n_action_steps: int = field(init=False)
    n_timesteps: int = field(init=False)
    effective_lr: float = field(init=False)
    _target_set: set = field(init=False, repr=False)

    def __post_init__(self):
        from .env import EnvConfig, set_env_config
        env_config = EnvConfig(alphabet=self.alphabet, max_seq_len=self.max_seq_len)
        set_env_config(env_config)

        self.n_timesteps = env_config.n_timesteps
        self.n_action_steps = env_config.n_timesteps - 1

        if self.objective not in ("TB", "DB", "FLDB"):
            raise ValueError(f"Unknown objective: {self.objective}")

        if not torch.cuda.is_available() and self.device == "cuda":
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

        if self.auto_scale_lr:
            scale_factor = self.batch_size / 16
            self.effective_lr = self.learning_rate * scale_factor
        else:
            self.effective_lr = self.learning_rate

        if self.target_sequences is not None:
            self._target_set = set(tuple(seq) for seq in self.target_sequences)
        else:
            self._target_set = set()

    @property
    def n_episodes(self):
        return self.batch_size * self.n_iterations

    @property
    def search_space_size(self):
        vocab_size = len(self.alphabet)
        if self.insert_only:
            return vocab_size ** self.max_seq_len
        return sum(vocab_size ** k for k in range(self.max_seq_len + 1))


def get_char_mappings(device):
    """Get char<->index mappings for the current config."""
    config = get_env_config()
    char_to_idx = {char: idx for idx, char in enumerate(config.alphabet)}
    char_to_idx['ε'] = len(config.alphabet)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    eps_idx = char_to_idx['ε']
    return char_to_idx, idx_to_char, eps_idx


def init_batch_states(batch_size, device):
    """Initialize batch of empty states at timestep 0."""
    config = get_env_config()
    _, _, eps_idx = get_char_mappings(device)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    sequences = torch.full(
        (batch_size, config.max_seq_len), eps_idx,
        dtype=torch.long, device=device
    )
    return timesteps, sequences


def states_to_tensor_batch(timesteps, sequences, device):
    """Convert batch of states to one-hot tensor for neural network input."""
    config = get_env_config()
    batch_size = timesteps.shape[0]

    time_onehot = torch.zeros(batch_size, config.n_timesteps, device=device)
    time_onehot.scatter_(1, timesteps.unsqueeze(1), 1.0)

    n_chars = config.vocab_size + 1
    seq_onehot = torch.zeros(batch_size, config.max_seq_len, n_chars, device=device)
    seq_onehot.scatter_(2, sequences.unsqueeze(2), 1.0)
    seq_flat = seq_onehot.view(batch_size, -1)

    return torch.cat([time_onehot, seq_flat], dim=1)


def get_sequence_lengths(sequences, eps_idx):
    """Count non-epsilon characters per sequence."""
    return (sequences != eps_idx).sum(dim=1)


def calculate_forward_masks_batch(sequences, device, insert_only=False):
    """Calculate forward action masks for a batch of sequences."""
    config = get_env_config()
    batch_size = sequences.shape[0]
    max_len = config.max_seq_len
    vocab_size = config.vocab_size
    max_actions = config.max_actions

    _, _, eps_idx = get_char_mappings(device)
    seq_lens = get_sequence_lengths(sequences, eps_idx)
    mask = torch.zeros(batch_size, max_actions, dtype=torch.bool, device=device)

    # Insertions
    positions = torch.arange(max_len, device=device)
    can_insert_at_pos = (positions.unsqueeze(0) <= seq_lens.unsqueeze(1))
    can_insert = (seq_lens < max_len).unsqueeze(1)
    can_insert_at_pos = can_insert_at_pos & can_insert
    can_insert_expanded = can_insert_at_pos.unsqueeze(2).expand(-1, -1, vocab_size)
    mask[:, :max_len * vocab_size] = can_insert_expanded.reshape(batch_size, max_len * vocab_size)

    if insert_only:
        return mask

    # Deletions
    deletion_offset = max_len * vocab_size
    mask[:, deletion_offset:deletion_offset + max_len] = (sequences != eps_idx)

    # Mutations
    mutation_offset = deletion_offset + max_len
    can_mutate = (sequences != eps_idx).unsqueeze(2).expand(-1, -1, vocab_size)
    mask[:, mutation_offset:mutation_offset + max_len * vocab_size] = \
        can_mutate.reshape(batch_size, max_len * vocab_size)

    return mask


def calculate_backward_masks_batch(timesteps, sequences, device, insert_only=False):
    """Calculate backward action masks for a batch of sequences."""
    config = get_env_config()
    batch_size = timesteps.shape[0]
    max_len = config.max_seq_len
    vocab_size = config.vocab_size
    max_actions = config.max_actions

    _, _, eps_idx = get_char_mappings(device)
    seq_lens = get_sequence_lengths(sequences, eps_idx)
    mask = torch.zeros(batch_size, max_actions, dtype=torch.bool, device=device)

    at_root = (timesteps == 0)

    insertion_offset = 0
    deletion_offset = max_len * vocab_size
    mutation_offset = deletion_offset + max_len

    if insert_only:
        can_delete = (sequences != eps_idx) & (~at_root).unsqueeze(1)
        mask[:, deletion_offset:deletion_offset + max_len] = can_delete
        return mask

    max_prev_len = timesteps - 1
    positions = torch.arange(max_len, device=device)

    case1 = (seq_lens > max_prev_len) & ~at_root
    case2 = (seq_lens == max_prev_len) & ~at_root
    case3 = (seq_lens < max_prev_len) & ~at_root

    # Deletions
    can_delete = (sequences != eps_idx)
    mask[:, deletion_offset:deletion_offset + max_len] = \
        can_delete & (case1 | case2 | case3).unsqueeze(1)

    # Mutations
    can_mutate = (sequences != eps_idx).unsqueeze(2).expand(-1, -1, vocab_size)
    mutation_valid = (case2 | case3).unsqueeze(1).unsqueeze(2)
    mask[:, mutation_offset:mutation_offset + max_len * vocab_size] = \
        (can_mutate & mutation_valid).reshape(batch_size, max_len * vocab_size)

    # Insertions
    can_insert_at_pos = (positions.unsqueeze(0) <= seq_lens.unsqueeze(1))
    insertion_valid = case3.unsqueeze(1)
    can_insert = (can_insert_at_pos & insertion_valid).unsqueeze(2).expand(-1, -1, vocab_size)
    mask[:, insertion_offset:insertion_offset + max_len * vocab_size] = \
        can_insert.reshape(batch_size, max_len * vocab_size)

    return mask


def count_valid_backward_actions(timesteps, sequences, device, insert_only=False):
    """Count valid backward actions for uniform backward probability."""
    mask = calculate_backward_masks_batch(timesteps, sequences, device, insert_only=insert_only)
    return mask.sum(dim=1).float()


def perform_actions_batch(timesteps, sequences, action_indices, device):
    """Execute actions on a batch of states."""
    config = get_env_config()
    batch_size = timesteps.shape[0]
    max_len = config.max_seq_len
    vocab_size = config.vocab_size

    _, _, eps_idx = get_char_mappings(device)

    insertion_end = max_len * vocab_size
    deletion_end = insertion_end + max_len

    is_insertion = action_indices < insertion_end
    is_deletion = (action_indices >= insertion_end) & (action_indices < deletion_end)
    is_mutation = action_indices >= deletion_end

    new_sequences = sequences.clone()

    if is_insertion.any():
        insertion_actions = action_indices[is_insertion]
        insertion_pos = insertion_actions // vocab_size
        insertion_char = insertion_actions % vocab_size
        seqs_to_insert = new_sequences[is_insertion]

        for i, (seq, pos, char) in enumerate(zip(seqs_to_insert, insertion_pos, insertion_char)):
            if pos < max_len - 1:
                seq[pos+1:] = seq[pos:-1].clone()
            seq[pos] = char

        new_sequences[is_insertion] = seqs_to_insert

    if is_deletion.any():
        deletion_actions = action_indices[is_deletion] - insertion_end
        seqs_to_delete = new_sequences[is_deletion]

        for i, (seq, pos) in enumerate(zip(seqs_to_delete, deletion_actions)):
            if pos < max_len - 1:
                seq[pos:-1] = seq[pos+1:].clone()
            seq[-1] = eps_idx

        new_sequences[is_deletion] = seqs_to_delete

    if is_mutation.any():
        mutation_actions = action_indices[is_mutation] - deletion_end
        mutation_pos = mutation_actions // vocab_size
        mutation_char = mutation_actions % vocab_size
        batch_indices = torch.where(is_mutation)[0]
        new_sequences[batch_indices, mutation_pos] = mutation_char

    return timesteps + 1, new_sequences


def sample_trajectories_batch_tb(model, reward_fn, config):
    """Sample batch of trajectories using TB objective."""
    device = torch.device(config.device)
    batch_size = config.batch_size
    _, idx_to_char, eps_idx = get_char_mappings(device)

    timesteps, sequences = init_batch_states(batch_size, device)
    total_log_P_F = torch.zeros(batch_size, device=device)
    total_log_P_B = torch.zeros(batch_size, device=device)

    for t in range(config.n_action_steps):
        state_tensor = states_to_tensor_batch(timesteps, sequences, device)
        P_F_logits, _ = model(state_tensor)
        forward_mask = calculate_forward_masks_batch(sequences, device, insert_only=config.insert_only)

        P_F_masked = torch.where(forward_mask, P_F_logits, torch.tensor(-100.0, device=device))
        P_F_tempered = P_F_masked / config.temperature
        policy_probs = F.softmax(P_F_tempered, dim=1)

        n_valid = forward_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform_probs = forward_mask.float() / n_valid
        action_probs = (1 - config.explore_ratio) * policy_probs + config.explore_ratio * uniform_probs

        # Renormalize to prevent numerical issues
        action_probs = action_probs * forward_mask.float()
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)

        action_indices = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

        # Use original policy for loss computation
        categorical = Categorical(logits=P_F_masked)
        total_log_P_F += categorical.log_prob(action_indices)

        new_timesteps, new_sequences = perform_actions_batch(
            timesteps, sequences, action_indices, device
        )

        if config.uniform_backward:
            n_valid_backward = count_valid_backward_actions(
                new_timesteps, new_sequences, device, insert_only=config.insert_only
            )
            total_log_P_B += -torch.log(n_valid_backward.clamp(min=1.0))

        timesteps, sequences = new_timesteps, new_sequences

    # Compute terminal rewards
    final_states_list = []
    rewards_list = []
    for i in range(batch_size):
        seq_indices = sequences[i].cpu().tolist()
        seq_chars = [idx_to_char[idx] for idx in seq_indices]
        final_states_list.append([timesteps[i].item(), seq_chars])
        rewards_list.append(reward_fn(seq_chars))

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    return total_log_P_F, total_log_P_B, rewards, sequences, final_states_list


def compute_batch_rewards(sequences, reward_fn, idx_to_char, eps_idx,
                          n_workers=1, is_intermediate=False):
    """Compute rewards for a batch of sequences."""
    device = sequences.device
    batch_size = sequences.shape[0]

    # Use GPU batch computation if available
    if hasattr(reward_fn, 'supports_batch') and reward_fn.supports_batch:
        if is_intermediate and hasattr(reward_fn, 'batch_reward_progressive'):
            return reward_fn.batch_reward_progressive(sequences)
        elif not is_intermediate and hasattr(reward_fn, 'batch_reward_terminal'):
            return reward_fn.batch_reward_terminal(sequences)
        else:
            return reward_fn.batch_reward(sequences)

    # CPU fallback
    def seq_to_chars(seq_tensor):
        return [idx_to_char[idx.item()] for idx in seq_tensor]

    seq_lists = [seq_to_chars(sequences[i].cpu()) for i in range(batch_size)]

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            rewards_list = list(executor.map(reward_fn, seq_lists))
    else:
        rewards_list = [reward_fn(seq) for seq in seq_lists]

    return torch.tensor(rewards_list, dtype=torch.float32, device=device)


def sample_trajectories_batch_db(model, reward_fn, config, use_fldb=False):
    """Sample batch of trajectories for DB/FL-DB objective."""
    device = torch.device(config.device)
    batch_size = config.batch_size
    n_steps = config.n_action_steps
    _, idx_to_char, eps_idx = get_char_mappings(device)

    timesteps, sequences = init_batch_states(batch_size, device)

    log_flows = torch.zeros(batch_size, n_steps + 1, device=device)
    log_P_Fs = torch.zeros(batch_size, n_steps, device=device)
    log_P_Bs = torch.zeros(batch_size, n_steps, device=device)
    action_indices_all = torch.zeros(batch_size, n_steps, dtype=torch.long, device=device)

    intermediate_rewards = torch.zeros(batch_size, n_steps, device=device) if use_fldb else None

    for t in range(n_steps):
        state_tensor = states_to_tensor_batch(timesteps, sequences, device)
        P_F_logits, P_B_logits, log_F = model(state_tensor)
        log_flows[:, t] = log_F

        forward_mask = calculate_forward_masks_batch(sequences, device, insert_only=config.insert_only)
        P_F_masked = torch.where(forward_mask, P_F_logits, torch.tensor(-100.0, device=device))
        P_F_tempered = P_F_masked / config.temperature
        policy_probs = F.softmax(P_F_tempered, dim=1)

        n_valid = forward_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform_probs = forward_mask.float() / n_valid
        action_probs = (1 - config.explore_ratio) * policy_probs + config.explore_ratio * uniform_probs

        action_probs = action_probs * forward_mask.float()
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)

        action_indices = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        action_indices_all[:, t] = action_indices

        categorical = Categorical(logits=P_F_masked)
        log_P_Fs[:, t] = categorical.log_prob(action_indices)

        new_timesteps, new_sequences = perform_actions_batch(
            timesteps, sequences, action_indices, device
        )

        # FL-DB: compute R(s') after executing action
        if use_fldb:
            intermediate_rewards[:, t] = compute_batch_rewards(
                new_sequences, reward_fn, idx_to_char, eps_idx,
                n_workers=config.n_reward_workers, is_intermediate=True
            )

        if config.uniform_backward:
            n_valid_backward = count_valid_backward_actions(
                new_timesteps, new_sequences, device, insert_only=config.insert_only
            )
            log_P_Bs[:, t] = -torch.log(n_valid_backward.clamp(min=1.0))
        else:
            new_state_tensor = states_to_tensor_batch(new_timesteps, new_sequences, device)
            _, P_B_new, _ = model(new_state_tensor)
            backward_mask = calculate_backward_masks_batch(
                new_timesteps, new_sequences, device, insert_only=config.insert_only
            )
            P_B_masked = torch.where(backward_mask, P_B_new, torch.tensor(-100.0, device=device))
            log_P_Bs[:, t] = Categorical(logits=P_B_masked).log_prob(action_indices)

        timesteps, sequences = new_timesteps, new_sequences

    # Terminal state
    final_state_tensor = states_to_tensor_batch(timesteps, sequences, device)
    _, _, log_F_terminal = model(final_state_tensor)
    log_flows[:, -1] = log_F_terminal

    terminal_rewards = compute_batch_rewards(
        sequences, reward_fn, idx_to_char, eps_idx,
        n_workers=config.n_reward_workers, is_intermediate=False
    )

    final_states_list = []
    for i in range(batch_size):
        seq_indices = sequences[i].cpu().tolist()
        seq_chars = [idx_to_char[idx] for idx in seq_indices]
        final_states_list.append([timesteps[i].item(), seq_chars])

    return (log_flows, log_P_Fs, log_P_Bs, terminal_rewards,
            intermediate_rewards, action_indices_all, sequences, final_states_list)


def compute_db_loss_batch(log_flows, log_P_Fs, log_P_Bs, log_terminal_rewards,
                          use_fldb=False, log_intermediate_rewards=None):
    """Compute batched DB or FL-DB loss over all transitions."""
    log_F_current = log_flows[:, :-1]
    log_F_next = log_flows[:, 1:].clone()
    log_F_next[:, -1] = log_terminal_rewards

    diff = log_F_current + log_P_Fs - log_F_next - log_P_Bs

    if use_fldb and log_intermediate_rewards is not None:
        diff = diff - log_intermediate_rewards

    return diff.pow(2).sum(dim=1)


def train_fast(reward_fn, config=None, verbose=True):
    """Train a GFlowNet model using batch trajectory sampling."""
    if config is None:
        config = FastTrainingConfig()

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    if config.objective == "TB":
        return _train_fast_tb(reward_fn, config, device, verbose)
    elif config.objective in ("DB", "FLDB"):
        return _train_fast_db(reward_fn, config, device, verbose, use_fldb=(config.objective == "FLDB"))
    else:
        raise ValueError(f"Unknown objective: {config.objective}")


def _compute_hit_rate(final_states, target_set):
    if not target_set:
        return 0.0
    hits = sum(1 for state in final_states
               if tuple(s for s in state[1] if s != 'ε') in target_set)
    return hits / len(final_states) if final_states else 0.0


def _update_target_coverage(final_states, target_set, hit_targets):
    if not target_set:
        return 0.0
    for state in final_states:
        seq = tuple(s for s in state[1] if s != 'ε')
        if seq in target_set:
            hit_targets.add(seq)
    return len(hit_targets) / len(target_set)


def _collect_hit_trajectories(
    final_states, rewards, target_set, iteration, hit_count_tracker,
    log_P_Fs=None, log_P_Bs=None, log_flows=None,
    action_indices=None, intermediate_rewards=None,
):
    """Collect all hit trajectories from a batch."""
    if not target_set:
        return []

    hits = []
    rewards_list = rewards.cpu().tolist() if isinstance(rewards, torch.Tensor) else rewards

    for batch_idx, state in enumerate(final_states):
        seq = tuple(s for s in state[1] if s != 'ε')
        if seq in target_set:
            hit_count_tracker[seq] = hit_count_tracker.get(seq, 0) + 1

            hits.append(HitTrajectory(
                sequence=list(seq),
                iteration=iteration,
                reward=rewards_list[batch_idx] if batch_idx < len(rewards_list) else 1.0,
                hit_count=hit_count_tracker[seq],
                batch_index=batch_idx,
                actions=action_indices[batch_idx].cpu().tolist() if action_indices is not None else None,
                log_P_Fs=log_P_Fs[batch_idx].cpu().tolist() if log_P_Fs is not None else None,
                log_P_Bs=log_P_Bs[batch_idx].cpu().tolist() if log_P_Bs is not None else None,
                log_flows=log_flows[batch_idx].cpu().tolist() if log_flows is not None else None,
                intermediate_rewards=intermediate_rewards[batch_idx].cpu().tolist() if intermediate_rewards is not None else None,
            ))

    return hits


def _train_fast_tb(reward_fn, config, device, verbose):
    model = TBModel(config.hidden_layers, config.uniform_backward).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.effective_lr)

    losses, logZs, sampled_states = [], [], []
    hit_rates = [] if config.target_sequences else None
    target_coverages = [] if config.target_sequences else None
    hit_targets = set()
    hit_trajectories = [] if config.target_sequences else None
    hit_count_tracker = {}

    iterator = range(config.n_iterations)
    if verbose:
        iterator = tqdm(iterator, ncols=70, desc="Fast TB Training")

    for iteration in iterator:
        log_P_F, log_P_B, rewards, _, final_states = sample_trajectories_batch_tb(
            model, reward_fn, config
        )

        log_rewards = torch.log(rewards).clamp(min=-20.0)
        tb_losses = (model.logZ + log_P_F - log_rewards - log_P_B).pow(2)
        loss = tb_losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logZs.append(model.logZ.item())

        if hit_rates is not None:
            hit_rates.append(_compute_hit_rate(final_states, config._target_set))
            target_coverages.append(_update_target_coverage(final_states, config._target_set, hit_targets))
            hit_trajectories.extend(_collect_hit_trajectories(
                final_states, rewards, config._target_set, iteration, hit_count_tracker
            ))

        if final_states:
            sampled_states.extend(final_states[:min(10, len(final_states))])

    return TrainingResult(
        model=model.cpu(), losses=losses, logZs=logZs,
        sampled_states=sampled_states, objective="TB",
        hit_rates=hit_rates, target_coverages=target_coverages,
        n_targets=len(config._target_set) if config.target_sequences else 0,
        hit_trajectories=hit_trajectories,
    )


def _train_fast_db(reward_fn, config, device, verbose, use_fldb=False):
    objective_name = "FL-DB" if use_fldb else "DB"
    model = DBModel(config.hidden_layers, config.uniform_backward).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.effective_lr)

    losses, logZs, sampled_states = [], [], []
    hit_rates = [] if config.target_sequences else None
    target_coverages = [] if config.target_sequences else None
    hit_targets = set()
    hit_trajectories = [] if config.target_sequences else None
    hit_count_tracker = {}

    _, idx_to_char, eps_idx = get_char_mappings(device)

    iterator = range(config.n_iterations)
    if verbose:
        iterator = tqdm(iterator, ncols=70, desc=f"Fast {objective_name} Training")

    for iteration in iterator:
        (log_flows, log_P_Fs, log_P_Bs, terminal_rewards,
         intermediate_rewards, action_indices_batch, _, final_states) = \
            sample_trajectories_batch_db(model, reward_fn, config, use_fldb=use_fldb)

        log_terminal_rewards = torch.log(terminal_rewards).clamp(min=-20.0)
        log_intermediate_rewards = None
        if use_fldb and intermediate_rewards is not None:
            log_intermediate_rewards = torch.log(intermediate_rewards).clamp(min=-20.0)

        db_losses = compute_db_loss_batch(
            log_flows, log_P_Fs, log_P_Bs, log_terminal_rewards,
            use_fldb=use_fldb, log_intermediate_rewards=log_intermediate_rewards
        )
        loss = db_losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logZs.append(log_flows[0, 0].item())

        if hit_rates is not None:
            hit_rates.append(_compute_hit_rate(final_states, config._target_set))
            target_coverages.append(_update_target_coverage(final_states, config._target_set, hit_targets))
            hit_trajectories.extend(_collect_hit_trajectories(
                final_states, terminal_rewards, config._target_set, iteration, hit_count_tracker,
                log_P_Fs=log_P_Fs.detach(), log_P_Bs=log_P_Bs.detach(),
                log_flows=log_flows.detach(), action_indices=action_indices_batch,
                intermediate_rewards=intermediate_rewards.detach() if intermediate_rewards is not None else None,
            ))

        if final_states:
            sampled_states.extend(final_states[:min(10, len(final_states))])

    return TrainingResult(
        model=model.cpu(), losses=losses, logZs=logZs,
        sampled_states=sampled_states, objective=objective_name,
        hit_rates=hit_rates, target_coverages=target_coverages,
        n_targets=len(config._target_set) if config.target_sequences else 0,
        hit_trajectories=hit_trajectories,
    )
