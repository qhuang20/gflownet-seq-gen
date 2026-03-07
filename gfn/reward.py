"""Reward functions for sequence generation."""

import math
from collections import Counter
from typing import List, Union, Optional, Dict
import torch


class RewardFunction:
    """Base class for reward functions."""

    def __init__(self, r_min: float = 0.1):
        self.r_min = r_min

    def __call__(self, seq: List[str]) -> float:
        raise NotImplementedError


class TargetMatchReward(RewardFunction):
    """Returns 1.0 for exact target matches, r_min otherwise."""

    def __init__(self, target_sequences, r_min=0.1):
        super().__init__(r_min)
        self.target_sequences = target_sequences

    def __call__(self, seq):
        if seq in self.target_sequences:
            return 1.0
        return self.r_min


class CountReward(RewardFunction):
    """Reward based on counting a specific character."""

    def __init__(self, target_char='A', r_min=0.1):
        super().__init__(r_min)
        self.target_char = target_char

    def __call__(self, seq):
        count = seq.count(self.target_char)
        return float(count) if count > 0 else self.r_min


class AlignmentReward(RewardFunction):
    """Needleman-Wunsch alignment-based partial credit reward."""

    def __init__(self, target_sequences, match_score=1.0, mismatch_score=-1.0,
                 gap_score=-1.0, r_min=0.1):
        super().__init__(r_min)
        self.target_sequences = target_sequences
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score

    def _alignment_score(self, candidate, target):
        """Normalized Needleman-Wunsch alignment score."""
        cell_scores = {(-1, -1): 0}
        for i in range(len(candidate)):
            cell_scores[(i, -1)] = (i + 1) * self.gap_score
        for j in range(len(target)):
            cell_scores[(-1, j)] = (j + 1) * self.gap_score

        for i in range(len(candidate)):
            for j in range(len(target)):
                match = self.match_score if candidate[i] == target[j] else self.mismatch_score
                cell_scores[(i, j)] = max(
                    cell_scores[(i - 1, j - 1)] + match,
                    cell_scores[(i - 1, j)] + self.gap_score,
                    cell_scores[(i, j - 1)] + self.gap_score
                )

        max_possible = self.match_score * len(target)
        if max_possible == 0:
            return 0.0
        return cell_scores[(len(candidate) - 1, len(target) - 1)] / max_possible

    def __call__(self, seq):
        if isinstance(seq, list):
            seq = ''.join(s for s in seq if s != 'ε')
        if not seq:
            return self.r_min
        scores = [self._alignment_score(seq, target) for target in self.target_sequences]
        return max(max(scores), self.r_min)


class HammingReward(RewardFunction):
    """GPU-accelerated Hamming similarity reward."""

    def __init__(self, target_sequences, alphabet=None, r_min=0.1, device='cuda'):
        super().__init__(r_min)
        self.target_sequences = target_sequences
        self.device = device if torch.cuda.is_available() else 'cpu'

        if alphabet is None:
            all_chars = set()
            for seq in target_sequences:
                all_chars.update(seq)
            all_chars.discard('ε')
            alphabet = sorted(list(all_chars))

        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self.char_to_idx['ε'] = len(alphabet)
        self.eps_idx = len(alphabet)
        self._precompute_targets()

    def _precompute_targets(self):
        """Convert targets to tensor format for fast comparison."""
        max_len = max(len(seq) for seq in self.target_sequences)
        target_indices = []
        for seq in self.target_sequences:
            indices = [self.char_to_idx.get(c, self.eps_idx) for c in seq]
            indices += [self.eps_idx] * (max_len - len(indices))
            target_indices.append(indices)

        self.target_tensor = torch.tensor(
            target_indices, dtype=torch.long, device=self.device
        )
        self.target_lengths = torch.tensor(
            [sum(1 for c in seq if c != 'ε') for seq in self.target_sequences],
            dtype=torch.float32, device=self.device
        )

    def __call__(self, seq):
        seq_clean = [c for c in seq if c != 'ε']
        if not seq_clean:
            return self.r_min

        max_sim = 0.0
        for target in self.target_sequences:
            target_clean = [c for c in target if c != 'ε']
            min_len = min(len(seq_clean), len(target_clean))
            if min_len == 0:
                continue
            matches = sum(1 for a, b in zip(seq_clean[:min_len], target_clean[:min_len]) if a == b)
            sim = matches / len(target_clean) if target_clean else 0
            max_sim = max(max_sim, sim)

        return max(max_sim, self.r_min)

    def batch_reward(self, sequences):
        """Compute rewards for a batch of sequences on GPU. Input: [batch, seq_len] tensor."""
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        target_len = self.target_tensor.shape[1]

        if sequences.device != self.target_tensor.device:
            sequences = sequences.to(self.target_tensor.device)

        min_len = min(seq_len, target_len)
        seq_expanded = sequences[:, :min_len].unsqueeze(1)
        target_expanded = self.target_tensor[:, :min_len].unsqueeze(0)

        seq_not_eps = seq_expanded != self.eps_idx
        target_not_eps = target_expanded != self.eps_idx
        valid_positions = seq_not_eps & target_not_eps

        matches = (seq_expanded == target_expanded) & valid_positions
        match_counts = matches.float().sum(dim=2)

        target_lengths = self.target_lengths.unsqueeze(0)
        similarities = match_counts / target_lengths.clamp(min=1)
        max_similarities = similarities.max(dim=1).values

        return max_similarities.clamp(min=self.r_min)

    @property
    def supports_batch(self):
        return True


def _compute_sequence_entropy(seq, alphabet_size=4):
    """Normalized Shannon entropy of a sequence."""
    seq_clean = [c for c in seq if c != 'ε']
    if not seq_clean:
        return 0.0

    counts = Counter(seq_clean)
    total = len(seq_clean)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    max_entropy = math.log(min(alphabet_size, len(seq_clean)))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


class EntropyWeightedHammingReward(HammingReward):
    """Hamming reward weighted by target sequence entropy.

    Complex (high-entropy) targets get bonus reward to prevent mode collapse
    on easy repetitive patterns.
    """

    def __init__(self, target_sequences, alphabet=None, r_min=0.1, device='cuda',
                 entropy_weight=1.0):
        super().__init__(target_sequences, alphabet, r_min, device)
        self.entropy_weight = entropy_weight
        self._compute_entropy_weights()

    def _compute_entropy_weights(self):
        alphabet_size = len(self.alphabet)
        entropies = [_compute_sequence_entropy(seq, alphabet_size)
                     for seq in self.target_sequences]

        self.entropy_multipliers = torch.tensor(
            [1.0 + self.entropy_weight * e for e in entropies],
            dtype=torch.float32, device=self.device
        )
        self._target_entropies = entropies

        print(f"EntropyWeightedHammingReward:")
        print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
        print(f"  Weight range: [{self.entropy_multipliers.min():.3f}, {self.entropy_multipliers.max():.3f}]")

    def __call__(self, seq):
        seq_clean = [c for c in seq if c != 'ε']
        if not seq_clean:
            return self.r_min

        max_weighted_sim = 0.0
        for i, target in enumerate(self.target_sequences):
            target_clean = [c for c in target if c != 'ε']
            min_len = min(len(seq_clean), len(target_clean))
            if min_len == 0:
                continue
            matches = sum(1 for a, b in zip(seq_clean[:min_len], target_clean[:min_len]) if a == b)
            sim = matches / len(target_clean) if target_clean else 0
            weighted_sim = sim * self.entropy_multipliers[i].item()
            max_weighted_sim = max(max_weighted_sim, weighted_sim)

        return max(max_weighted_sim, self.r_min)

    def batch_reward(self, sequences):
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        target_len = self.target_tensor.shape[1]

        if sequences.device != self.target_tensor.device:
            sequences = sequences.to(self.target_tensor.device)

        min_len = min(seq_len, target_len)
        seq_expanded = sequences[:, :min_len].unsqueeze(1)
        target_expanded = self.target_tensor[:, :min_len].unsqueeze(0)

        seq_not_eps = seq_expanded != self.eps_idx
        target_not_eps = target_expanded != self.eps_idx
        valid_positions = seq_not_eps & target_not_eps

        matches = (seq_expanded == target_expanded) & valid_positions
        match_counts = matches.float().sum(dim=2)

        target_lengths = self.target_lengths.unsqueeze(0)
        similarities = match_counts / target_lengths.clamp(min=1)
        weighted_similarities = similarities * self.entropy_multipliers.unsqueeze(0)
        max_weighted = weighted_similarities.max(dim=1).values

        return max_weighted.clamp(min=self.r_min)


class AdaptiveHammingReward(HammingReward):
    """Hamming reward with dynamic decay for frequently hit targets.

    decay_factor = 1 / (1 + decay_rate * log(1 + hit_count))
    """

    def __init__(self, target_sequences, alphabet=None, r_min=0.1, device='cuda',
                 decay_rate=0.5, min_multiplier=0.1):
        super().__init__(target_sequences, alphabet, r_min, device)
        self.decay_rate = decay_rate
        self.min_multiplier = min_multiplier
        self.hit_counts: Dict[tuple, int] = {}
        self._target_to_idx = {tuple(seq): i for i, seq in enumerate(self.target_sequences)}
        self.decay_multipliers = torch.ones(
            len(self.target_sequences), dtype=torch.float32, device=self.device
        )

    def _compute_decay(self, hit_count):
        if hit_count == 0:
            return 1.0
        decay = 1.0 / (1.0 + self.decay_rate * math.log(1 + hit_count))
        return max(decay, self.min_multiplier)

    def _update_decay_multipliers(self):
        for seq_tuple, idx in self._target_to_idx.items():
            hit_count = self.hit_counts.get(seq_tuple, 0)
            self.decay_multipliers[idx] = self._compute_decay(hit_count)

    def register_hit(self, seq):
        """Register a target hit to update decay."""
        seq_tuple = tuple(seq)
        if seq_tuple in self._target_to_idx:
            self.hit_counts[seq_tuple] = self.hit_counts.get(seq_tuple, 0) + 1

    def __call__(self, seq):
        seq_clean = [c for c in seq if c != 'ε']
        if not seq_clean:
            return self.r_min

        max_decayed_sim = 0.0
        for i, target in enumerate(self.target_sequences):
            target_clean = [c for c in target if c != 'ε']
            min_len = min(len(seq_clean), len(target_clean))
            if min_len == 0:
                continue
            matches = sum(1 for a, b in zip(seq_clean[:min_len], target_clean[:min_len]) if a == b)
            sim = matches / len(target_clean) if target_clean else 0
            decay = self.decay_multipliers[i].item()
            max_decayed_sim = max(max_decayed_sim, sim * decay)

        seq_tuple = tuple(seq)
        if seq_tuple in self._target_to_idx:
            self.hit_counts[seq_tuple] = self.hit_counts.get(seq_tuple, 0) + 1
            self._update_decay_multipliers()

        return max(max_decayed_sim, self.r_min)

    def batch_reward(self, sequences):
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        target_len = self.target_tensor.shape[1]

        if sequences.device != self.target_tensor.device:
            sequences = sequences.to(self.target_tensor.device)

        min_len = min(seq_len, target_len)
        seq_expanded = sequences[:, :min_len].unsqueeze(1)
        target_expanded = self.target_tensor[:, :min_len].unsqueeze(0)

        seq_not_eps = seq_expanded != self.eps_idx
        target_not_eps = target_expanded != self.eps_idx
        valid_positions = seq_not_eps & target_not_eps

        matches = (seq_expanded == target_expanded) & valid_positions
        match_counts = matches.float().sum(dim=2)

        target_lengths = self.target_lengths.unsqueeze(0)
        similarities = match_counts / target_lengths.clamp(min=1)
        decayed_similarities = similarities * self.decay_multipliers.unsqueeze(0)
        max_decayed, best_targets = decayed_similarities.max(dim=1)

        # Register exact hits
        is_exact_hit = (similarities.gather(1, best_targets.unsqueeze(1)).squeeze(1) >= 0.999)
        if is_exact_hit.any():
            hit_target_indices = best_targets[is_exact_hit].cpu().tolist()
            for target_idx in hit_target_indices:
                target_seq = tuple(self.target_sequences[target_idx])
                self.hit_counts[target_seq] = self.hit_counts.get(target_seq, 0) + 1
            self._update_decay_multipliers()

        return max_decayed.clamp(min=self.r_min)

    def get_hit_stats(self):
        return {
            'total_hits': sum(self.hit_counts.values()),
            'unique_targets_hit': len(self.hit_counts),
            'hits_per_target': {
                ''.join(c for c in k if c != 'ε'): v
                for k, v in sorted(self.hit_counts.items(), key=lambda x: -x[1])
            },
            'decay_multipliers': {
                ''.join(c for c in self.target_sequences[i] if c != 'ε'): self.decay_multipliers[i].item()
                for i in range(len(self.target_sequences))
                if self.decay_multipliers[i].item() < 1.0
            }
        }

    def reset_hits(self):
        self.hit_counts.clear()
        self.decay_multipliers.fill_(1.0)


class ProgressiveHammingReward(HammingReward):
    """Hamming reward with current-length normalization for better FL-DB gradients.

    Standard: matches / target_len  (weak signal for partial sequences)
    Progressive: matches / current_len  (strong signal even at early steps)
    Terminal rewards still use target_len normalization.
    """

    def __init__(self, target_sequences, alphabet=None, r_min=0.1, device='cuda',
                 prefix_boost=0.0, prefix_decay=0.1):
        super().__init__(target_sequences, alphabet, r_min, device)
        self.prefix_boost = prefix_boost
        self.prefix_decay = prefix_decay

        max_len = max(len(seq) for seq in target_sequences)
        self._compute_position_weights(max_len)

    def _compute_position_weights(self, max_len):
        positions = torch.arange(max_len, dtype=torch.float32, device=self.device)
        self.position_weights = 1.0 + self.prefix_boost * torch.exp(-self.prefix_decay * positions)

    def __call__(self, seq, use_progressive=True):
        seq_clean = [c for c in seq if c != 'ε']
        if not seq_clean:
            return self.r_min

        current_len = len(seq_clean)
        max_sim = 0.0

        for target in self.target_sequences:
            target_clean = [c for c in target if c != 'ε']
            min_len = min(current_len, len(target_clean))
            if min_len == 0:
                continue

            if self.prefix_boost > 0:
                weighted_matches = sum(
                    self.position_weights[i].item()
                    for i, (a, b) in enumerate(zip(seq_clean[:min_len], target_clean[:min_len]))
                    if a == b
                )
                total_weight = self.position_weights[:min_len].sum().item()
                matches = weighted_matches / total_weight * min_len
            else:
                matches = sum(1 for a, b in zip(seq_clean[:min_len], target_clean[:min_len]) if a == b)

            if use_progressive:
                sim = matches / current_len
            else:
                sim = matches / len(target_clean)

            max_sim = max(max_sim, sim)

        return max(max_sim, self.r_min)

    def batch_reward(self, sequences, use_progressive=True):
        """Compute rewards for a batch. use_progressive=True for intermediate, False for terminal."""
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        target_len = self.target_tensor.shape[1]

        if sequences.device != self.target_tensor.device:
            sequences = sequences.to(self.target_tensor.device)

        min_len = min(seq_len, target_len)
        seq_expanded = sequences[:, :min_len].unsqueeze(1)
        target_expanded = self.target_tensor[:, :min_len].unsqueeze(0)

        seq_not_eps = seq_expanded != self.eps_idx
        target_not_eps = target_expanded != self.eps_idx
        valid_positions = seq_not_eps & target_not_eps
        matches = (seq_expanded == target_expanded) & valid_positions

        if self.prefix_boost > 0:
            pos_weights = self.position_weights[:min_len].view(1, 1, -1)
            weighted_matches = (matches.float() * pos_weights).sum(dim=2)
            total_weights = (valid_positions.float() * pos_weights).sum(dim=2).clamp(min=1)
            valid_counts = valid_positions.float().sum(dim=2).clamp(min=1)
            match_counts = weighted_matches / total_weights * valid_counts
        else:
            match_counts = matches.float().sum(dim=2)

        seq_lengths = (sequences != self.eps_idx).float().sum(dim=1)

        if use_progressive:
            similarities = match_counts / seq_lengths.unsqueeze(1).clamp(min=1)
        else:
            target_lengths = self.target_lengths.unsqueeze(0)
            similarities = match_counts / target_lengths.clamp(min=1)

        max_similarities = similarities.max(dim=1).values
        return max_similarities.clamp(min=self.r_min)

    def batch_reward_progressive(self, sequences):
        return self.batch_reward(sequences, use_progressive=True)

    def batch_reward_terminal(self, sequences):
        return self.batch_reward(sequences, use_progressive=False)


class ConservationWeightedHammingReward(ProgressiveHammingReward):
    """Hamming reward weighted by evolutionary conservation (species count).

    weight = alpha + (1 - alpha) * log(species_count) / log(max_count)
    """

    def __init__(self, target_sequences, species_counts, alphabet=None, r_min=0.1,
                 device='cuda', alpha=0.3, use_log_scale=True,
                 prefix_boost=0.0, prefix_decay=0.1):
        super().__init__(target_sequences, alphabet, r_min, device,
                         prefix_boost, prefix_decay)
        self.alpha = alpha
        self.use_log_scale = use_log_scale
        self.species_counts = species_counts
        self._compute_conservation_weights()

    def _compute_conservation_weights(self):
        counts = []
        for seq in self.target_sequences:
            seq_str = ''.join(c for c in seq if c != 'ε')
            counts.append(self.species_counts.get(seq_str, 1))

        max_count = max(counts)

        if self.use_log_scale:
            log_counts = [math.log1p(c) for c in counts]
            max_log = math.log1p(max_count)
            conservation_scores = [lc / max_log for lc in log_counts]
        else:
            conservation_scores = [c / max_count for c in counts]

        self.conservation_weights = torch.tensor(
            [self.alpha + (1 - self.alpha) * score for score in conservation_scores],
            dtype=torch.float32, device=self.device
        )
        self._counts = counts
        self._conservation_scores = conservation_scores

        print(f"ConservationWeightedHammingReward:")
        print(f"  Alpha: {self.alpha}, log scaling: {self.use_log_scale}")
        print(f"  Species count range: [{min(counts)}, {max(counts)}]")
        print(f"  Weight range: [{self.conservation_weights.min():.3f}, {self.conservation_weights.max():.3f}]")

    def __call__(self, seq, use_progressive=True):
        seq_clean = [c for c in seq if c != 'ε']
        if not seq_clean:
            return self.r_min

        current_len = len(seq_clean)
        max_weighted_sim = 0.0

        for i, target in enumerate(self.target_sequences):
            target_clean = [c for c in target if c != 'ε']
            min_len = min(current_len, len(target_clean))
            if min_len == 0:
                continue

            matches = sum(1 for a, b in zip(seq_clean[:min_len], target_clean[:min_len]) if a == b)
            if use_progressive:
                sim = matches / current_len
            else:
                sim = matches / len(target_clean)

            weighted_sim = sim * self.conservation_weights[i].item()
            max_weighted_sim = max(max_weighted_sim, weighted_sim)

        return max(max_weighted_sim, self.r_min)

    def batch_reward(self, sequences, use_progressive=True):
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        target_len = self.target_tensor.shape[1]

        if sequences.device != self.target_tensor.device:
            sequences = sequences.to(self.target_tensor.device)

        min_len = min(seq_len, target_len)
        seq_expanded = sequences[:, :min_len].unsqueeze(1)
        target_expanded = self.target_tensor[:, :min_len].unsqueeze(0)

        seq_not_eps = seq_expanded != self.eps_idx
        target_not_eps = target_expanded != self.eps_idx
        valid_positions = seq_not_eps & target_not_eps

        matches = (seq_expanded == target_expanded) & valid_positions
        match_counts = matches.float().sum(dim=2)

        seq_lengths = (sequences != self.eps_idx).float().sum(dim=1)

        if use_progressive:
            similarities = match_counts / seq_lengths.unsqueeze(1).clamp(min=1)
        else:
            target_lengths = self.target_lengths.unsqueeze(0)
            similarities = match_counts / target_lengths.clamp(min=1)

        weighted_similarities = similarities * self.conservation_weights.unsqueeze(0)
        max_weighted = weighted_similarities.max(dim=1).values

        return max_weighted.clamp(min=self.r_min)

    def get_conservation_stats(self):
        seq_weights = []
        for i, seq in enumerate(self.target_sequences):
            seq_str = ''.join(c for c in seq if c != 'ε')
            seq_weights.append({
                'sequence': seq_str,
                'species_count': self._counts[i],
                'conservation_score': self._conservation_scores[i],
                'weight': self.conservation_weights[i].item()
            })

        return {
            'alpha': self.alpha,
            'use_log_scale': self.use_log_scale,
            'weight_range': (self.conservation_weights.min().item(),
                           self.conservation_weights.max().item()),
            'sequences': sorted(seq_weights, key=lambda x: -x['weight'])
        }


def create_target_reward(targets, r_min=0.1):
    """Create a simple target matching reward function."""
    return TargetMatchReward(targets, r_min)


# Default target sequences for the toy example
DEFAULT_TARGETS = [
    ['A', 'B', 'B', 'C'],
    ['A', 'B', 'C', 'ε'],
    ['C', 'A', 'C', 'C'],
    ['C', 'B', 'A', 'ε'],
    ['C', 'C', 'B', 'A'],
    ['C', 'C', 'C', 'A']
]
