"""
GFlowNet Reward Functions

This module contains reward functions for sequence generation:
- Simple target matching reward
- Alignment-based partial reward
"""

from typing import List, Union


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, r_min: float = 0.1):
        """
        Args:
            r_min: Minimum reward for non-target sequences
        """
        self.r_min = r_min
    
    def __call__(self, seq: List[str]) -> float:
        raise NotImplementedError


class TargetMatchReward(RewardFunction):
    """
    Reward function that returns 1.0 for exact target matches, r_min otherwise.
    """
    
    def __init__(self, target_sequences: List[List[str]], r_min: float = 0.1):
        """
        Args:
            target_sequences: List of target sequences, each is a list of characters
                              e.g., [['A', 'B', 'C', 'ε'], ['C', 'B', 'A', 'ε']]
            r_min: Minimum reward for non-target sequences
        """
        super().__init__(r_min)
        self.target_sequences = target_sequences
    
    def __call__(self, seq: List[str]) -> float:
        """
        Calculate reward for a sequence.
        
        Args:
            seq: Sequence as list of characters (including 'ε' padding)
        
        Returns:
            1.0 if sequence matches a target, r_min otherwise
        """
        if seq in self.target_sequences:
            return 1.0
        return self.r_min


class CountReward(RewardFunction):
    """
    Reward function based on counting specific characters.
    """
    
    def __init__(self, target_char: str = 'A', r_min: float = 0.1):
        """
        Args:
            target_char: Character to count
            r_min: Minimum reward when count is 0
        """
        super().__init__(r_min)
        self.target_char = target_char
    
    def __call__(self, seq: List[str]) -> float:
        """
        Calculate reward based on count of target character.
        
        Args:
            seq: Sequence as list of characters
        
        Returns:
            Number of target characters, or r_min if count is 0
        """
        count = seq.count(self.target_char)
        if count > 0:
            return float(count)
        return self.r_min


class AlignmentReward(RewardFunction):
    """
    Reward function based on sequence alignment scores.
    Provides partial credit for sequences similar to targets.
    """
    
    def __init__(
        self,
        target_sequences: List[str],
        match_score: float = 1.0,
        mismatch_score: float = -1.0,
        gap_score: float = -1.0,
        r_min: float = 0.1
    ):
        """
        Args:
            target_sequences: List of target sequences as strings
            match_score: Score for matching characters
            mismatch_score: Penalty for mismatching characters
            gap_score: Penalty for gaps (insertions/deletions)
            r_min: Minimum reward
        """
        super().__init__(r_min)
        self.target_sequences = target_sequences
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
    
    def _alignment_score(self, candidate: str, target: str) -> float:
        """
        Calculate normalized alignment score between two sequences using
        Needleman-Wunsch algorithm.
        """
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
    
    def __call__(self, seq: Union[List[str], str]) -> float:
        """
        Calculate alignment-based reward.
        
        Args:
            seq: Sequence as list of characters or string
        
        Returns:
            Maximum alignment score across all targets
        """
        if isinstance(seq, list):
            seq = ''.join(s for s in seq if s != 'ε')
        
        if not seq:
            return self.r_min
            
        scores = [self._alignment_score(seq, target) for target in self.target_sequences]
        return max(max(scores), self.r_min)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_target_reward(
    targets: List[List[str]],
    r_min: float = 0.1
) -> TargetMatchReward:
    """
    Create a simple target matching reward function.
    
    Args:
        targets: List of target sequences
        r_min: Minimum reward
    
    Returns:
        TargetMatchReward instance
    """
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
