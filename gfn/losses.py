"""
GFlowNet Loss Functions

This module contains various loss functions for GFlowNet training:
- Trajectory Balance (TB)
- Detailed Balance (DB)
- Forward-Looking Detailed Balance (FL-DB)
"""

import torch


def trajectory_balance_loss(
    logZ: torch.Tensor,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    reward: torch.Tensor
) -> torch.Tensor:
    """
    Trajectory Balance (TB) loss.
    
    The TB objective is:
        Z * P_F(τ) = R(x) * P_B(τ)
    
    Loss: (log Z + Σ log P_F - log R - Σ log P_B)²
    
    Args:
        logZ: Log partition function (learnable scalar)
        log_P_F: Sum of log forward probabilities along trajectory
        log_P_B: Sum of log backward probabilities along trajectory
        reward: Terminal reward R(x)
    
    Returns:
        Scalar loss value
    """
    log_reward = torch.log(reward).clamp(min=-20.0)
    loss = (logZ + log_P_F - log_reward - log_P_B).pow(2)
    return loss


def detailed_balance_loss(
    log_F_s: torch.Tensor,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    log_F_s_next: torch.Tensor,
) -> torch.Tensor:
    """
    Detailed Balance (DB) loss for a single transition.
    
    The DB objective for each transition (s, a, s') is:
        F(s) * P_F(a|s) = F(s') * P_B(a|s')
    
    For terminal transitions, F(s') = R(x).
    
    Loss: (log F(s) + log P_F - log F(s') - log P_B)²
    
    Args:
        log_F_s: Log flow at current state
        log_P_F: Log forward probability of action
        log_P_B: Log backward probability of action
        log_F_s_next: Log flow at next state (or log R for terminal)
    
    Returns:
        Loss for this transition
    """
    loss = (log_F_s + log_P_F - log_F_s_next - log_P_B).pow(2)
    return loss


def forward_looking_db_loss(
    log_F_s: torch.Tensor,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    log_F_s_next: torch.Tensor,
    log_trajectory_reward: torch.Tensor,
) -> torch.Tensor:
    """
    Forward-Looking Detailed Balance (FL-DB) loss for a single transition.
    
    FL-DB extends DB by incorporating intermediate rewards:
        F(s) * P_F(a|s) = F(s') * P_B(a|s') * R(s,a,s')
    
    Loss: (log F(s) + log P_F - log F(s') - log P_B - log R(s,a,s'))²
    
    Args:
        log_F_s: Log flow at current state
        log_P_F: Log forward probability of action
        log_P_B: Log backward probability of action
        log_F_s_next: Log flow at next state
        log_trajectory_reward: Log intermediate reward for this transition
    
    Returns:
        Loss for this transition
    """
    loss = (log_F_s + log_P_F - log_F_s_next - log_P_B - log_trajectory_reward).pow(2)
    return loss


def compute_db_trajectory_loss(
    log_flows: torch.Tensor,
    log_P_Fs: torch.Tensor,
    log_P_Bs: torch.Tensor,
    log_terminal_reward: torch.Tensor,
    use_fldb: bool = False,
    log_trajectory_rewards: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute total DB or FL-DB loss for a full trajectory.
    
    Args:
        log_flows: Log flows for each state in trajectory, shape (T+1,)
        log_P_Fs: Log forward probs for each action, shape (T,)
        log_P_Bs: Log backward probs for each action, shape (T,)
        log_terminal_reward: Log of terminal reward
        use_fldb: Whether to use FL-DB (with trajectory rewards)
        log_trajectory_rewards: Log rewards for each transition, shape (T,)
    
    Returns:
        Total loss summed over all transitions
    """
    T = len(log_P_Fs)
    total_loss = torch.tensor(0.0)
    
    for t in range(T):
        log_F_s = log_flows[t]
        log_P_F = log_P_Fs[t]
        log_P_B = log_P_Bs[t]
        
        # For last transition, use terminal reward instead of F(s')
        if t == T - 1:
            log_F_s_next = log_terminal_reward
        else:
            log_F_s_next = log_flows[t + 1]
        
        if use_fldb and log_trajectory_rewards is not None:
            loss = forward_looking_db_loss(
                log_F_s, log_P_F, log_P_B, log_F_s_next, 
                log_trajectory_rewards[t]
            )
        else:
            loss = detailed_balance_loss(
                log_F_s, log_P_F, log_P_B, log_F_s_next
            )
        
        total_loss = total_loss + loss
    
    return total_loss
