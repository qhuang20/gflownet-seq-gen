"""GFlowNet loss functions: TB, DB, and FL-DB."""

import torch


def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward):
    """TB loss: (log Z + sum log P_F - log R - sum log P_B)^2"""
    log_reward = torch.log(reward).clamp(min=-20.0)
    return (logZ + log_P_F - log_reward - log_P_B).pow(2)


def detailed_balance_loss(log_F_s, log_P_F, log_P_B, log_F_s_next):
    """DB loss: (log F(s) + log P_F - log F(s') - log P_B)^2"""
    return (log_F_s + log_P_F - log_F_s_next - log_P_B).pow(2)


def forward_looking_db_loss(log_F_s, log_P_F, log_P_B, log_F_s_next, log_trajectory_reward):
    """FL-DB loss: extends DB with intermediate reward term."""
    return (log_F_s + log_P_F - log_F_s_next - log_P_B - log_trajectory_reward).pow(2)


def compute_db_trajectory_loss(
    log_flows, log_P_Fs, log_P_Bs, log_terminal_reward,
    use_fldb=False, log_trajectory_rewards=None,
):
    """Compute total DB or FL-DB loss summed over all transitions in a trajectory."""
    T = len(log_P_Fs)
    total_loss = torch.tensor(0.0)

    for t in range(T):
        log_F_s = log_flows[t]
        log_P_F = log_P_Fs[t]
        log_P_B = log_P_Bs[t]

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
