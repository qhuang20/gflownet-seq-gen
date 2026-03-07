#!/usr/bin/env python3
"""
Train GFlowNet on LET-7 22bp miRNA sequences.

Usage:
    nohup python train_LET7_22bp.py > training.log 2>&1 &
    tail -f training.log
"""

import time
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from gfn import train_fast, FastTrainingConfig, TrainingResult
from gfn.reward import EntropyWeightedHammingReward, HammingReward

DATA_PATH = Path(__file__).parent / "data" / "LET7_22bp_targets.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    "alphabet": ['A', 'U', 'G', 'C'],
    "max_seq_len": 22,
    "hidden_layers": [128, 64, 32],
    "batch_size": 2048,
    "n_iterations": 3000,
    "learning_rate": 3e-3,
    "device": "cuda",
    "objective": "FLDB",
    "explore_ratio": 0.3,
    "temperature": 2.0,
    "insert_only": True,
    "seed": 42,
}

CHECKPOINT_EVERY = 200
LOG_EVERY = 50
REWARD_SCHEME = "entropy"
ENTROPY_WEIGHT = 1.0


def train_with_checkpoints(reward_fn, config, save_dir, checkpoint_every=100, log_every=10):
    """Train with periodic checkpoints and logging."""
    from gfn.training_fast import (
        DBModel,
        sample_trajectories_batch_db, compute_db_loss_batch, get_char_mappings,
        _compute_hit_rate, _update_target_coverage, _collect_hit_trajectories,
    )
    from tqdm import tqdm

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    model = DBModel(config.hidden_layers, config.uniform_backward).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.effective_lr)

    losses, logZs, sampled_states = [], [], []
    hit_rates = [] if config.target_sequences else None
    target_coverages = [] if config.target_sequences else None
    hit_targets = set()
    hit_trajectories = [] if config.target_sequences else None
    hit_count_tracker = {}

    _, idx_to_char, eps_idx = get_char_mappings(device)

    start_time = time.time()

    for iteration in tqdm(range(config.n_iterations), desc="Training", ncols=80):
        (log_flows, log_P_Fs, log_P_Bs, terminal_rewards,
         intermediate_rewards, action_indices_batch, _, final_states) = \
            sample_trajectories_batch_db(model, reward_fn, config, use_fldb=True)

        log_terminal_rewards = torch.log(terminal_rewards).clamp(min=-20.0)
        log_intermediate_rewards = torch.log(intermediate_rewards).clamp(min=-20.0) \
            if intermediate_rewards is not None else None

        db_losses = compute_db_loss_batch(
            log_flows, log_P_Fs, log_P_Bs, log_terminal_rewards,
            use_fldb=True, log_intermediate_rewards=log_intermediate_rewards
        )

        loss = db_losses.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logZs.append(log_flows[0, 0].item())

        if hit_rates is not None:
            hit_rate = _compute_hit_rate(final_states, config._target_set)
            hit_rates.append(hit_rate)
            coverage = _update_target_coverage(final_states, config._target_set, hit_targets)
            target_coverages.append(coverage)

            batch_hits = _collect_hit_trajectories(
                final_states, terminal_rewards, config._target_set, iteration, hit_count_tracker,
                log_P_Fs=log_P_Fs.detach(), log_P_Bs=log_P_Bs.detach(),
                log_flows=log_flows.detach(), action_indices=action_indices_batch,
                intermediate_rewards=intermediate_rewards.detach() if intermediate_rewards is not None else None,
            )
            hit_trajectories.extend(batch_hits)

        if final_states:
            sampled_states.extend(final_states[:min(10, len(final_states))])

        if (iteration + 1) % log_every == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (iteration + 1) * config.batch_size / elapsed

            msg = f"[{iteration+1:5d}/{config.n_iterations}] "
            msg += f"Loss: {loss.item():.4f} | logZ: {log_flows[0, 0].item():.2f} | "
            if hit_rates:
                msg += f"HitRate: {hit_rate*100:.3f}% | "
                msg += f"Coverage: {len(hit_targets)}/{len(config._target_set)} ({coverage*100:.1f}%) | "
            msg += f"{eps_per_sec:.0f} eps/s"
            tqdm.write(msg)

        if (iteration + 1) % checkpoint_every == 0:
            checkpoint_path = save_dir / f"checkpoint_iter{iteration+1:05d}"
            n_targets = len(config._target_set) if config.target_sequences else 0

            partial_result = TrainingResult(
                model=model.cpu(), losses=losses.copy(), logZs=logZs.copy(),
                sampled_states=sampled_states.copy(), objective="FL-DB",
                hit_rates=hit_rates.copy() if hit_rates else None,
                target_coverages=target_coverages.copy() if target_coverages else None,
                n_targets=n_targets,
                hit_trajectories=hit_trajectories.copy() if hit_trajectories else None,
            )
            partial_result.save(str(checkpoint_path))
            model.to(device)
            tqdm.write(f">>> Checkpoint saved: {checkpoint_path}")

    n_targets = len(config._target_set) if config.target_sequences else 0
    return TrainingResult(
        model=model.cpu(), losses=losses, logZs=logZs,
        sampled_states=sampled_states, objective="FL-DB",
        hit_rates=hit_rates, target_coverages=target_coverages,
        n_targets=n_targets, hit_trajectories=hit_trajectories,
    )


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet on LET-7 miRNA")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"GFlowNet Training: LET-7 22bp miRNA")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if CONFIG["device"] == "cuda":
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            print("CUDA not available, falling back to CPU")
            CONFIG["device"] = "cpu"

    # Load data
    with open(DATA_PATH, 'r') as f:
        targets_dict = json.load(f)

    target_sequences = [list(seq) for seq in targets_dict.values()]
    unique_seqs = set(targets_dict.values())
    print(f"\nTargets: {len(unique_seqs)} unique / {len(target_sequences)} total "
          f"(length {CONFIG['max_seq_len']}bp, space 4^{CONFIG['max_seq_len']}={4**CONFIG['max_seq_len']:,})")

    # Reward function
    if REWARD_SCHEME == "entropy":
        reward_fn = EntropyWeightedHammingReward(
            target_sequences, alphabet=CONFIG["alphabet"],
            r_min=0.01, device=CONFIG["device"], entropy_weight=ENTROPY_WEIGHT,
        )
    else:
        reward_fn = HammingReward(
            target_sequences, alphabet=CONFIG["alphabet"],
            r_min=0.01, device=CONFIG["device"],
        )

    config = FastTrainingConfig(
        alphabet=CONFIG["alphabet"], max_seq_len=CONFIG["max_seq_len"],
        seed=CONFIG["seed"], hidden_layers=CONFIG["hidden_layers"],
        batch_size=CONFIG["batch_size"], n_iterations=CONFIG["n_iterations"],
        learning_rate=CONFIG["learning_rate"], device=CONFIG["device"],
        objective=CONFIG["objective"], target_sequences=target_sequences,
        explore_ratio=CONFIG["explore_ratio"], temperature=CONFIG["temperature"],
        insert_only=CONFIG["insert_only"],
    )

    print(f"Config: batch={config.batch_size}, iters={config.n_iterations}, "
          f"hidden={config.hidden_layers}, eps={config.explore_ratio}, T={config.temperature}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RESULTS_DIR / f"fldb_LET7_22bp_{timestamp}"
    save_dir.mkdir(exist_ok=True)

    config_save = {**CONFIG, "timestamp": timestamp, "reward_scheme": REWARD_SCHEME}
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)

    print(f"\nTraining... (saving to {save_dir})\n")

    start_time = time.time()
    result = train_with_checkpoints(
        reward_fn, config, save_dir,
        checkpoint_every=CHECKPOINT_EVERY, log_every=LOG_EVERY,
    )
    train_time = time.time() - start_time

    print(f"\nDone in {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"Final Z: {result.final_Z:.2f}")
    if result.hit_rates:
        print(f"Hit rate: {result.final_hit_rate*100:.4f}%")
    if result.target_coverages:
        print(f"Coverage: {result.n_unique_targets_hit}/{result.n_targets} ({result.target_coverages[-1]*100:.1f}%)")

    final_path = save_dir / "final"
    result.save(str(final_path))
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
