# GFlowNet for Sequence Generation

A modular GFlowNet implementation for learning to generate sequences with probability proportional to reward.

## Quick Start

```python
from gfn import train, TrainingConfig
from gfn.reward import TargetMatchReward
from gfn.visualization import plot_flow_network

targets = [['A', 'B', 'C', 'ε'], ['C', 'B', 'A', 'ε']]
result = train(TargetMatchReward(targets), TrainingConfig(n_episodes=20_000))
plot_flow_network(result.model, target_sequences=targets)
```

Or run `run_training.ipynb` for a full demo.

## Structure

```
gfn/
├── env.py           # States, actions, transitions
├── model.py         # Trajectory Balance model
├── reward.py        # Reward functions
├── training.py      # Training loop
└── visualization.py # Plotting
```
