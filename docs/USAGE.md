# Using the examples

## Public results

From the repository root:

```bash
uv sync --locked --group notebook
uv run --locked --group notebook jupyter lab view_results.ipynb
```

The analysis notebook reads `examples/toy_results.json` and plots recorded results. It requires no `gfn` package. The environment uses Python 3.12, with dependencies pinned in `uv.lock`.

## Training examples

`run_training.ipynb` and `run_training_gpu.ipynb` show configuration, training calls and visualization through the private `gfn` API. The core package is intentionally absent; `uv sync` does not install it. These notebooks cannot train from the public checkout alone.

In an authorized checkout containing the private package:

```bash
uv run --locked --group notebook python scripts/run_notebook.py
```

The runner executes the CPU notebook in a fresh kernel and saves outputs under `results/`. The GPU example requires a compatible CUDA device. Linux dependencies select PyTorch's CUDA 12.8 build; macOS uses the PyPI build.

## Reading the results

The CPU source notebook completed four 20,000-episode runs on 2026-09-12. The saved summary covers the TB, DB and FL-DB examples; exact target hit rates were 7.37%, 5.99% and 9.43% respectively, among samples collected during training.

TB/DB used exact-match rewards while FL-DB used alignment rewards. These results are illustrative and do not isolate the effect of the objective, estimate final-policy performance, or establish biological function. The full LET-7 experiment and this GPU notebook were not rerun as part of that verification.

The restored `run_training_gpu_long*.ipynb` and `run_analysis*.ipynb` notebooks retain their original example code with saved outputs cleared. Analysis examples may also require pandas, SciPy, NetworkX, toytree and other optional packages; the locked environment covers the toy workflow, not every historical analysis.
