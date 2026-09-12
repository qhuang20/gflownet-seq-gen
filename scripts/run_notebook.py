"""Execute the first notebook in a fresh kernel using this Python environment."""

import json
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    notebook = nbformat.read(root / "run_training.ipynb", as_version=4)
    output = output_dir / "run_training.executed.ipynb"
    started = time.monotonic()

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="gfn-kernel-") as temporary:
        spec = Path(temporary) / "gfn"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "GFlowNet (current environment)",
            "language": "python",
            "env": {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        }))
        manager = KernelManager(
            kernel_name="gfn",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False),
        )
        client = NotebookClient(
            notebook, km=manager, kernel_name="gfn", timeout=1800,
            resources={"metadata": {"path": str(root)}}, on_cell_start=progress,
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    errors = [o for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors
    elapsed = time.monotonic() - started
    print(f"Completed in {elapsed:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
