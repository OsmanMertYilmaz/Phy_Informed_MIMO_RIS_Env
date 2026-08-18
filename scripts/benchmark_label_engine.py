"""Benchmark entry-point scaffold.

The validated B3-B benchmark notebook is retained under notebooks/validation.
The production CLI will be wired to the controlled 4,000-bank scenario
generator after that generator is frozen.
"""

from pathlib import Path

if __name__ == "__main__":
    print("See notebooks/validation/stage8b3b_multi_w_z_symmetric_gg_benchmark.ipynb")
    print("Next: wire this CLI to configs/nn_dataset_4000.yaml")
