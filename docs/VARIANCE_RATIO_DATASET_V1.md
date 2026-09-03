# Variance-ratio teacher dataset v1

This production path implements the frozen 27,200-bank specification with one
bit-flip sweep per optimized trajectory.

## Frozen contract

- 27,200 banks: 19,040 train, 4,080 validation, 4,080 final test.
- 32 Type-I rank-1 W candidates per bank.
- 512 Z per W: 480 shared canonical and 32 W-specific optimized.
- Canonical split: 336 train and 144 holdout.
- Optimized split: four analytical objectives times eight diverse seeds.
- Supervised target: `targetLogVarRatio = log(varEmpMC)-log(sigma2Wick)`.
- One complete coordinate sweep (`optimizationSweepCount=1`).
- Adaptive MC starts at 64,000 and retains stateful continuation.
- Channel semantics remain Case 2: shared ray/polarization random phases.

## Colab pilot

After cloning the repository and mounting Drive at `/content/drive`, run:

```bash
pip install -e .
python scripts/run_variance_ratio_pilot.py --pilot-banks 100
```

The pilot uses one bank per Parquet shard. Each completed bank is first closed
and verified on local SSD, then copied atomically to Google Drive and recorded
in `manifest.json`. Rerunning the same command resumes from the manifest and
stops when the absolute 100-bank target is reached.

The pilot is stored separately from full production. After its audit passes,
full production should use a new output directory and a larger atomic shard
size selected from the measured bank runtime and file-size results.
