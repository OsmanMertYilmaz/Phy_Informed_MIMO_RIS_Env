#!/usr/bin/env python3
"""Resumable 100-bank Colab pilot with per-bank Google Drive checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ris_env.environment_dataset import (
    generate_environment_dataframe,
    load_dataset_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Phy_Informed_MIMO_RIS_Env",
    )
    ap.add_argument("--pilot-banks", type=int, default=100)
    ap.add_argument("--local-root", default="/content/ris_variance_ratio_work")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    drive_root = Path(args.drive_root)
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive root does not exist: {drive_root}. Mount Google Drive first."
        )

    config = repo / "configs/variance_ratio_dataset_27200.yaml"
    env_dir = drive_root / "environments"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_path = env_dir / "environments_27200.csv"
    if not env_path.exists():
        cfg = load_dataset_config(config)
        frame = generate_environment_dataframe(cfg)
        with tempfile.TemporaryDirectory(prefix="ris_env_") as tmp:
            local = Path(tmp) / env_path.name
            frame.to_csv(local, index=False, float_format="%.12g")
            copying = env_path.with_suffix(".csv.copying")
            shutil.copy2(local, copying)
            copying.replace(env_path)
        print(f"Environment CSV created: {env_path}")
    else:
        print(f"Environment CSV reused : {env_path}")

    output_root = drive_root / f"teacher_variance_ratio_pilot_{int(args.pilot_banks)}"
    cmd = [
        sys.executable,
        str(repo / "scripts/generate_teacher_dataset.py"),
        "--environments", str(env_path),
        "--output-root", str(output_root),
        "--local-root", str(args.local_root),
        "--candidate-design", "variance_ratio_v1",
        "--optimization-sweeps", "1",
        "--banks-per-shard", "1",
        "--target-completed-banks", str(int(args.pilot_banks)),
        "--splits", "train", "validation", "final_test",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    print("Starting/resuming pilot. Every completed bank is copied to Drive.")
    subprocess.run(cmd, cwd=repo, check=True)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/audit_variance_ratio_dataset.py"),
            str(output_root),
            "--max-banks", str(int(args.pilot_banks)),
        ],
        cwd=repo,
        check=True,
    )


if __name__ == "__main__":
    main()
