#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from ris_env.environment_dataset import (
    load_dataset_config, generate_environment_dataframe, distance_summary
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nn_dataset_4000.yaml")
    ap.add_argument("--output", default="environments_4000.csv")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    df = generate_environment_dataframe(cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.12g")

    print("=" * 78)
    print("CONTROLLED NN ENVIRONMENT DATASET")
    print("=" * 78)
    print(f"Output              : {out}")
    print(f"Banks               : {len(df):,}")
    print("\nSplit counts")
    print(df["split"].value_counts().reindex(
        list(cfg["environments"]["split"])
    ).to_string())
    print("\nScenario family counts")
    print(df["family"].value_counts().sort_index().to_string())
    print("\nnRIS counts")
    print(df["nRIS"].value_counts().sort_index().to_string())
    print("\nnT counts")
    print(df["nT"].value_counts().sort_index().to_string())
    print("\nDistance summary [m]")
    print(distance_summary(df).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("\nGeneration PASS")

if __name__ == "__main__":
    main()
