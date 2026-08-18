#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from ris_env.environment_dataset import (
    load_dataset_config, validate_environment_dataframe,
    distance_summary, eta_bin_summary
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nn_dataset_4000.yaml")
    ap.add_argument("csv", nargs="?", default="environments_4000.csv")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    df = pd.read_csv(args.csv)
    report = validate_environment_dataframe(df, cfg, strict=False)

    print("=" * 88)
    print("ENVIRONMENT DATASET AUDIT")
    print("=" * 88)
    print(f"CSV                         : {args.csv}")
    print(f"Banks                       : {report['rows']:,}")
    print(f"nT minimum                  : {report['nT_min']}")
    print(f"Exact split geometry overlap: {report['exact_geometry_overlap_across_splits']}")
    print(f"Interpolation failures      : {report['interpolation_failures']}")
    print(f"Missing train geometry cells: {report['missing_train_geometry_cells']}")

    print("\nSplit counts")
    print(df["split"].value_counts().reindex(
        ["train","validation","test_interpolation"]
    ).to_string())

    print("\nScenario x split")
    print(pd.crosstab(df["family"], df["split"]).to_string())

    print("\nLOS/NLOS pair x scenario")
    print(pd.crosstab(df["family"], df["link_state"]).to_string())

    print("\nnRIS x scenario")
    print(pd.crosstab(df["family"], df["nRIS"]).to_string())

    print("\nnT")
    print(df["nT"].value_counts().sort_index().to_string())

    print("\nDistance/eta summary [m]")
    print(distance_summary(df).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\nEta regions")
    print(
        eta_bin_summary(df)
        .pivot(index="family", columns="eta_region", values="banks")
        .fillna(0).astype(int).to_string()
    )

    if report["errors"]:
        print("\nAUDIT FAIL")
        for e in report["errors"]:
            print(" -", e)
    else:
        print("\nAUDIT PASS")
        print("4,000-bank environment file can be frozen before MC label generation.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report saved           : {out}")

    raise SystemExit(0 if report["pass"] else 2)

if __name__ == "__main__":
    main()
