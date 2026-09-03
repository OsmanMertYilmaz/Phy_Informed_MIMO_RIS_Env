#!/usr/bin/env python3
"""Audit completed variance-ratio Parquet shards bank by bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ris_env.variance_ratio_contract import validate_variance_ratio_bank_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--max-banks", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(Path(args.dataset_root).glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet shards under {args.dataset_root}")

    reports = []
    for path in paths:
        frame = pd.read_parquet(path)
        for bank_id, bank in frame.groupby("bankID", sort=True):
            report = validate_variance_ratio_bank_frame(bank, strict=False)
            report["shard"] = str(path.relative_to(args.dataset_root))
            reports.append(report)
            print(
                f"bank={int(bank_id):5d} pass={report['pass']} "
                f"dup={100 * report.get('duplicate_rate', float('nan')):.3f}% "
                f"conv={100 * report.get('mc_converged_rate', float('nan')):.2f}% "
                f"targetErr={report.get('target_identity_max', float('nan')):.3e}"
            )
            if args.max_banks is not None and len(reports) >= args.max_banks:
                break
        if args.max_banks is not None and len(reports) >= args.max_banks:
            break

    failed = [x for x in reports if not x["pass"]]
    print(f"\nAudited banks : {len(reports)}")
    print(f"Failed banks  : {len(failed)}")
    if failed:
        for item in failed[:10]:
            print(f"- bank={item.get('bankID')} errors={item['errors']}")
        raise SystemExit(1)
    print("PILOT/DATASET CONTRACT PASS")


if __name__ == "__main__":
    main()
