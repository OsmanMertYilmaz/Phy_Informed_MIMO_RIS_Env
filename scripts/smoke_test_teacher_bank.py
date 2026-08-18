#!/usr/bin/env python3
"""One-bank full teacher smoke test: 32 W x 512 Z x 64k MC."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ris_env.teacher_pipeline import (
    Z_TYPE_COUNTS,
    prepare_teacher_bank,
    run_teacher_bank,
)


def select_row(df: pd.DataFrame, mode: str, bank_id: int | None):
    if bank_id is not None:
        s = df[df["bankID"].astype(int) == int(bank_id)]
        if s.empty:
            raise ValueError(f"bankID={bank_id} not found.")
        return s.iloc[0]

    if mode == "worst_case":
        # Heavy deterministic/stochastic shape first.
        return (
            df.sort_values(
                ["nRIS", "nT", "nR", "bankID"],
                ascending=[False, False, False, True],
            )
            .iloc[0]
        )

    if mode == "first":
        return df.sort_values("bankID").iloc[0]

    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nn_dataset_4000.yaml")
    ap.add_argument("--environments", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--select", choices=["worst_case", "first"], default="worst_case")
    ap.add_argument("--bank-id", type=int, default=None)

    ap.add_argument("--n-mc", type=int, default=64_000)
    ap.add_argument("--k-w", type=int, default=32)
    ap.add_argument("--mc-chunk", type=int, default=256)
    ap.add_argument("--w-chunk", type=int, default=4)
    ap.add_argument("--z-chunk", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dev.type != "cuda":
        print("WARNING: CUDA is not active. Full 64k smoke test may be very slow.")

    df = pd.read_csv(args.environments)
    row = select_row(df, args.select, args.bank_id)

    print("=" * 90)
    print("ONE-BANK TEACHER SMOKE TEST")
    print("=" * 90)
    print(f"Device                    : {dev}")
    if dev.type == "cuda":
        print(f"GPU                       : {torch.cuda.get_device_name(dev)}")
    print(f"Bank                      : {int(row.bankID)}")
    print(f"Split                     : {row.splitID}")
    print(f"Scenario BR / RU          : {row.scenario_BR} / {row.scenario_RU}")
    print(f"nT / nR / nRIS            : {int(row.nT)} / {int(row.nR)} / {int(row.nRIS)}")
    print(f"N_MC                      : {args.n_mc:,}")
    print()

    t0 = time.perf_counter()
    prepared = prepare_teacher_bank(
        row,
        k_w=args.k_w,
        device=dev,
        parity=False,
        z_chunk=args.z_chunk,
    )
    prep_s = time.perf_counter() - t0

    print(f"W unique                  : {len(np.unique(prepared.WIdx.cpu().numpy(), axis=0))} / {args.k_w}")
    print(f"Z unique                  : {len({x.tobytes() for x in prepared.z.cpu().numpy().astype(np.uint8)})} / 512")
    for key in ("anchor","global_random","density_stratified","structured","local_perturbation"):
        print(f"Z {key:24s}: {int(np.sum(prepared.z_candidate_type == key))}")
    print(f"Teacher prepare            : {prep_s:.3f} s")
    print()

    result = run_teacher_bank(
        prepared,
        n_mc=args.n_mc,
        mc_chunk=args.mc_chunk,
        w_chunk=args.w_chunk,
        z_chunk=args.z_chunk,
        device=dev,
        parity=False,
    )

    labels = result["labels"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write locally/Drive as Parquet; pyarrow is a runtime dependency.
    labels.to_parquet(out, index=False, compression="zstd")

    q = labels["q05GG"].to_numpy(np.float64)
    v = labels["varEmp"].to_numpy(np.float64)
    clamp_pct = 100.0 * labels["lookupClamped"].mean()

    checks = {
        "W_unique": result["w_unique"] == args.k_w,
        "Z_unique": result["z_unique"] == 512,
        "Z_type_counts": result["z_type_counts"] == Z_TYPE_COUNTS,
        "rows": len(labels) == args.k_w * 512,
        "var_finite_positive": bool(np.isfinite(v).all() and (v > 0).all()),
        "q05_finite_positive": bool(np.isfinite(q).all() and (q > 0).all()),
    }

    print("\n" + "=" * 90)
    print("SMOKE RESULT")
    print("=" * 90)
    print(f"Output rows               : {len(labels):,}")
    print(f"Empirical MC + contraction: {result['empirical_seconds']:.3f} s")
    print(f"Analytic muSNR            : {result['analytic_mu_seconds']:.3f} s")
    print(f"Symmetric-GG lookup       : {result['gg_lookup_seconds']:.6f} s")
    print(f"Teacher engine total      : {result['total_seconds']:.3f} s")
    print(f"FINAL GG labels/s         : {result['labels_per_second']:,.2f}")
    print(f"Sample evaluations/s      : {result['sample_evaluations_per_second']:,.0f}")
    print(f"Peak CUDA allocated       : {result['peak_memory_MB']:.1f} MB")
    print(f"Lookup clamped            : {clamp_pct:.3f}%")
    print(f"Analytic-vs-MC mean MdAPE : {result['mean_mdape_pct']:.3f}%")
    print(f"Analytic-vs-MC mean P90   : {result['mean_p90ape_pct']:.3f}%")
    print(f"q05GG min / median / max  : {q.min():.6g} / {np.median(q):.6g} / {q.max():.6g}")
    print(f"Saved                     : {out}")

    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print("\nSMOKE TEST FAIL")
        for k in failed:
            print(" -", k)
        raise SystemExit(2)

    # At locked N=64k, mean consistency should be comfortably inside this
    # loose guard. It catches structural bugs without overfitting to one bank.
    if args.n_mc >= 64_000:
        if result["mean_mdape_pct"] > 3.0 or result["mean_p90ape_pct"] > 5.0:
            print("\nSMOKE TEST FAIL")
            print(" - analytic-vs-MC mean consistency")
            raise SystemExit(2)

    print("\nSMOKE TEST PASS")


if __name__ == "__main__":
    main()
