#!/usr/bin/env python3
"""
Resumable production teacher dataset generator.

Each persistent shard contains whole banks. Physics is executed on GPU, the
Parquet shard is created on local SSD, then copied to Drive only after close.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from importlib import resources
import gc
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ris_env.adaptive_teacher import run_teacher_bank_adaptive
from ris_env.dataset_writer import (
    make_generation_spec,
    plan_shards,
    load_or_create_manifest,
    save_manifest,
    reconcile_existing_shards,
    existing_shard_path,
    standardize_teacher_bank_frame,
    sha256_file,
    summarize_manifest,
)
from ris_env.teacher_pipeline import (
    prepare_teacher_bank,
    run_teacher_bank,
)
from ris_env.variance_ratio_contract import validate_variance_ratio_bank_frame


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def format_seconds(sec: float) -> str:
    if not np.isfinite(sec):
        return "?"
    sec = max(float(sec), 0.0)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--environments", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--local-root", default="/content/ris_teacher_work")

    p.add_argument("--n-mc", type=int, default=64_000)
    p.add_argument("--k-w", type=int, default=32)
    p.add_argument("--banks-per-shard", type=int, default=10)
    p.add_argument("--mc-chunk", type=int, default=4000)
    p.add_argument("--w-chunk", type=int, default=8)
    p.add_argument("--z-chunk", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--candidate-design",
        choices=["legacy_q05gg_v5", "variance_ratio_v1"],
        default="legacy_q05gg_v5",
    )
    p.add_argument("--optimization-sweeps", type=int, default=1)

    p.add_argument(
        "--splits",
        nargs="*",
        default=["train","validation","test_interpolation"],
    )
    p.add_argument("--max-banks", type=int, default=None)
    p.add_argument(
        "--target-completed-banks",
        type=int,
        default=None,
        help="Absolute completed-bank cap; safe for resumable pilot reruns.",
    )
    p.add_argument("--status-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if int(args.n_mc) != 64_000:
        raise ValueError(
            "Adaptive Teacher V5 requires --n-mc 64000 "
            "(frozen base MC budget)."
        )

    dev = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dev.type != "cuda" and not args.status_only:
        raise RuntimeError("Full teacher generation requires CUDA.")

    env_path = Path(args.environments)
    out_root = Path(args.output_root)
    local_root = Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    D = pd.read_csv(env_path)
    D = D[D["splitID"].astype(str).isin(args.splits)].copy()
    D = D.sort_values("bankID").reset_index(drop=True)

    asset_ref = resources.files("ris_env").joinpath("assets/gg_q05_lookup_log_v3.npz")
    with resources.as_file(asset_ref) as lookup_path:
        lookup_sha256 = sha256_file(lookup_path)

    spec = make_generation_spec(
        environment_csv=env_path,
        n_mc=args.n_mc,
        k_w=args.k_w,
        mc_chunk=args.mc_chunk,
        w_chunk=args.w_chunk,
        z_chunk=args.z_chunk,
        banks_per_shard=args.banks_per_shard,
        gg_lookup_sha256=lookup_sha256,
        candidate_design=args.candidate_design,
        optimization_sweeps=args.optimization_sweeps,
    )
    plans = [
        x for x in plan_shards(D, banks_per_shard=args.banks_per_shard)
        if x["split"] in args.splits
    ]

    manifest = load_or_create_manifest(
        out_root,
        generation_spec=spec,
        total_banks=len(D),
    )
    manifest["git_commit"] = git_commit()
    reconcile_existing_shards(out_root, manifest, plans)
    save_manifest(out_root, manifest)

    summary = summarize_manifest(manifest)
    print("=" * 94)
    print("FULL TEACHER DATASET GENERATOR")
    print("=" * 94)
    print(f"Environment CSV     : {env_path}")
    print(f"Environment SHA256  : {spec['environment_sha256'][:16]}...")
    print(f"Output root         : {out_root}")
    print(f"Local SSD work      : {local_root}")
    print(f"Device              : {dev}")
    if dev.type == "cuda":
        print(f"GPU                 : {torch.cuda.get_device_name(dev)}")
    print(f"N_MC base           : {args.n_mc:,}")
    print("N_MC max            : 512,000")
    print("MC policy           : adaptive_logq_stability_v1")
    print(f"W / Z per bank      : {args.k_w} / 512")
    print(f"Candidate design    : {args.candidate_design}")
    if args.candidate_design == "variance_ratio_v1":
        print(f"Optimization sweeps : {args.optimization_sweeps}")
        print("Teacher target      : targetLogVarRatio")
    print(f"Rows per bank       : {args.k_w * 512:,}")
    print(f"Banks per shard     : {args.banks_per_shard}")
    print(f"Total banks         : {summary['total_banks']:,}")
    print(f"Completed banks     : {summary['completed_banks']:,}")
    print(f"Remaining banks     : {summary['remaining_banks']:,}")
    print(f"Completed shards    : {summary['completed_shards']:,}")
    print(f"Progress            : {summary['progress_pct']:.2f}%")
    print("=" * 94)

    if args.status_only:
        return

    completed = set(map(int, manifest.get("completed_banks", [])))
    if (
        args.target_completed_banks is not None
        and len(completed) >= int(args.target_completed_banks)
    ):
        print(
            f"Requested completed-bank target already reached: "
            f"{len(completed):,}/{int(args.target_completed_banks):,}"
        )
        return
    processed_this_run = 0
    bank_times = []

    for plan_no, plan in enumerate(plans, start=1):
        planned_ids = list(map(int, plan["bank_ids"]))
        if all(b in completed for b in planned_ids):
            continue

        # A shard is atomic: if only a subset appears completed, refuse rather
        # than silently create an inconsistent shard.
        partial = [b for b in planned_ids if b in completed]
        if partial:
            raise RuntimeError(
                f"Manifest contains a partial planned shard {plan['split']}/"
                f"{plan['filename']}: completed subset={partial}. "
                "Use reconcile/recovery instead of mixing shard boundaries."
            )

        if args.max_banks is not None:
            remaining_budget = int(args.max_banks) - processed_this_run
            if remaining_budget <= 0:
                break

        if args.target_completed_banks is not None:
            remaining_total = (
                int(args.target_completed_banks)
                - len(completed)
            )
            if remaining_total <= 0:
                break
            if len(planned_ids) > remaining_total:
                print(
                    f"Stopping before {plan['filename']}: absolute pilot target "
                    "would split an atomic shard."
                )
                break
            if len(planned_ids) > remaining_budget:
                print(
                    f"Stopping before {plan['filename']}: --max-banks budget "
                    "would split an atomic shard."
                )
                break

        split_dir = out_root / plan["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        persistent_path = existing_shard_path(out_root, plan)

        local_split = local_root / plan["split"]
        local_split.mkdir(parents=True, exist_ok=True)
        local_tmp = local_split / (plan["filename"] + ".partial")
        local_final = local_split / plan["filename"]

        local_tmp.unlink(missing_ok=True)
        local_final.unlink(missing_ok=True)

        print()
        print("-" * 94)
        print(
            f"SHARD {plan_no}/{len(plans)} | {plan['split']} | "
            f"{plan['filename']} | banks {planned_ids[0]}..{planned_ids[-1]}"
        )
        print("-" * 94)

        writer = None
        rows_written = 0
        shard_bank_metrics = []
        shard_t0 = time.perf_counter()

        try:
            for j, bank_id in enumerate(planned_ids, start=1):
                row = D.loc[D["bankID"].astype(int) == bank_id].iloc[0]

                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(dev)

                bank_t0 = time.perf_counter()

                prepared = prepare_teacher_bank(
                    row,
                    k_w=args.k_w,
                    device=dev,
                    parity=False,
                    z_chunk=args.z_chunk,
                    candidate_design=args.candidate_design,
                    optimization_sweeps=args.optimization_sweeps,
                )
                result = run_teacher_bank_adaptive(
                    prepared,
                        half_n=32_000,
                    base_n=64_000,
                    early_p99_threshold=0.05,
                    stability_p90_threshold=0.10,
                    max_n=512_000,
                    mc_chunk=args.mc_chunk,
                    w_chunk=args.w_chunk,
                    z_chunk=args.z_chunk,
                    device=dev,
                    parity=False,
                )

                # Generator progress diagnostics only.
                # These are NOT NN inputs and do not change the Teacher target.
                _mu = result["muSNR"].to(torch.float64)
                _mean = result["meanEmp"].to(torch.float64)
                _tiny = torch.finfo(torch.float64).tiny

                _mean_ape = (
                    torch.abs(_mu - _mean)
                    / torch.clamp(torch.abs(_mean), min=_tiny)
                    * 100.0
                )

                result["mean_mdape_pct"] = float(
                    torch.median(_mean_ape).item()
                )

                result["mean_p90ape_pct"] = float(
                    torch.quantile(_mean_ape, 0.90).item()
                )
                frame = standardize_teacher_bank_frame(
                    result["labels"],
                    prepared.row_physical,
                    n_mc=args.n_mc,
                )

                expected_bank_rows = args.k_w * 512
                if len(frame) != expected_bank_rows:
                    raise RuntimeError(
                        f"bankID={bank_id}: {len(frame)} rows, "
                        f"expected {expected_bank_rows}."
                    )
                target_column = (
                    "targetLogVarRatio"
                    if args.candidate_design == "variance_ratio_v1"
                    else "logQ05GG"
                )
                if not np.isfinite(frame[target_column].to_numpy()).all():
                    raise RuntimeError(
                        f"bankID={bank_id}: non-finite {target_column}."
                    )
                q05_diag = frame["q05GG"].to_numpy(np.float64)
                if not np.isfinite(q05_diag).all():
                    raise RuntimeError(f"bankID={bank_id}: non-finite diagnostic q05GG.")
                if not (q05_diag >= 0).all():
                    raise RuntimeError(f"bankID={bank_id}: negative diagnostic q05GG.")
                if (
                    args.candidate_design == "legacy_q05gg_v5"
                    and bool(frame["lookupClamped"].astype(bool).any())
                ):
                    cv2_max=float(frame.loc[frame["lookupClamped"].astype(bool),"cv2"].max())
                    raise RuntimeError(
                        f"bankID={bank_id}: GG lookup clamp detected (max CV2={cv2_max:.6g}). "
                        "Production generation stops instead of writing clipped labels."
                    )
                if args.candidate_design == "variance_ratio_v1":
                    validate_variance_ratio_bank_frame(frame, strict=True)

                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        local_tmp,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                elif table.schema != writer.schema:
                    raise RuntimeError(
                        f"bankID={bank_id}: Parquet schema changed inside shard."
                    )
                writer.write_table(table)
                rows_written += len(frame)

                bank_sec = time.perf_counter() - bank_t0
                bank_times.append(bank_sec)
                processed_this_run += 1

                clamp_pct = 100.0 * float(frame["lookupClamped"].mean())
                peak = float(result["peak_memory_MB"])
                shard_bank_metrics.append({
                    "bankID": bank_id,
                    "seconds": bank_sec,
                    "meanMdAPE_pct": float(result["mean_mdape_pct"]),
                    "meanP90APE_pct": float(result["mean_p90ape_pct"]),
                    "lookupClamped_pct": clamp_pct,
                    "peakMemory_MB": peak,
                })

                total_done_est = len(completed) + j
                total_target = len(D)
                avg = float(np.mean(bank_times[-20:]))
                eta = avg * max(total_target - total_done_est, 0)

                print(
                    f"[{j:02d}/{len(planned_ids):02d}] bank={bank_id:4d} "
                    f"shape={int(row.nT)}/{int(row.nR)}/{int(row.nRIS)} "
                    f"time={bank_sec:6.2f}s "
                    f"meanMdAPE={result['mean_mdape_pct']:.3f}% "
                    f"clamp={clamp_pct:.3f}% "
                    f"peak={peak:.0f}MB "
                    f"overall≈{100*total_done_est/max(total_target,1):6.2f}% "
                    f"ETA≈{format_seconds(eta)}"
                )

                del frame, table, result, prepared
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            if writer is None:
                raise RuntimeError("No bank was written into the shard.")
            writer.close()
            writer = None

            if rows_written != int(plan["expected_rows"]):
                raise RuntimeError(
                    f"{plan['filename']}: rows={rows_written}, "
                    f"expected={plan['expected_rows']}."
                )

            os.replace(local_tmp, local_final)

            # Validate local file before persistent copy.
            md = pq.read_metadata(local_final)
            if md.num_rows != rows_written:
                raise RuntimeError(
                    f"Local Parquet metadata rows={md.num_rows}, expected={rows_written}."
                )

            # Copy as .copying first, then atomic rename within persistent dir.
            persistent_tmp = persistent_path.with_suffix(".parquet.copying")
            persistent_tmp.unlink(missing_ok=True)
            shutil.copy2(local_final, persistent_tmp)
            os.replace(persistent_tmp, persistent_path)

            persistent_md = pq.read_metadata(persistent_path)
            if persistent_md.num_rows != rows_written:
                raise RuntimeError(
                    f"Persistent Parquet rows={persistent_md.num_rows}, "
                    f"expected={rows_written}."
                )

            shard_sec = time.perf_counter() - shard_t0
            entry = {
                "split": str(plan["split"]),
                "filename": str(plan["filename"]),
                "bank_ids": planned_ids,
                "rows": int(rows_written),
                "bytes": int(persistent_path.stat().st_size),
                "sha256": sha256_file(persistent_path),
                "seconds": float(shard_sec),
                "bank_metrics": shard_bank_metrics,
            }
            manifest.setdefault("completed_shards", []).append(entry)
            completed.update(planned_ids)
            manifest["completed_banks"] = sorted(completed)
            save_manifest(out_root, manifest)

            print(
                f"SHARD PASS | rows={rows_written:,} | "
                f"size={persistent_path.stat().st_size/1024**2:.1f} MiB | "
                f"time={format_seconds(shard_sec)}"
            )
            print(f"Saved: {persistent_path}")

            local_final.unlink(missing_ok=True)

        except Exception:
            if writer is not None:
                writer.close()
            # Keep local partial only for debugging during the current runtime.
            raise

    summary = summarize_manifest(manifest)
    print()
    print("=" * 94)
    print("GENERATION STATUS")
    print("=" * 94)
    print(json.dumps(summary, indent=2))
    if summary["remaining_banks"] == 0:
        print("FULL TEACHER DATASET COMPLETE")
    else:
        print("Run the same command again to resume.")


if __name__ == "__main__":
    main()
