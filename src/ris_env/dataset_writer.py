"""
Resumable/sharded writer utilities for production teacher datasets.

Design goals
------------
- Never write Parquet row groups directly to Google Drive while physics runs.
- Build each shard on local SSD first.
- Copy a completed shard to the persistent output directory only after it closes.
- Update manifest only after the persistent shard is verified.
- Resume by completed bank IDs / existing shard files.
- Refuse to mix incompatible generation settings in one dataset directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd


DATASET_SCHEMA_VERSION = "teacher_q05gg_v5"
VARIANCE_RATIO_SCHEMA_VERSION = "teacher_variance_ratio_v1"
ANALYTIC_MEAN_VERSION = "stage3_clamped_second_moment_v2"


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_generation_spec(
    *,
    environment_csv: str | Path,
    n_mc: int,
    k_w: int,
    mc_chunk: int,
    w_chunk: int,
    z_chunk: int,
    banks_per_shard: int,
    gg_lookup_sha256: str,
    candidate_design: str = "legacy_q05gg_v5",
    optimization_sweeps: int | None = None,
) -> Dict[str, Any]:

    environment_csv = Path(environment_csv)

    spec = {
        "schema_version": (
            VARIANCE_RATIO_SCHEMA_VERSION
            if candidate_design == "variance_ratio_v1"
            else DATASET_SCHEMA_VERSION
        ),

        "environment_csv_name":
            environment_csv.name,

        "environment_sha256":
            sha256_file(environment_csv),

        # n_mc now means the common BASE MC budget.
        "n_mc":
            int(n_mc),

        "n_mc_role":
            "adaptive_base_budget",

        "k_w":
            int(k_w),

        "z_count":
            512,

        "mc_chunk":
            int(mc_chunk),

        "w_chunk":
            int(w_chunk),

        "z_chunk":
            int(z_chunk),

        "banks_per_shard":
            int(banks_per_shard),

        "gg_lookup_sha256":
            str(gg_lookup_sha256),

        "analytic_mean_version":
            ANALYTIC_MEAN_VERSION,

        "candidate_design": str(candidate_design),

        "optimization_sweeps": (
            None if optimization_sweeps is None else int(optimization_sweeps)
        ),

        # ----------------------------------------------------
        # Frozen Adaptive-MC policy
        # ----------------------------------------------------
        "mc_policy":
            "adaptive_logq_stability_v1",

        "mc_half_n":
            32_000,

        "mc_base_n":
            64_000,

        "mc_max_n":
            512_000,

        "mc_initial_trigger":
            "P99_z_abs_logQ64_minus_logQ32",

        "mc_initial_threshold":
            0.05,

        "mc_refinement_stability":
            "P90_z_abs_logQ_new_minus_logQ_previous",

        "mc_stability_threshold":
            0.10,

        # ----------------------------------------------------
        # Teacher target
        # ----------------------------------------------------
        "target": (
            "targetLogVarRatio"
            if candidate_design == "variance_ratio_v1"
            else "logQ05GG"
        ),

        "target_representation": "direct_log_domain",

        "q05_definition":
            "symmetric_gamma_gamma(meanEmp_MC,varEmp_MC)",

        "q05GG_role":
            "diagnostic_may_underflow_to_zero",
    }

    spec["signature"] = stable_json_hash(spec)

    return spec


def plan_shards(
    environments: pd.DataFrame,
    *,
    banks_per_shard: int,
) -> List[Dict[str, Any]]:
    required = {"bankID", "splitID"}
    missing = required - set(environments.columns)
    if missing:
        raise ValueError(f"Environment CSV missing columns: {sorted(missing)}")

    if environments["bankID"].duplicated().any():
        dup = environments.loc[environments["bankID"].duplicated(), "bankID"].tolist()
        raise ValueError(f"Duplicate bankID values found, examples: {dup[:5]}")

    plans: List[Dict[str, Any]] = []
    split_order = ["train", "validation", "test_interpolation"]
    actual = list(dict.fromkeys(environments["splitID"].astype(str).tolist()))
    for split in actual:
        if split not in split_order:
            split_order.append(split)

    for split in split_order:
        sdf = environments[environments["splitID"].astype(str) == split]
        if sdf.empty:
            continue
        bank_ids = sorted(map(int, sdf["bankID"].tolist()))
        for shard_idx, start in enumerate(range(0, len(bank_ids), banks_per_shard)):
            ids = bank_ids[start:start + banks_per_shard]
            plans.append({
                "split": split,
                "shard_index": int(shard_idx),
                "bank_ids": ids,
                "expected_rows": int(len(ids) * 32 * 512),
                "filename": f"shard_{shard_idx:04d}.parquet",
            })
    return plans


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_or_create_manifest(
    output_root: str | Path,
    *,
    generation_spec: Mapping[str, Any],
    total_banks: int,
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        old_sig = manifest.get("generation_spec", {}).get("signature")
        new_sig = generation_spec.get("signature")
        if old_sig != new_sig:
            raise RuntimeError(
                "Existing teacher_dataset manifest was generated with different "
                "settings/environment CSV.\n"
                f"Existing signature: {old_sig}\n"
                f"Requested signature: {new_sig}\n"
                "Use a different output directory or restore the locked settings."
            )
        return manifest

    manifest = {
        "dataset_schema_version": generation_spec.get(
            "schema_version", DATASET_SCHEMA_VERSION
        ),
        "created_unix": time.time(),
        "updated_unix": time.time(),
        "generation_spec": dict(generation_spec),
        "total_banks": int(total_banks),
        "completed_banks": [],
        "completed_shards": [],
        "failed_banks": [],
    }
    _atomic_json_write(path, manifest)
    return manifest


def save_manifest(output_root: str | Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_unix"] = time.time()
    _atomic_json_write(Path(output_root) / "manifest.json", manifest)


def existing_shard_path(output_root: str | Path, plan: Mapping[str, Any]) -> Path:
    return Path(output_root) / str(plan["split"]) / str(plan["filename"])


def reconcile_existing_shards(
    output_root: str | Path,
    manifest: Dict[str, Any],
    plans: Sequence[Mapping[str, Any]],
) -> bool:
    """
    Recover from the narrow crash window:
        persistent .parquet copied successfully, manifest not yet updated.
    """
    changed = False
    completed = set(map(int, manifest.get("completed_banks", [])))
    known_files = {
        (str(x["split"]), str(x["filename"]))
        for x in manifest.get("completed_shards", [])
    }

    import pyarrow.parquet as pq

    for plan in plans:
        path = existing_shard_path(output_root, plan)
        key = (str(plan["split"]), str(plan["filename"]))
        if not path.exists() or key in known_files:
            continue

        table = pq.read_table(path, columns=["bankID"])
        ids = sorted(set(map(int, table.column("bankID").to_pylist())))
        expected_ids = sorted(map(int, plan["bank_ids"]))
        if ids != expected_ids:
            raise RuntimeError(
                f"Existing shard {path} does not match planned bank IDs.\n"
                f"Found: {ids}\nExpected: {expected_ids}"
            )
        if table.num_rows != int(plan["expected_rows"]):
            raise RuntimeError(
                f"Existing shard {path} has {table.num_rows} rows, "
                f"expected {plan['expected_rows']}."
            )

        entry = {
            "split": str(plan["split"]),
            "filename": str(plan["filename"]),
            "bank_ids": expected_ids,
            "rows": int(table.num_rows),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "reconciled": True,
        }
        manifest.setdefault("completed_shards", []).append(entry)
        completed.update(expected_ids)
        known_files.add(key)
        changed = True

    if changed:
        manifest["completed_banks"] = sorted(completed)
        save_manifest(output_root, manifest)
    return changed


def add_constant_physical_columns(
    labels: pd.DataFrame,
    row_physical: pd.Series,
) -> pd.DataFrame:
    """
    Persist deployable bank-level environment/LSP scalars next to each candidate.

    Parquet dictionary/RLE compression makes these repeated constants cheap.
    Empirical targets are NOT treated as inputs here; they remain separate columns.
    """
    out = labels.copy()
    skip = set(out.columns)
    for key, value in row_physical.items():
        if key in skip:
            continue
        if isinstance(value, (str, bool, np.bool_)):
            out[key] = value
        elif np.isscalar(value):
            if pd.isna(value):
                out[key] = value
            elif isinstance(value, (int, np.integer)):
                out[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[key] = float(value)
    return out


def standardize_teacher_bank_frame(
    labels: pd.DataFrame,
    row_physical: pd.Series,
    *,
    n_mc: int,
) -> pd.DataFrame:
    """
    Final per-bank Parquet schema.

    Supports the legacy logQ05GG target and the frozen variance-ratio target.

    Adaptive Teacher rows contain variable MC budgets.
    Therefore empirical target-side moments are stored as:

        meanEmpMC
        varEmpMC
        N_MC_used

    rather than pretending every row used 64k samples.

    logQ05GG is always preserved directly from the
    log-domain Gamma-Gamma engine.
    """

    out = add_constant_physical_columns(
        labels,
        row_physical,
    )

    adaptive = (
        "N_MC_used" in out.columns
    )


    variance_ratio = "targetLogVarRatio" in out.columns
    target_name = "targetLogVarRatio" if variance_ratio else "logQ05GG"
    if target_name not in out.columns:
        raise ValueError(f"Teacher labels missing target {target_name}.")
    target_values = out[target_name].to_numpy(np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError(f"Teacher labels contain non-finite {target_name}.")


    # ========================================================
    # Adaptive V5 path
    # ========================================================

    if adaptive:

        required = {
            "meanEmp",
            "varEmp",
            "N_MC_used",
            "mcRefined",
            "mcConverged",
            "mcEarlyP99",
            "mcFinalP90Delta",
        }

        missing = (
            required
            - set(out.columns)
        )

        if missing:
            raise ValueError(
                "Adaptive Teacher labels missing columns: "
                f"{sorted(missing)}"
            )


        out = out.rename(
            columns={
                "meanEmp":
                    "meanEmpMC",

                "varEmp":
                    "varEmpMC",
            }
        )

        if variance_ratio:
            required_vr = {
                "sigma2Wick", "wickCV2", "Neff", "varRatio",
                "targetIdentityError", "analyticValid", "isReliable",
                "canonicalSplit", "isOptimized", "isDuplicate",
            }
            missing_vr = required_vr - set(out.columns)
            if missing_vr:
                raise ValueError(
                    "Variance-ratio Teacher labels missing columns: "
                    f"{sorted(missing_vr)}"
                )

            canonical_train = out["canonicalSplit"].astype(str).eq("train")
            train_counts = (
                out.loc[canonical_train]
                .groupby("wCandidate", sort=True)
                .size()
            )
            if len(train_counts) != 32 or not (train_counts == 336).all():
                raise ValueError(
                    "Hierarchical targets require exactly 336 canonical-train "
                    "Z rows for each of 32 W candidates."
                )

            train_rows = out.loc[
                canonical_train, ["wCandidate", "targetLogVarRatio"]
            ]
            w_mean = train_rows.groupby("wCandidate")["targetLogVarRatio"].mean()
            w_std = train_rows.groupby("wCandidate")["targetLogVarRatio"].std(ddof=0)
            bank_mean = float(train_rows["targetLogVarRatio"].mean())
            bank_std = float(train_rows["targetLogVarRatio"].std(ddof=0))
            mapped_mean = out["wCandidate"].map(w_mean).to_numpy(np.float64)
            mapped_std = out["wCandidate"].map(w_std).to_numpy(np.float64)
            out["targetBankMean"] = bank_mean
            out["targetBankStd"] = bank_std
            out["targetWMean"] = mapped_mean
            out["targetWStd"] = mapped_std
            out["targetDeltaW"] = mapped_mean - bank_mean
            out["targetDeltaZ"] = (
                out["targetLogVarRatio"].to_numpy(np.float64) - mapped_mean
            )
            reconstructed = (
                out["targetBankMean"].to_numpy(np.float64)
                + out["targetDeltaW"].to_numpy(np.float64)
                + out["targetDeltaZ"].to_numpy(np.float64)
            )
            out["hierarchyIdentityError"] = np.abs(
                out["targetLogVarRatio"].to_numpy(np.float64) - reconstructed
            )


        n_used = out[
            "N_MC_used"
        ].to_numpy(
            np.int64
        )

        if np.any(n_used < int(n_mc)):
            raise ValueError(
                "Adaptive N_MC_used is smaller than "
                f"base n_mc={int(n_mc)}."
            )

        if np.any(n_used > 512_000):
            raise ValueError(
                "Adaptive N_MC_used exceeded frozen "
                "512k production cap."
            )


        lead = [
            "bankID",
            "splitID",

            "wCandidate",
            "zCandidate",

            "WIdx_i11",
            "WIdx_i12",
            "WIdx_i2",

            "zString",
            "candidateType",
            "anchorIndex",

            "canonicalSplit",
            "isOptimized",
            "optimizationObjective",
            "optimizationSeedRank",
            "optimizationSweepCount",
            "isDuplicate",

            "N_MC_base",
        ]


        target_tail = [
            # deployable analytic physics quantity
            "muSNR",

            # MC-only target-side diagnostics
            "meanEmpMC",
            "varEmpMC",
            "N_MC_used",
            "mcRefined",
            "mcConverged",
            "mcEarlyP99",
            "mcFinalP90Delta",

            # GG diagnostics / target
            "cv2",
            "ggShapeA",
            "q05GG",
            "logQ05GG",
            "lookupClamped",
        ]

        if variance_ratio:
            target_tail = [
                "muSNR", "sigma2Wick", "wickCV2", "Neff",
                "meanEmpMC", "varEmpMC", "N_MC_used",
                "mcRefined", "mcConverged", "mcEarlyP99",
                "mcFinalP90Delta", "analyticValid", "isReliable",
                "targetIdentityError", "hierarchyIdentityError",
                "optimizationInitialObjective", "optimizationFinalObjective",
                "optimizationAcceptedFlips",
                "targetBankMean", "targetBankStd",
                "targetWMean", "targetWStd",
                "targetDeltaW", "targetDeltaZ",
                "varRatio", "targetLogVarRatio",
                "cv2", "ggShapeA", "q05GG", "logQ05GG", "lookupClamped",
            ]


    # ========================================================
    # Legacy fixed-N path
    #
    # Kept only so existing tests / old non-adaptive utilities
    # do not break.
    # ========================================================

    else:

        if int(n_mc) == 64_000:

            mean_name = "meanEmp64k"
            var_name = "varEmp64k"

        else:

            suffix = (
                f"{int(n_mc)//1000}k"
                if int(n_mc) % 1000 == 0
                else str(int(n_mc))
            )

            mean_name = (
                f"meanEmp{suffix}"
            )

            var_name = (
                f"varEmp{suffix}"
            )


        out = out.rename(
            columns={
                "meanEmp":
                    mean_name,

                "varEmp":
                    var_name,
            }
        )


        lead = [
            "bankID",
            "splitID",

            "wCandidate",
            "zCandidate",

            "WIdx_i11",
            "WIdx_i12",
            "WIdx_i2",

            "zString",
            "candidateType",
            "anchorIndex",

            "N_MC",
        ]


        target_tail = [
            "muSNR",
            mean_name,
            var_name,
            "cv2",
            "ggShapeA",
            "q05GG",
            "logQ05GG",
            "lookupClamped",
        ]


    # ========================================================
    # Stable column order
    # ========================================================

    ordered = []

    for c in lead:
        if (
            c in out.columns
            and c not in ordered
        ):
            ordered.append(c)

    for c in out.columns:
        if (
            c not in ordered
            and c not in target_tail
        ):
            ordered.append(c)

    for c in target_tail:
        if (
            c in out.columns
            and c not in ordered
        ):
            ordered.append(c)

    return out[ordered]


def summarize_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    total = int(manifest.get("total_banks", 0))
    done = len(set(map(int, manifest.get("completed_banks", []))))
    shards = len(manifest.get("completed_shards", []))
    return {
        "total_banks": total,
        "completed_banks": done,
        "remaining_banks": max(total - done, 0),
        "completed_shards": shards,
        "progress_pct": 100.0 * done / max(total, 1),
    }
