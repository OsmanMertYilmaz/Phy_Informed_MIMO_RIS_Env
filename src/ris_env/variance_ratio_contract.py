"""Validation contract for variance-ratio teacher banks and pilot shards."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


CANONICAL_CATEGORY_COUNTS = {
    "anchor": (3, 1),
    "block": (45, 19),
    "structured": (45, 19),
    "random": (243, 105),
}


def validate_variance_ratio_bank_frame(
    frame: pd.DataFrame,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    required = {
        "bankID", "wCandidate", "zCandidate", "zString",
        "candidateType", "canonicalSplit", "isOptimized", "isDuplicate",
        "optimizationObjective", "optimizationInitialObjective",
        "optimizationFinalObjective", "optimizationSweepCount",
        "varEmpMC", "sigma2Wick", "targetLogVarRatio",
        "targetIdentityError", "hierarchyIdentityError",
        "targetBankMean", "targetDeltaW", "targetDeltaZ",
        "mcConverged", "isReliable",
    }
    errors = []
    missing = required - set(frame.columns)
    if missing:
        errors.append(f"missing columns={sorted(missing)}")
        report = {"pass": False, "errors": errors}
        if strict:
            raise ValueError("Variance-ratio bank validation failed:\n- " + "\n- ".join(errors))
        return report

    bank_ids = frame["bankID"].unique()
    if len(bank_ids) != 1:
        errors.append(f"expected one bank, got {len(bank_ids)}")
    if len(frame) != 32 * 512:
        errors.append(f"rows={len(frame)} expected={32 * 512}")
    w_counts = frame.groupby("wCandidate").size()
    if len(w_counts) != 32 or not (w_counts == 512).all():
        errors.append("32 W x 512 Z row contract failed")

    canonical = frame[frame["zCandidate"] < 480]
    optimized = frame[frame["zCandidate"] >= 480]
    shared = canonical.groupby("zCandidate")["zString"].nunique()
    if len(shared) != 480 or not (shared == 1).all():
        errors.append("canonical 480 Z are not shared across all W")
    opt_counts = optimized.groupby("wCandidate").size()
    if len(opt_counts) != 32 or not (opt_counts == 32).all():
        errors.append("each W must contain 32 optimized Z")

    for name, (n_train, n_holdout) in CANONICAL_CATEGORY_COUNTS.items():
        subset = canonical[canonical["candidateType"].astype(str).eq(name)]
        per_w_train = subset[subset["canonicalSplit"].eq("train")].groupby("wCandidate").size()
        per_w_holdout = subset[subset["canonicalSplit"].eq("holdout")].groupby("wCandidate").size()
        if len(per_w_train) != 32 or not (per_w_train == n_train).all():
            errors.append(f"{name} train split must be {n_train}/W")
        if len(per_w_holdout) != 32 or not (per_w_holdout == n_holdout).all():
            errors.append(f"{name} holdout split must be {n_holdout}/W")

    finite_columns = [
        "varEmpMC", "sigma2Wick", "targetLogVarRatio",
        "targetIdentityError", "hierarchyIdentityError",
    ]
    for name in finite_columns:
        if not np.isfinite(frame[name].to_numpy(np.float64)).all():
            errors.append(f"non-finite {name}")
    var_emp = frame["varEmpMC"].to_numpy(np.float64)
    sigma_wick = frame["sigma2Wick"].to_numpy(np.float64)
    analytic_valid = (
        np.isfinite(var_emp) & np.isfinite(sigma_wick)
        & (var_emp > 0.0) & (sigma_wick > 0.0)
    )
    invalid_analytic_rate = float(np.mean(~analytic_valid))
    if not analytic_valid.any():
        errors.append("all rows have invalid empirical/Wick variance")
        identity_max = float("inf")
    else:
        recomputed_target_error = np.abs(
            frame["targetLogVarRatio"].to_numpy(np.float64)[analytic_valid]
            - (
                np.log(var_emp[analytic_valid])
                - np.log(sigma_wick[analytic_valid])
            )
        )
        identity_max = float(np.max(recomputed_target_error))
    recomputed_hierarchy_error = np.abs(
        frame["targetLogVarRatio"].to_numpy(np.float64)
        - (
            frame["targetBankMean"].to_numpy(np.float64)
            + frame["targetDeltaW"].to_numpy(np.float64)
            + frame["targetDeltaZ"].to_numpy(np.float64)
        )
    )
    hierarchy_max = float(np.max(recomputed_hierarchy_error))
    if identity_max > 1e-10:
        errors.append(f"targetIdentityError max={identity_max:.3e}")
    if hierarchy_max > 1e-10:
        errors.append(f"hierarchyIdentityError max={hierarchy_max:.3e}")

    if not (optimized["optimizationSweepCount"].astype(int) == 1).all():
        errors.append("optimized rows must use exactly one sweep")
    direction_failures = 0
    for row in optimized.itertuples(index=False):
        initial = float(row.optimizationInitialObjective)
        final = float(row.optimizationFinalObjective)
        tol = max(1e-10 * abs(initial), 1e-12)
        if str(row.optimizationObjective).endswith("_max"):
            direction_failures += int(final < initial - tol)
        else:
            direction_failures += int(final > initial + tol)
    if direction_failures:
        errors.append(f"optimization direction failures={direction_failures}")

    report = {
        "bankID": int(bank_ids[0]) if len(bank_ids) == 1 else None,
        "rows": int(len(frame)),
        "w_count": int(frame["wCandidate"].nunique()),
        "duplicate_rate": float(optimized["isDuplicate"].astype(bool).mean()),
        "mc_converged_rate": float(frame["mcConverged"].astype(bool).mean()),
        "reliable_rate": float(frame["isReliable"].astype(bool).mean()),
        "invalid_analytic_rate": invalid_analytic_rate,
        "target_identity_max": identity_max,
        "hierarchy_identity_max": hierarchy_max,
        "direction_failures": int(direction_failures),
        "errors": errors,
        "pass": not errors,
    }
    if strict and errors:
        raise ValueError("Variance-ratio bank validation failed:\n- " + "\n- ".join(errors))
    return report
