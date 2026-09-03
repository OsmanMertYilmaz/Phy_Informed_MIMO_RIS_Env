import numpy as np
import pandas as pd

from ris_env.dataset_writer import standardize_teacher_bank_frame


def test_hierarchical_targets_use_only_336_canonical_train_rows():
    k_count, c_count = 32, 512
    w = np.repeat(np.arange(k_count), c_count)
    z = np.tile(np.arange(c_count), k_count)
    canonical_split = np.where(
        z < 336, "train", np.where(z < 480, "holdout", "optimized")
    )
    # Holdout/optimized offsets are intentionally huge. They must not alter
    # targetBankMean or targetWMean.
    y = w.astype(float) + 0.001 * z
    y = y + np.where(canonical_split == "train", 0.0, 100.0)
    n = len(w)
    labels = pd.DataFrame({
        "bankID": np.ones(n, dtype=int),
        "splitID": np.full(n, "train"),
        "wCandidate": w,
        "zCandidate": z,
        "WIdx_i11": np.zeros(n, dtype=int),
        "WIdx_i12": np.zeros(n, dtype=int),
        "WIdx_i2": np.zeros(n, dtype=int),
        "zString": np.full(n, "0"),
        "candidateType": np.where(z < 480, "random", "optimized"),
        "anchorIndex": np.full(n, -1, dtype=int),
        "canonicalSplit": canonical_split,
        "isOptimized": z >= 480,
        "optimizationObjective": np.full(n, ""),
        "optimizationSeedRank": np.full(n, -1, dtype=int),
        "optimizationSweepCount": np.where(z >= 480, 1, 0),
        "isDuplicate": np.zeros(n, dtype=bool),
        "N_MC_base": np.full(n, 64000),
        "meanEmp": np.ones(n),
        "varEmp": np.exp(y),
        "N_MC_used": np.full(n, 64000),
        "mcRefined": np.zeros(n, dtype=bool),
        "mcConverged": np.ones(n, dtype=bool),
        "mcEarlyP99": np.zeros(n),
        "mcFinalP90Delta": np.zeros(n),
        "muSNR": np.ones(n),
        "sigma2Wick": np.ones(n),
        "wickCV2": np.ones(n),
        "Neff": np.ones(n),
        "varRatio": np.exp(y),
        "targetLogVarRatio": y,
        "targetIdentityError": np.zeros(n),
        "analyticValid": np.ones(n, dtype=bool),
        "isReliable": np.ones(n, dtype=bool),
        "optimizationInitialObjective": np.full(n, np.nan),
        "optimizationFinalObjective": np.full(n, np.nan),
        "optimizationAcceptedFlips": np.zeros(n, dtype=int),
    })

    out = standardize_teacher_bank_frame(
        labels, pd.Series(dtype=object), n_mc=64000
    )
    expected_w_mean = np.arange(k_count) + 0.001 * np.mean(np.arange(336))
    expected_bank_mean = float(np.mean(expected_w_mean))
    got_w_mean = out.groupby("wCandidate")["targetWMean"].first().to_numpy()
    assert np.allclose(got_w_mean, expected_w_mean)
    assert np.isclose(out["targetBankMean"].iloc[0], expected_bank_mean)
    assert np.max(out["hierarchyIdentityError"].to_numpy()) < 1e-12
    assert np.max(np.abs(
        out.groupby("wCandidate")["targetDeltaW"].first().mean()
    )) < 1e-12
