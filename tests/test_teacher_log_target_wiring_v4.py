import numpy as np
import pandas as pd
import torch

import ris_env.label_engine as le

from ris_env.dataset_writer import (
    DATASET_SCHEMA_VERSION,
    standardize_teacher_bank_frame,
)

from ris_env.teacher_pipeline import (
    load_packaged_gg_lookup,
)


def test_schema_v4_and_packaged_log_lookup():

    assert (
        DATASET_SCHEMA_VERSION
        == "teacher_q05gg_v4"
    )

    lookup = load_packaged_gg_lookup()

    assert (
        lookup.cv2_max
        >= 1e6 * (1 - 1e-12)
    )

    assert np.isfinite(
        lookup.log_qnorm
    ).all()


def test_writer_preserves_direct_log_target_when_q05_underflows():

    labels = pd.DataFrame({
        "bankID": [1],
        "splitID": ["train"],

        "wCandidate": [0],
        "zCandidate": [0],

        "WIdx_i11": [1],
        "WIdx_i12": [1],
        "WIdx_i2": [1],

        "zString": ["0"],
        "candidateType": ["anchor"],
        "anchorIndex": [0],

        "N_MC": [64000],

        "muSNR": [1.0],
        "meanEmp": [1.0],
        "varEmp": [1e6],

        "cv2": [1e6],
        "ggShapeA": [0.0010010005],

        # Diagnostic q05 underflow edebilir.
        "q05GG": [0.0],

        # Gerçek supervised target burada korunmalı.
        "logQ05GG": [-15748.461953870325],

        "lookupClamped": [False],
    })

    out = standardize_teacher_bank_frame(
        labels,
        pd.Series(dtype=object),
        n_mc=64_000,
    )

    assert out.loc[0, "q05GG"] == 0.0

    assert (
        out.loc[0, "logQ05GG"]
        == labels.loc[0, "logQ05GG"]
    )

    assert np.isfinite(
        out.loc[0, "logQ05GG"]
    )


def test_label_engine_returns_direct_log_target(monkeypatch):

    mean_emp = torch.tensor(
        [[2.0]],
        dtype=torch.float64,
    )

    var_emp = torch.tensor(
        [[0.5]],
        dtype=torch.float64,
    )

    analytic_mu = torch.tensor(
        [[10.0]],
        dtype=torch.float32,
    )

    def fake_empirical(*args, **kwargs):
        return {
            "meanEmp": mean_emp,
            "varEmp": var_emp,
            "seconds": 1.0,
            "candidate_count": 1,
            "sample_evaluations": 1,
            "candidate_labels_per_second": 1.0,
            "sample_evaluations_per_second": 1.0,
            "realization_pairs": 1,
            "peak_memory_MB": 0.0,
            "seed_br": 1,
            "seed_ru": 2,
        }

    def fake_analytic(*args, **kwargs):
        return analytic_mu

    def fake_lookup(*args, **kwargs):
        return object()

    expected_logq = torch.tensor(
        [[-1234.5]],
        dtype=torch.float64,
    )

    def fake_gg(mean, var, lookup):

        assert torch.equal(
            mean,
            mean_emp,
        )

        assert torch.equal(
            var,
            var_emp,
        )

        return {
            "logQ05GG": expected_logq,
            "q05GG": torch.zeros_like(mean),
            "shapeA": torch.ones_like(mean),
            "cv2": var / (mean * mean),
            "lookupClamped": torch.zeros_like(
                mean,
                dtype=torch.bool,
            ),
        }

    monkeypatch.setattr(
        le,
        "empirical_mean_variance_multi_w_z",
        fake_empirical,
    )

    monkeypatch.setattr(
        le,
        "analytic_mu_snr_multi_w_z",
        fake_analytic,
    )

    monkeypatch.setattr(
        le,
        "torch_log_lookup",
        fake_lookup,
    )

    monkeypatch.setattr(
        le,
        "symmetric_gg_logq05_torch",
        fake_gg,
    )

    out = le.run_symmetric_gg_label_engine(
        pd.Series(dtype=object),
        static_env=object(),
        W=torch.zeros(
            (1, 1),
            dtype=torch.complex64,
        ),
        gamma=torch.ones(
            (1, 1),
            dtype=torch.complex64,
        ),
        lookup=object(),
        n_mc=1,
        mc_chunk=1,
        w_chunk=1,
        z_chunk=1,
        device="cpu",
        parity=False,
    )

    assert torch.equal(
        out["logQ05GG"],
        expected_logq,
    )

    assert torch.equal(
        out["q05GG"],
        torch.zeros_like(mean_emp),
    )

    assert torch.equal(
        out["muSNR"],
        analytic_mu,
    )
