import pandas as pd
import torch

import ris_env.label_engine as le


def test_run_label_engine_uses_empirical_mean_for_teacher_gg(monkeypatch):

    mean_emp = torch.tensor([[2.0]], dtype=torch.float64)
    var_emp = torch.tensor([[0.5]], dtype=torch.float64)
    analytic_mu = torch.tensor([[10.0]], dtype=torch.float32)

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

    captured = {}

    def fake_lookup(*args, **kwargs):
        return object()

    def fake_gg(mean, var, lookup):

        captured["mean"] = mean.detach().clone()
        captured["var"] = var.detach().clone()

        return {
            "logQ05GG": torch.zeros_like(mean),
            "q05GG": torch.ones_like(mean),
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

    # GG teacher empirical mean kullanmalı.
    assert torch.equal(
        captured["mean"],
        mean_emp,
    )

    assert torch.equal(
        captured["var"],
        var_emp,
    )

    # Analytic mean ayrıca fizik feature olarak korunmalı.
    assert torch.equal(
        out["muSNR"],
        analytic_mu,
    )

    assert torch.equal(
        out["meanEmp"],
        mean_emp,
    )

    # Direct-log target engine'den çıkmalı.
    assert "logQ05GG" in out
