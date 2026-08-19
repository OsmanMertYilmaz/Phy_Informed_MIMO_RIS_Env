from types import SimpleNamespace

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

    def fake_gg(mu, var, lookup):
        captured["mu"] = mu.detach().clone()
        captured["var"] = var.detach().clone()
        return {
            "q05GG": torch.ones_like(mu),
            "shapeA": torch.ones_like(mu),
            "cv2": var / (mu * mu),
            "lookupClamped": torch.zeros_like(mu, dtype=torch.bool),
        }

    monkeypatch.setattr(
        le, "empirical_mean_variance_multi_w_z", fake_empirical
    )
    monkeypatch.setattr(
        le, "analytic_mu_snr_multi_w_z", fake_analytic
    )
    monkeypatch.setattr(le, "torch_gg_lookup", fake_lookup)
    monkeypatch.setattr(le, "symmetric_gg_q05_torch", fake_gg)

    out = le.run_symmetric_gg_label_engine(
        pd.Series(dtype=object),
        static_env=object(),
        W=torch.zeros((1, 1), dtype=torch.complex64),
        gamma=torch.ones((1, 1), dtype=torch.complex64),
        lookup=object(),
        n_mc=1,
        mc_chunk=1,
        w_chunk=1,
        z_chunk=1,
        device="cpu",
        parity=False,
    )

    # Teacher GG must receive empirical mean, NOT analytic mu=10.
    assert torch.equal(captured["mu"], mean_emp)
    assert torch.equal(captured["var"], var_emp)

    # Analytic mean remains available separately as a physics feature.
    assert torch.equal(out["muSNR"], analytic_mu)
    assert torch.equal(out["meanEmp"], mean_emp)
