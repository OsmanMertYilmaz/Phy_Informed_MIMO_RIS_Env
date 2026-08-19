import numpy as np
import torch

from ris_env.gamma_gamma import GGQ05Lookup, symmetric_gg_q05_numpy
from ris_env.gamma_gamma_log import (
    load_log_lookup_npz,
    symmetric_gg_logq05_numpy,
    torch_log_lookup,
    symmetric_gg_logq05_torch,
)


def _paths():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return (
        root / "src/ris_env/assets/gg_q05_lookup.npz",
        root / "src/ris_env/assets/gg_q05_lookup_log_v3.npz",
    )


def test_v3_preserves_old_support():
    old_path,new_path = _paths()

    a = np.load(old_path)
    old = GGQ05Lookup(
        a["log_cv2"],
        a["qnorm"],
        float(np.asarray(a["legacy_linear_max"]).reshape(())),
    )
    new = load_log_lookup_npz(new_path)

    x = np.linspace(old.log_cv2[0],old.log_cv2[-1],20001)
    cv2 = np.exp(x)
    mean = np.ones_like(cv2)
    var = cv2.copy()

    q_old,clamp_old = symmetric_gg_q05_numpy(mean,var,old)
    out = symmetric_gg_logq05_numpy(mean,var,new)

    assert not np.any(clamp_old)
    assert not np.any(out["lookupClamped"])

    q_new = np.exp(out["logQ05GG"])
    rel = np.abs(q_new/q_old - 1.0)
    assert np.nanmax(rel) < 5e-12


def test_v3_extends_to_one_million_cv2():
    _,new_path = _paths()
    lookup = load_log_lookup_npz(new_path)

    cv2 = np.array([1024.0,9194.699108770275,27745.33256797937,1e5,1e6])
    mean = np.ones_like(cv2)
    var = cv2.copy()

    out = symmetric_gg_logq05_numpy(mean,var,lookup)

    assert np.isfinite(out["logQ05GG"]).all()
    assert not out["lookupClamped"].any()
    assert np.all(np.diff(out["logQ05GG"]) < 0)


def test_torch_matches_numpy():
    _,new_path = _paths()
    lookup = load_log_lookup_npz(new_path)

    cv2 = np.geomspace(0.2,1e6,4000)
    mean = np.geomspace(1e-5,1e3,4000)
    var = cv2*mean**2

    n = symmetric_gg_logq05_numpy(mean,var,lookup)

    tlookup = torch_log_lookup(lookup,device="cpu",dtype=torch.float64)
    t = symmetric_gg_logq05_torch(
        torch.from_numpy(mean),torch.from_numpy(var),tlookup
    )

    assert np.allclose(
        t["logQ05GG"].numpy(),n["logQ05GG"],rtol=0,atol=2e-12
    )


def test_log_target_survives_q05_underflow():
    _,new_path = _paths()
    lookup = load_log_lookup_npz(new_path)

    out = symmetric_gg_logq05_numpy(
        np.array([1.0]),np.array([1e6]),lookup
    )

    assert np.isfinite(out["logQ05GG"][0])
    assert out["q05GG"][0] >= 0.0
