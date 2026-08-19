from pathlib import Path

import numpy as np
import torch

from ris_env.gamma_gamma import (
    GGQ05Lookup,
    symmetric_gg_q05_numpy,
)

from ris_env.label_engine import (
    torch_gg_lookup,
    symmetric_gg_q05_torch,
)


def _load_legacy_v2_lookup():

    asset = (
        Path(__file__).resolve().parents[1]
        / "src/ris_env/assets/gg_q05_lookup.npz"
    )

    a = np.load(asset)

    return GGQ05Lookup(
        log_cv2=np.asarray(
            a["log_cv2"],
            dtype=np.float64,
        ),
        qnorm=np.asarray(
            a["qnorm"],
            dtype=np.float64,
        ),
        legacy_linear_max=(
            float(
                np.asarray(
                    a["legacy_linear_max"]
                ).reshape(())
            )
            if "legacy_linear_max" in a.files
            else None
        ),
    )


def test_packaged_lookup_v2_range_and_tail_values():

    lut = _load_legacy_v2_lookup()

    assert lut.legacy_linear_max is not None
    assert lut.cv2_max >= 1024.0 * (1 - 1e-12)

    cvs = np.array([
        1.0,
        15.0,
        20.0,
        75.0,
        512.0,
        1024.0,
    ])

    q, cl = symmetric_gg_q05_numpy(
        np.ones_like(cvs),
        cvs,
        lut,
    )

    assert np.isfinite(q).all()
    assert (q > 0).all()
    assert not cl.any()

    assert np.all(np.diff(q) < 0)


def test_torch_and_numpy_lookup_v2_match():

    lut = _load_legacy_v2_lookup()

    cvs = np.geomspace(
        max(lut.cv2_min, 0.2),
        lut.cv2_max,
        2000,
    )

    mu = np.ones_like(cvs)
    var = cvs.copy()

    q_np, cl_np = symmetric_gg_q05_numpy(
        mu,
        var,
        lut,
    )

    tlut = torch_gg_lookup(
        lut,
        device="cpu",
        dtype=torch.float64,
    )

    out = symmetric_gg_q05_torch(
        torch.from_numpy(mu),
        torch.from_numpy(var),
        tlut,
    )

    q_t = out["q05GG"].numpy()
    cl_t = out["lookupClamped"].numpy()

    assert np.allclose(
        q_t,
        q_np,
        rtol=1e-11,
        atol=0,
    )

    assert np.array_equal(
        cl_t,
        cl_np,
    )


def test_lookup_clamps_only_beyond_new_guard_range():

    lut = _load_legacy_v2_lookup()

    cvs = np.array([
        lut.cv2_max * 0.999,
        lut.cv2_max,
        lut.cv2_max * 1.001,
    ])

    _, cl = symmetric_gg_q05_numpy(
        np.ones_like(cvs),
        cvs,
        lut,
    )

    assert not cl[0]
    assert not cl[1]
    assert cl[2]
