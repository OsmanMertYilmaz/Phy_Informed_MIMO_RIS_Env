import numpy as np
import torch

from ris_env.teacher_pipeline import load_packaged_gg_lookup
from ris_env.gamma_gamma import symmetric_gg_q05_numpy
from ris_env.label_engine import torch_gg_lookup, symmetric_gg_q05_torch


def test_packaged_lookup_v2_range_and_tail_values():
    lut=load_packaged_gg_lookup()
    assert lut.legacy_linear_max is not None
    assert lut.cv2_max >= 1024.0 * (1-1e-12)

    cvs=np.array([1.0,15.0,20.0,75.0,512.0,1024.0])
    q,cl=symmetric_gg_q05_numpy(np.ones_like(cvs),cvs,lut)
    assert not cl.any()
    assert np.all(q>0)
    # Reference values from the symmetric Gamma-Gamma Meijer-G CDF inversion.
    ref=np.array([
        0.10734311067970262,  # legacy lookup is intentionally preserved here
        3.490809430788376e-06, # legacy lookup preserved below breakpoint
        2.7564494126840455e-07,
        3.0114688605055086e-15,
        4.0324652649361255e-43,
        4.072072934765906e-62,
    ])
    # Legacy section follows the historical table; extension is numerical-exact
    # to well below 0.1% with log-tail interpolation.
    assert abs(q[0]/ref[0]-1) < 5e-4
    assert abs(q[1]/ref[1]-1) < 5e-3
    assert np.max(np.abs(q[2:]/ref[2:]-1)) < 1e-3


def test_torch_and_numpy_lookup_v2_match():
    lut=load_packaged_gg_lookup()
    cvs=np.array([0.2,1.0,10.0,15.4,20.0,75.0,256.0,900.0])
    qn,cn=symmetric_gg_q05_numpy(np.ones_like(cvs),cvs,lut)
    tlut=torch_gg_lookup(lut,device='cpu',dtype=torch.float64)
    out=symmetric_gg_q05_torch(
        torch.ones(len(cvs),dtype=torch.float64),
        torch.tensor(cvs,dtype=torch.float64),
        tlut,
    )
    qt=out['q05GG'].numpy(); ct=out['lookupClamped'].numpy()
    assert np.allclose(qt,qn,rtol=1e-12,atol=0)
    assert np.array_equal(ct,cn)


def test_lookup_clamps_only_beyond_new_guard_range():
    lut=load_packaged_gg_lookup()
    q,c=symmetric_gg_q05_numpy(np.array([1.0]),np.array([2000.0]),lut)
    assert bool(c[0])
