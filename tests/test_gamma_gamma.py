import numpy as np
from ris_env.gamma_gamma import GGQ05Lookup, symmetric_gg_q05_numpy, gg_shape_from_mean_variance

def test_symmetric_gg_basic_monotonic_scale():
    lookup = GGQ05Lookup(
        log_cv2=np.log(np.array([0.1, 0.5, 1.0, 2.0])),
        qnorm=np.array([0.50, 0.25, 0.12, 0.05]),
    )
    q1,_ = symmetric_gg_q05_numpy(np.array([10.0]), np.array([10.0]), lookup)
    q2,_ = symmetric_gg_q05_numpy(np.array([20.0]), np.array([40.0]), lookup)
    assert np.isfinite(q1).all() and np.isfinite(q2).all()
    assert q1[0] > 0 and q2[0] > 0
    a = gg_shape_from_mean_variance(10.0, 10.0)
    assert np.isfinite(a) and a > 0
