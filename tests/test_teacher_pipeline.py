import numpy as np

from ris_env.teacher_pipeline import (
    Z_TYPE_COUNTS,
    build_base_z_pool,
    add_local_perturbations,
    load_packaged_gg_lookup,
)


def test_packaged_gg_lookup():
    lut = load_packaged_gg_lookup()
    assert lut.log_cv2.ndim == 1
    assert lut.qnorm.ndim == 1
    assert len(lut.log_cv2) > 1000
    assert np.all(np.diff(lut.log_cv2) > 0)
    assert np.isfinite(lut.qnorm).all()
    assert (lut.qnorm >= 0).all()


def test_final_z_pool_counts_and_uniqueness_without_physics():
    for nx, ny in [(8,4), (8,8), (16,8), (16,16)]:
        base_z, base_types = build_base_z_pool(nx, ny, seed=123)
        assert base_z.shape[0] == 420
        assert len({x.tobytes() for x in base_z}) == 420

        z, types, anchors = add_local_perturbations(
            base_z,
            base_types,
            seed_indices=[0,1,2,3],
            seed=456,
        )
        assert z.shape == (512, 2*nx*ny)
        assert len({x.tobytes() for x in z}) == 512

        counts = {k: int(np.sum(types == k)) for k in Z_TYPE_COUNTS}
        assert counts == Z_TYPE_COUNTS
        assert np.sum(anchors >= 0) == 92
