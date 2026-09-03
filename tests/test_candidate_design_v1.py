import numpy as np
import torch
from types import SimpleNamespace

from ris_env.candidate_design import (
    CANONICAL_COUNTS,
    build_canonical_z_pool,
    max_min_hamming_order,
    optimize_z_batch_one_sweep,
)
from ris_env.snr_statistics import WState, evaluate_gamma_batch


def test_canonical_480_contract_and_split():
    pack = build_canonical_z_pool(8, 4, seed=20260903)
    z = pack["z"]
    kind = pack["candidate_type"]
    split = pack["canonical_split"]

    assert z.shape == (480, 64)
    assert len({row.tobytes() for row in z}) == 480
    for name, (total, train) in CANONICAL_COUNTS.items():
        mask = kind == name
        assert int(mask.sum()) == total
        assert int(np.sum(mask & (split == "train"))) == train
        assert int(np.sum(mask & (split == "holdout"))) == total - train
    assert int(np.sum(split == "train")) == 336
    assert int(np.sum(split == "holdout")) == 144


def test_random_train_seed_order_is_deterministic_and_diverse():
    pack = build_canonical_z_pool(8, 4, seed=123)
    mask = (
        (pack["candidate_type"] == "random")
        & (pack["canonical_split"] == "train")
    )
    random_train = pack["z"][mask]
    order1 = max_min_hamming_order(random_train)
    order2 = max_min_hamming_order(random_train)
    assert np.array_equal(order1, order2)
    assert len(np.unique(order1[:8])) == 8
    selected = random_train[order1[:8]]
    distances = np.sum(selected[:, None, :] != selected[None, :, :], axis=2)
    assert int(np.min(distances[np.triu_indices(8, 1)])) > 0


def test_incremental_one_sweep_matches_direct_analytic_evaluation():
    torch.manual_seed(3)
    cd = torch.complex128
    n_ris, n_r, trajectories = 8, 2, 4

    def complex_random(*shape):
        return (
            torch.randn(*shape, dtype=torch.float64)
            + 1j * torch.randn(*shape, dtype=torch.float64)
        ).to(cd)

    def psd():
        a = complex_random(n_ris, n_ris)
        return a @ a.conj().T / n_ris

    ubar = complex_random(n_ris)
    mu_ru = complex_random(n_r, n_ris)
    cov = torch.zeros(n_r, n_r, n_ris, n_ris, dtype=cd)
    for r in range(n_r):
        cov[r, r] = psd()
    eff = torch.stack([cov[r, r].clone() for r in range(n_r)])
    env = SimpleNamespace(
        muRU=mu_ru,
        sigma2BR=torch.tensor(1.0, dtype=torch.float64),
    )
    state = WState(
        env=env,
        w=torch.ones(1, dtype=cd),
        ubarBR=ubar,
        UBR=torch.eye(n_ris, dtype=cd),
        eff_moment_kernel=eff,
        cov_kernel=cov,
    )
    objectives = ["muSNR_max", "wickCV2_min", "wickCV2_max", "Neff_min"]
    initial_z = torch.randint(0, 2, (trajectories, n_ris))
    result = optimize_z_batch_one_sweep(
        state, initial_z, objectives, parity=True
    )
    direct = evaluate_gamma_batch(state, result["gamma"])
    mu = direct["muSNR"]
    wick_cv2 = direct["sigma2Wick"] / (mu * mu)
    cmat = direct["Cmat"]
    trace = torch.diagonal(cmat, dim1=-2, dim2=-1).sum(dim=-1).real
    neff = trace * trace / torch.sum(torch.abs(cmat) ** 2, dim=(1, 2))
    expected = torch.stack([mu[0], wick_cv2[1], wick_cv2[2], neff[3]])
    assert torch.allclose(
        expected, result["final_objective"], rtol=1e-10, atol=1e-10
    )
