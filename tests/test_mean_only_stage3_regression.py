import torch

from ris_env.label_engine import (
    MeanOnlyWState,
    evaluate_analytic_mu_snr_mean_only,
)


def test_mean_only_reproduces_stage3_nonnegative_projection():
    # One receive branch, two RIS elements.
    # Construct an intentionally non-PSD second-moment kernel so raw
    # gamma^T K gamma* is negative.  This reproduces the failure mode seen in
    # the long-range RMa bank.  Stage-3 does NOT return the raw quadratic form;
    # it clamps variance = second - |mu|^2 to >=0 and then adds |mu|^2.
    K = torch.tensor(
        [[[-2.0, 0.0],
          [ 0.0,-1.0]]],
        dtype=torch.complex64,
    )
    ubar = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    muRU = torch.tensor([[0.25, 0.0]], dtype=torch.complex64)

    state = MeanOnlyWState(
        eff_kernel=K,
        ubarBR=ubar,
        muRU=muRU,
    )

    gamma = torch.tensor(
        [[1.0, 1.0],
         [1.0,-1.0]],
        dtype=torch.complex64,
    )

    got = evaluate_analytic_mu_snr_mean_only(
        state, gamma, z_chunk=2
    )

    # mu = 0.25 for both candidates. Since second_eff is negative,
    # sigma2 clamps to zero and muSNR = |mu|^2 = 0.0625.
    expected = torch.full((2,), 0.0625, dtype=torch.float32)

    assert torch.allclose(got, expected, rtol=0, atol=1e-7)
    assert torch.all(got > 0)


def test_mean_only_keeps_raw_second_when_it_is_physically_large_enough():
    K = torch.tensor(
        [[[2.0, 0.0],
          [0.0, 1.0]]],
        dtype=torch.complex64,
    )
    ubar = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    muRU = torch.tensor([[0.25, 0.0]], dtype=torch.complex64)
    state = MeanOnlyWState(K, ubar, muRU)

    gamma = torch.tensor([[1.0, 1.0]], dtype=torch.complex64)
    got = evaluate_analytic_mu_snr_mean_only(state, gamma, z_chunk=1)

    # second_eff=3, |mu|^2=0.0625 -> sigma2=2.9375 -> total=3.
    assert torch.allclose(
        got, torch.tensor([3.0], dtype=torch.float32),
        rtol=0, atol=1e-6,
    )
