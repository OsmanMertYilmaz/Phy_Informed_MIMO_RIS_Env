import pytest
import torch

from ris_env.correlation_validation import (
    ChannelMCAccumulator, prepare_contracted_moments, evaluate_contracted_moments,
    four_sum_reference, gaussian_scalar_power_variance, run_correlation_mc,
)
from ris_env.snr_statistics import build_static_environment, prepare_w_state, evaluate_gamma_batch
from ris_env.environment import BankInput, build_deterministic_bank
from ris_env.ris_response import generate_ris_response_from_z


def random_environment(noncentral=True):
    gen = torch.Generator().manual_seed(101)
    def cn(*shape):
        return torch.randn(*shape, dtype=torch.complex128, generator=gen)
    def corr(n, l=3):
        a = cn(l, n, n)
        q = a @ a.mH
        d = q.diagonal(dim1=-2, dim2=-1).real.sqrt()
        return (q/d[:, :, None]/d[:, None, :]).permute(1, 2, 0)
    n, nt, nr = 6, 4, 4
    rb, br, ru, ur = corr(n), corr(nt), corr(nr), corr(n)
    sign = torch.tensor([1, 1, 1, -1, -1, -1])
    sigma_ru = 0.7
    cg = sigma_ru*ur.mean(-1)*(1+sign[:, None]*sign[None, :])/2
    env = build_static_environment(
        rho_RB=rb, rho_BR=br, rho_RU=ru, rho_UR=ur, rho_RUhop=cg,
        mu_XPR_BR=8, sigma_XPR_BR=3, mu_XPR_RU=7, sigma_XPR_RU=2,
        muBR=cn(n, nt)*int(noncentral), muRU=cn(nr, n)*int(noncentral),
        sigma2BR=0.8, sigma2RU=sigma_ru, device="cpu", parity=True)
    w = cn(nt)
    w /= w.norm()
    # Non-unit amplitudes and arbitrary complex phases catch conjugation mistakes.
    gamma = cn(5, n)
    return env, w, gamma


@pytest.mark.parametrize("noncentral", [False, True])
def test_four_sum_and_production_agree(noncentral):
    env, w, gamma = random_environment(noncentral)
    state = prepare_contracted_moments(env, w)
    got = evaluate_contracted_moments(state, gamma)
    ref = four_sum_reference(env, w, gamma)
    old = evaluate_gamma_batch(prepare_w_state(env, w), gamma)
    torch.testing.assert_close(got["variance"], ref["variance"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(got["mean"], ref["mean"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(got["variance"], old["sigma2Feff"], rtol=2e-12, atol=2e-12)
    # A common global RIS phase cannot change variance or power.
    phase = torch.exp(torch.tensor(0.71j, dtype=gamma.dtype))
    rotated = evaluate_contracted_moments(state, gamma*phase)
    torch.testing.assert_close(rotated["variance"], got["variance"])
    torch.testing.assert_close(rotated["mean"], got["mean"]*phase)
    assert torch.linalg.eigvalsh(state.covariance_x).min() > -1e-12


def test_guard_and_raw_negative_diagnostic():
    env, w, gamma = random_environment()
    with pytest.raises(ValueError, match="max_entries"):
        four_sum_reference(env, w, gamma, max_entries=1)
    with pytest.raises(ValueError, match="w must"):
        prepare_contracted_moments(env, w[:-1])
    state = prepare_contracted_moments(env, w)
    state.centered_kernels = -torch.eye(6, dtype=w.dtype)[None].repeat(4, 1, 1)
    assert (evaluate_contracted_moments(state, gamma)["variance"] < 0).all()


def test_streaming_population_moments_and_standard_error():
    gen = torch.Generator().manual_seed(19)
    f = torch.randn(103, 3, 4, dtype=torch.complex128, generator=gen) + 7+2j
    mu = torch.full((3, 4), 7+2j, dtype=f.dtype)
    acc = ChannelMCAccumulator(mu)
    for chunk in (f[:37], f[37:90], f[90:]):
        acc.update(chunk)
    s = acc.snapshot()
    energy = (f-mu).abs().square()
    torch.testing.assert_close(s["variance"], f.var(0, correction=0))
    torch.testing.assert_close(s["centered_energy_se"], energy.std(0, correction=1)/103**0.5)
    torch.testing.assert_close(s["power_variance"], f.abs().square().sum(-1).var(0, correction=0))


def test_gaussian_power_trace_against_independent_mc():
    # Complex off-diagonal covariance makes the transpose in D.H Cg.T D essential.
    gen = torch.Generator().manual_seed(921)
    ax = torch.tensor([[1, 0], [0.4+0.6j, 0.8]], dtype=torch.complex128)
    ag = torch.tensor([[0.7, 0], [-0.3+0.5j, 1.2]], dtype=torch.complex128)
    gamma = torch.tensor([0.8+0.2j, -0.1+0.9j], dtype=ax.dtype)
    result = gaussian_scalar_power_variance(ax@ax.mH, ag@ag.mH, gamma)
    x = torch.randn(120_000, 2, dtype=ax.dtype, generator=gen) @ ax.T
    g = torch.randn(120_000, 2, dtype=ax.dtype, generator=gen) @ ag.T
    y = (g*x*gamma).sum(-1).abs().square()
    assert abs(float(y.var(correction=0)/result["power_variance"])-1) < 0.04
    one = torch.ones((1, 1), dtype=ax.dtype)
    scalar = gaussian_scalar_power_variance(one, one, one.flatten())
    assert float(scalar["power_variance"]) == 3.0
    assert float(scalar["wick_power_variance"]) == 1.0


@pytest.mark.parametrize("los", [False, True])
def test_native_channel_mc_small_bank(los):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        scenario = "Indoor-Office-LOS" if los else "Indoor-Office-NLOS"
        bank = BankInput(scenario, scenario, 3.5e9, (0, 0, 3), (-10, 2, 3),
                         (8, 3, 1.5), 2, 1, 2, 1, 2, 2, (1, 1, 2))
        prepared = build_deterministic_bank(bank, device="cpu", parity=True)
        z = torch.tensor([[0]*8, [0, 1]*4])
        gamma = generate_ris_response_from_z(z, device="cpu", parity=True)["gamma"]
        result = run_correlation_mc(prepared, gamma, n_mc=4096, mc_chunk=128,
                                    checkpoints=(1024,), seed=2026)
        assert result["checks"]["four_sum_variance_rel_error"] < 1e-11
        assert result["checks"]["sample_composition_rel_error"] < 1e-12
        assert result["checks"]["production_variance_rel_error"] < 1e-11
        final = [row for row in result["rows"] if row["n_mc"] == 4096]
        assert max(row["variance_rel_error"] for row in final) < 0.15
        assert max(abs(row["centered_residual_se"]) for row in final) < 6
    finally:
        torch.set_num_threads(old_threads)
