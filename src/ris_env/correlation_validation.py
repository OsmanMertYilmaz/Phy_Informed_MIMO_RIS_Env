"""Independent validation of the correlation PDF, without changing production.

The channel variance Var(F_eff[r]) is a second moment. It is NOT the
power variance Var(sum_r |F_eff[r]|**2). No Gaussian assumption is made
by the channel-moment contraction; accuracy of the supplied rho remains
a separate issue (production rho uses second-order angular/GH approximations).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .antenna import ArraySpec
from .channel_realizations import LinkConfig, generate_native_link_chunk
from .snr_statistics import StaticEnvironment, evaluate_gamma_batch


def _hermitian(a):
    return (a + a.transpose(-1, -2).conj()) / 2


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class ContractedChannelMoments:
    """Only O(nR*nRIS**2) state; no four-index antenna covariance."""
    mean_x: torch.Tensor
    covariance_x: torch.Tensor
    mean_g: torch.Tensor
    centered_kernels: torch.Tensor


@torch.inference_mode()
def prepare_contracted_moments(env: StaticEnvironment, w):
    """Contract BS antennas before RIS scoring; preserve dual-pol signs.

    q0[l] = w.T @ rho_BR[l] @ w.conj(), NOT w.H @ rho_BR[l] @ w.
    Cost O(L_BR*(nT**2+nRIS**2) + nR*nRIS**2).
    This independent implementation uses the same rho as production.
    """
    w = torch.as_tensor(w, dtype=env.muBR.dtype, device=env.muBR.device)
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    if w.ndim != 1 or w.numel() != env.muBR.shape[1]:
        raise ValueError("w must have shape [nT] or [nT,1]")
    if not torch.isfinite(w).all():
        raise ValueError("w must be finite")
    tx = env.rho_BR.permute(2, 0, 1)
    q0 = torch.einsum("t,ltu,u->l", w, tx, w.conj())
    sw = env.sT * w
    q1 = torch.einsum("t,ltu,u->l", sw, tx, sw.conj())
    eta = env.eKappaBR
    signs = env.sRIS[:, None] * env.sRIS[None, :]
    scale = env.sigma2BR / (2 * (1 + eta) * tx.shape[0])
    cx = scale * (
        torch.einsum("l,ijl->ij", q0 + eta*q1, env.rho_RB)
        + signs * torch.einsum("l,ijl->ij", eta*q0 + q1, env.rho_RB)
    )
    cx = _hermitian(cx)
    mx = env.muBR @ w
    mx2 = mx[:, None] * mx.conj()[None, :]
    mg = env.muRU
    mg2 = mg[:, :, None] * mg.conj()[:, None, :]
    cg = env.rho_RUhop
    # A+B+C, with no subtraction of large LOS second moments.
    kernels = cg[None] * (cx + mx2)[None] + cx[None] * mg2
    return ContractedChannelMoments(mx, cx, mg, _hermitian(kernels))


@torch.inference_mode()
def evaluate_contracted_moments(state: ContractedChannelMoments, gamma):
    """gamma includes amplitude AND phase once. Return raw, unclipped variance."""
    g = torch.as_tensor(gamma, dtype=state.mean_x.dtype, device=state.mean_x.device)
    if g.ndim == 1:
        g = g[None]
    if g.ndim != 2 or g.shape[1] != state.mean_x.numel() or g.shape[0] == 0:
        raise ValueError("gamma must be a nonempty [candidates,nRIS] array")
    if not torch.isfinite(g).all():
        raise ValueError("gamma must be finite")
    mu = (g * state.mean_x) @ state.mean_g.T
    q = torch.einsum("ci,rij,cj->cr", g, state.centered_kernels, g.conj())
    second = q.real + mu.abs().square()
    return {"mean": mu, "variance": q.real, "second_moment": second,
            "mean_power": second.sum(-1), "variance_imag_residual": q.imag}


@torch.inference_mode()
def four_sum_reference(env: StaticEnvironment, w, gamma, *, max_entries=1_000_000):
    """Small-array reference explicitly builds Cov(H_it,H_ju).

    Guard BEFORE allocating (nRIS*nT)**2 complex entries. This oracle is
    intentionally not the production algorithm and never runs at large N.
    """
    n, nt = env.muBR.shape
    if (n*nt)**2 > max_entries:
        raise ValueError("four-sum reference exceeds max_entries; use a smaller array")
    w = torch.as_tensor(w, dtype=env.muBR.dtype, device=env.muBR.device).reshape(-1)
    g = torch.as_tensor(gamma, dtype=w.dtype, device=w.device)
    if g.ndim == 1:
        g = g[None]
    if w.numel() != nt or g.ndim != 2 or g.shape[1] != n:
        raise ValueError("w/gamma dimensions do not match environment")
    si = env.sRIS[:, None, None, None]
    sj = env.sRIS[None, None, :, None]
    st = env.sT[None, :, None, None]
    su = env.sT[None, None, None, :]
    pol = 1 + env.eKappaBR*(st*su + si*sj) + si*sj*st*su
    cov_h = torch.zeros((n, nt, n, nt), dtype=w.dtype, device=w.device)
    for ell in range(env.rho_BR.shape[-1]):
        cov_h += (env.rho_RB[:, None, :, None, ell]
                  * env.rho_BR[None, :, None, :, ell] * pol)
    cov_h *= env.sigma2BR / (2*(1+env.eKappaBR)*env.rho_BR.shape[-1])
    rh = cov_h + env.muBR[:, :, None, None]*env.muBR.conj()[None, None, :, :]
    rg = env.rho_RUhop[None] + env.muRU[:, :, None]*env.muRU.conj()[:, None, :]
    # Explicit four-index weights. No contraction through x=H w in this oracle.
    weights = g[:, :, None, None, None]*g.conj()[:, None, None, :, None]
    weights = weights*w[None, None, :, None, None]*w.conj()[None, None, None, None, :]
    second = torch.stack([
        (weights*rh[None]*rg[r][None, :, None, :, None]).sum((1, 2, 3, 4)).real
        for r in range(env.muRU.shape[0])
    ], dim=1)
    mu = torch.einsum("ci,ri,it,t->cr", g, env.muRU, env.muBR, w)
    return {"mean": mu, "variance": second-mu.abs().square(), "second_moment": second}


@torch.inference_mode()
def gaussian_scalar_power_variance(cx, cg, gamma):
    """Exact ONLY for independent, zero-mean proper Gaussian x and g.

    f=g.T diag(gamma) x. This is a diagnostic control, never a replacement
    for native finite-ray/XPR power variance or the noncentral MIMO case.
    """
    if cx.ndim != 2 or cx.shape[0] != cx.shape[1] or cg.shape != cx.shape:
        raise ValueError("cx and cg must be matching square covariance matrices")
    g = torch.as_tensor(gamma, dtype=cx.dtype, device=cx.device).reshape(-1)
    if g.numel() != cx.shape[0]:
        raise ValueError("gamma length must match covariance dimension")
    a = g.conj()[:, None]*cg.T*g[None, :]
    ac = a @ cx
    mean = torch.trace(ac).real
    correction = 2*torch.trace(ac @ ac).real
    return {"mean_power": mean, "power_variance": mean.square()+correction,
            "wick_power_variance": mean.square(), "correction": correction}


def link_configs_from_prepared(prepared):
    """Use exactly the bank geometry/LSP and native shared-ray-phase generator."""
    bank = prepared["bank"]
    geom = prepared["geometry"]
    def make(lsp, tx, rx, arrival, departure):
        names = ("mu_XPR", "sigma_XPR", "mu_ASA", "sigma_ASA", "mu_ZSA", "sigma_ZSA",
                 "mu_ASD", "sigma_ASD", "mu_ZSD", "sigma_ZSD", "c_ASA", "c_ZSA",
                 "c_ASD", "c_ZSD", "mu_offset_ZOD")
        return LinkConfig(
            tx_spec=tx, rx_spec=rx,
            a_vector=arrival.detach().cpu().numpy().reshape(3),
            d_vector=departure.detach().cpu().numpy().reshape(3), fc=bank.fc,
            K=float(lsp.K_linear), is_los=bool(lsp.isLOS), M=int(lsp.M), L=int(lsp.L),
            **{name: float(getattr(lsp, name)) for name in names})
    br = make(prepared["lsp_br"], ArraySpec(bank.nT1, bank.nT2),
              ArraySpec(bank.nRIS_x, bank.nRIS_y), geom.ris2gnb, geom.gnb2ris)
    ru = make(prepared["lsp_ru"], ArraySpec(bank.nRIS_x, bank.nRIS_y),
              ArraySpec(bank.nR1, bank.nR2), geom.ue2ris, geom.ris2ue)
    return br, ru


class ChannelMCAccumulator:
    """Streaming population moments; SE uses iid samples of |F-analytic_mean|^2."""
    def __init__(self, analytic_mean):
        self.mu = analytic_mean.to(torch.complex128)
        self.n = 0
        self.sum_f = torch.zeros_like(self.mu)
        self.sum_energy = torch.zeros_like(self.mu.real)
        self.sum_energy2 = torch.zeros_like(self.mu.real)
        self.sum_delta = torch.zeros_like(self.mu)
        self.sum_pseudo = torch.zeros_like(self.mu)
        self.sum_y = torch.zeros(self.mu.shape[0], dtype=torch.float64, device=self.mu.device)
        self.sum_y2 = torch.zeros_like(self.sum_y)

    def update(self, f):
        # f: [realizations,candidates,receivers]
        f = f.to(torch.complex128)
        d = f-self.mu[None]
        energy = d.abs().square()
        y = f.abs().square().sum(-1)
        self.n += f.shape[0]
        self.sum_f += f.sum(0)
        self.sum_delta += d.sum(0)
        self.sum_energy += energy.sum(0)
        self.sum_energy2 += energy.square().sum(0)
        self.sum_pseudo += d.square().sum(0)
        self.sum_y += y.sum(0)
        self.sum_y2 += y.square().sum(0)

    def snapshot(self):
        if self.n < 2:
            raise ValueError("at least two MC samples are required")
        dmean = self.sum_delta/self.n
        energy = self.sum_energy/self.n
        variance = energy-dmean.abs().square()
        se = torch.sqrt(torch.clamp(self.sum_energy2/self.n-energy.square(), min=0)/(self.n-1))
        mean_y = self.sum_y/self.n
        return {"n": self.n, "mean": self.sum_f/self.n, "variance": variance,
                "centered_energy": energy, "centered_energy_se": se,
                "pseudo_covariance": self.sum_pseudo/self.n-dmean.square(),
                "mean_power": mean_y, "power_variance": self.sum_y2/self.n-mean_y.square()}


def _relative_error(a, b):
    den = torch.linalg.vector_norm(b.reshape(-1)).clamp_min(torch.finfo(b.real.dtype).tiny)
    return float(torch.linalg.vector_norm((a-b).reshape(-1))/den)


@torch.inference_mode()
def run_correlation_mc(prepared, gamma, *, n_mc=32_768, mc_chunk=128,
                       checkpoints=(1024, 4096, 16384), seed=20260910,
                       progress=None, reference_max_entries=1_000_000):
    """Fixed W/gamma, independent BR/RU streams, shared draws across candidates.

    The sampling path is the unmodified production native generator (exact
    trigonometry, random spreads/XPR/phases). MC discrepancy therefore includes
    angular approximation error in rho, not merely contraction error.
    progress receives JSON-ready rows at checkpoints, including the final N.
    """
    if n_mc < 2 or mc_chunk < 1:
        raise ValueError("n_mc >= 2 and mc_chunk >= 1 are required")
    env, w = prepared["static_env"], prepared["W"].reshape(-1)
    dev = w.device
    parity = w.dtype == torch.complex128
    gamma = torch.as_tensor(gamma, dtype=w.dtype, device=dev)
    if gamma.ndim == 1:
        gamma = gamma[None]
    _sync(dev)
    t0 = time.perf_counter()
    contracted = prepare_contracted_moments(env, w)
    _sync(dev)
    prepare_s = time.perf_counter()-t0
    t0 = time.perf_counter()
    theory = evaluate_contracted_moments(contracted, gamma)
    _sync(dev)
    score_s = time.perf_counter()-t0
    production = evaluate_gamma_batch(prepared["w_state"], gamma)
    checks = {
        "production_mean_rel_error": _relative_error(theory["mean"], production["muFeff"]),
        "production_variance_rel_error": _relative_error(theory["variance"], production["sigma2Feff"]),
        "production_UBR_rel_error": _relative_error(
            contracted.covariance_x + contracted.mean_x[:, None]*contracted.mean_x.conj()[None, :],
            prepared["w_state"].UBR),
        "raw_variance_min": float(theory["variance"].min()),
        "imaginary_variance_max_abs": float(theory["variance_imag_residual"].abs().max()),
        "four_sum_status": "skipped_size_guard",
    }
    if (env.muBR.numel())**2 <= reference_max_entries:
        _sync(dev)
        t0 = time.perf_counter()
        ref = four_sum_reference(env, w, gamma, max_entries=reference_max_entries)
        _sync(dev)
        checks.update(four_sum_status="computed", four_sum_seconds=time.perf_counter()-t0,
                      four_sum_mean_rel_error=_relative_error(theory["mean"], ref["mean"]),
                      four_sum_variance_rel_error=_relative_error(theory["variance"], ref["variance"]))
    br, ru = link_configs_from_prepared(prepared)
    gb = torch.Generator(device=dev).manual_seed(seed)
    gr = torch.Generator(device=dev).manual_seed(seed+1)
    acc = ChannelMCAccumulator(theory["mean"])
    stops = sorted({int(n) for n in checkpoints if 2 <= n <= n_mc} | {int(n_mc)})
    rows = []
    _sync(dev)
    t0 = time.perf_counter()
    for stop in stops:
        while acc.n < stop:
            count = min(mc_chunk, stop-acc.n)
            h = generate_native_link_chunk(br, count, generator=gb, device=dev, parity=parity)
            g = generate_native_link_chunk(ru, count, generator=gr, device=dev, parity=parity)
            x = h @ w
            f = torch.einsum("nri,ni,ci->ncr", g, x, gamma)
            if acc.n == 0:
                # Independent channel-composition order: first G diag(gamma) H, then w.
                direct = torch.einsum("nri,ci,nit->ncrt", g, gamma, h) @ w
                checks["sample_composition_rel_error"] = _relative_error(f, direct)
            acc.update(f)
        snap = acc.snapshot()
        _sync(dev)
        elapsed = time.perf_counter()-t0
        batch_rows = []
        for c in range(gamma.shape[0]):
            for r in range(env.muRU.shape[0]):
                av = float(theory["variance"][c, r])
                mv = float(snap["variance"][c, r])
                se = float(snap["centered_energy_se"][c, r])
                diff = float(snap["centered_energy"][c, r])-av
                am, mm = theory["mean"][c, r], snap["mean"][c, r]
                row = dict(n_mc=stop, candidate=c, receiver=r, analytic_variance=av,
                           mc_variance=mv, variance_rel_error=abs(mv-av)/max(abs(av), 1e-300),
                           centered_energy_se=se, centered_residual_se=diff/se if se > 0 else None,
                           analytic_mean_real=float(am.real), analytic_mean_imag=float(am.imag),
                           mc_mean_real=float(mm.real), mc_mean_imag=float(mm.imag),
                           mean_error_abs=float(abs(mm-am)),
                           mean_error_over_channel_std=float(abs(mm-am))/math.sqrt(max(av, 1e-300)),
                           pseudo_covariance_abs=float(snap["pseudo_covariance"][c, r].abs()),
                           analytic_mean_power=float(theory["mean_power"][c]),
                           mc_mean_power=float(snap["mean_power"][c]),
                           mc_power_variance=float(snap["power_variance"][c]),
                           wick_power_variance=float(production["sigma2Wick"][c]), seconds=elapsed)
                batch_rows.append(row)
        rows.extend(batch_rows)
        if progress is not None:
            progress(batch_rows)
    return {"checks": checks, "rows": rows, "timings": {
        "contracted_prepare_seconds": prepare_s, "contracted_score_seconds": score_s,
        "mc_seconds": elapsed}, "n_mc": n_mc, "mc_chunk": mc_chunk, "seed_br": seed,
        "seed_ru": seed+1, "device": str(dev), "dtype": str(w.dtype),
        "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "scope": "channel second moments; native power variance vs Wick is diagnostic only"}
