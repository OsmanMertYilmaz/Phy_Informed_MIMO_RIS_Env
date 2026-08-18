"""
Stage 3 GPU analytical RIS statistics engine.

Ports the current MATLAB analytical path:
    generate_eff_moments
    evaluate_gamma_metric   (without empirical Feff delta and without V3 feature summary)

Key optimization:
    NEVER materialize G = gamma*gamma^H for a candidate batch.

For any matrix K:
    sum_ij gamma_i conj(gamma_j) K_ij
      = gamma^T K conj(gamma)

so C RIS candidates are evaluated with batched matrix multiplies using
gamma [C,nRIS], not G [C,nRIS,nRIS]. This is critical for nRIS=512.

Current assumptions:
    - nl = 1
    - nRIS, nT, nR are even (dual polarization)
    - rho arrays use MATLAB ordering already validated in Stage 2
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import time
import numpy as np
import torch


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool):
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool):
    return torch.complex128 if parity else torch.complex64


def _ctensor(x, *, device, parity):
    return torch.as_tensor(x, dtype=_cdtype(parity), device=device)


def _rtensor(x, *, device, parity):
    return torch.as_tensor(x, dtype=_rdtype(parity), device=device)


def _hermitianize(A):
    return 0.5 * (A + A.transpose(-1, -2).conj())


def _pol_sign(n: int, *, device, dtype):
    if n % 2:
        raise ValueError("dual-pol dimension must be even")
    return torch.cat((
        torch.ones(n//2, device=device, dtype=dtype),
        -torch.ones(n//2, device=device, dtype=dtype),
    ))


def _xpr_inverse_mean(mu_xpr, sigma_xpr, *, device, parity):
    rd = _rdtype(parity)
    mu = torch.as_tensor(mu_xpr, dtype=rd, device=device)
    sig = torch.as_tensor(sigma_xpr, dtype=rd, device=device)
    ln10 = math.log(10.0)
    return torch.exp(
        -ln10 * mu / 10.0
        + (ln10 * ln10) * sig * sig / 200.0
    )


@dataclass
class StaticEnvironment:
    rho_RU: torch.Tensor       # [nR,nR,L_RU]
    rho_UR: torch.Tensor       # [nRIS,nRIS,L_RU]
    rho_RB: torch.Tensor       # [nRIS,nRIS,L_BR]
    rho_BR: torch.Tensor       # [nT,nT,L_BR]
    rho_RUhop: torch.Tensor    # [nRIS,nRIS], INCLUDES sigma2RU factor
    muBR: torch.Tensor         # [nRIS,nT]
    muRU: torch.Tensor         # [nR,nRIS]
    sigma2BR: torch.Tensor     # scalar real
    sigma2RU: torch.Tensor     # scalar real
    eKappaBR: torch.Tensor     # scalar real
    eKappaRU: torch.Tensor     # scalar real
    sRIS: torch.Tensor
    sT: torch.Tensor
    sR: torch.Tensor


@dataclass
class WState:
    env: StaticEnvironment
    w: torch.Tensor            # [nT]
    ubarBR: torch.Tensor       # [nRIS]
    UBR: torch.Tensor          # [nRIS,nRIS]
    # Matrices for sigma2Feff:
    eff_moment_kernel: torch.Tensor  # [nR,nRIS,nRIS]
    # Matrices for Cmat:
    cov_kernel: torch.Tensor         # [nR,nR,nRIS,nRIS]


def build_static_environment(
    *,
    rho_RU,
    rho_UR,
    rho_RB,
    rho_BR,
    rho_RUhop,
    mu_XPR_BR,
    sigma_XPR_BR,
    mu_XPR_RU,
    sigma_XPR_RU,
    muBR,
    sigma2BR,
    muRU,
    sigma2RU,
    device=None,
    parity: bool = False,
) -> StaticEnvironment:
    dev = _device(device)
    cd = _cdtype(parity)
    rd = _rdtype(parity)

    rho_RU = _ctensor(rho_RU, device=dev, parity=parity)
    rho_UR = _ctensor(rho_UR, device=dev, parity=parity)
    rho_RB = _ctensor(rho_RB, device=dev, parity=parity)
    rho_BR = _ctensor(rho_BR, device=dev, parity=parity)
    rho_RUhop = _ctensor(rho_RUhop, device=dev, parity=parity)
    muBR = _ctensor(muBR, device=dev, parity=parity)
    muRU = _ctensor(muRU, device=dev, parity=parity)
    sigma2BR = _rtensor(sigma2BR, device=dev, parity=parity).reshape(())
    sigma2RU = _rtensor(sigma2RU, device=dev, parity=parity).reshape(())

    nRIS, nT = muBR.shape
    nR, nRIS2 = muRU.shape
    if nRIS2 != nRIS:
        raise ValueError("muBR/muRU nRIS mismatch")

    ebr = _xpr_inverse_mean(mu_XPR_BR, sigma_XPR_BR, device=dev, parity=parity)
    eru = _xpr_inverse_mean(mu_XPR_RU, sigma_XPR_RU, device=dev, parity=parity)

    return StaticEnvironment(
        rho_RU=rho_RU,
        rho_UR=rho_UR,
        rho_RB=rho_RB,
        rho_BR=rho_BR,
        rho_RUhop=rho_RUhop,
        muBR=muBR,
        muRU=muRU,
        sigma2BR=sigma2BR,
        sigma2RU=sigma2RU,
        eKappaBR=ebr,
        eKappaRU=eru,
        sRIS=_pol_sign(nRIS, device=dev, dtype=rd),
        sT=_pol_sign(nT, device=dev, dtype=rd),
        sR=_pol_sign(nR, device=dev, dtype=rd),
    )


@torch.inference_mode()
def prepare_w_state(env: StaticEnvironment, w) -> WState:
    """
    Compute everything that is independent of gamma.

    This is intended to run ONCE per (bank,W). A Critic/Actor candidate pool
    can then score hundreds of gamma patterns without rebuilding UBR or RU terms.
    """
    dev = env.muBR.device
    cd = env.muBR.dtype

    w = torch.as_tensor(w, dtype=cd, device=dev).reshape(-1)
    nRIS, nT = env.muBR.shape
    nR = env.muRU.shape[0]
    LBR = env.rho_RB.shape[2]
    LRU = env.rho_RU.shape[2]

    if w.numel() != nT:
        raise ValueError(f"w length {w.numel()} != nT {nT}")

    SRIS = env.sRIS[:, None] * env.sRIS[None, :]
    ST = env.sT[:, None] * env.sT[None, :]

    c0BR = env.sigma2BR / (2.0 * (1.0 + env.eKappaBR) * LBR)
    c0RU = env.sigma2RU / (2.0 * (1.0 + env.eKappaRU) * LRU)

    # MATLAB: ubarBR = mu_gnb2ris*w
    ubar = env.muBR @ w

    U_nlos = torch.zeros((nRIS, nRIS), dtype=cd, device=dev)

    for ell in range(LBR):
        Grx = env.rho_RB[:, :, ell]
        Gtx = env.rho_BR[:, :, ell]

        # MATLAB uses w.' * Gtx * conj(w), not w' * Gtx * w.
        s1 = torch.sum((w @ Gtx) * w.conj())
        s2 = torch.sum((w @ (Gtx * ST)) * w.conj())

        GrxPol = Grx * SRIS
        BRdirect = Grx + env.eKappaBR * GrxPol
        BRcross = env.eKappaBR * Grx + GrxPol

        U_nlos = U_nlos + c0BR * (BRdirect * s1 + BRcross * s2)

    UBR = _hermitianize(
        ubar[:, None] * ubar.conj()[None, :] + U_nlos
    )

    # generate_eff_moments kernels:
    eff_kernel = torch.empty(
        (nR, nRIS, nRIS), dtype=cd, device=dev
    )
    for r in range(nR):
        mg = env.muRU[r, :]
        ARU = _hermitianize(
            mg[:, None] * mg.conj()[None, :] + env.rho_RUhop
        )
        eff_kernel[r] = ARU * UBR

    # evaluate_gamma_metric Cmat kernels:
    cov_kernel = torch.empty(
        (nR, nR, nRIS, nRIS), dtype=cd, device=dev
    )

    for r in range(nR):
        mgr = env.muRU[r, :]
        for rp in range(nR):
            mgrp = env.muRU[rp, :]
            MG = mgr[:, None] * mgrp.conj()[None, :]

            U_nlos_RU = torch.zeros(
                (nRIS, nRIS), dtype=cd, device=dev
            )
            pol_sign = env.sR[r] * env.sR[rp]

            for ell in range(LRU):
                rhoRx = env.rho_RU[r, rp, ell]
                Gt = env.rho_UR[:, :, ell]
                GtPol = Gt * SRIS

                RUdirect = Gt + env.eKappaRU * GtPol
                RUcross = env.eKappaRU * Gt + GtPol

                U_nlos_RU = U_nlos_RU + (
                    c0RU * rhoRx * (RUdirect + pol_sign * RUcross)
                )

            URU = MG + U_nlos_RU
            cov_kernel[r, rp] = URU * UBR

    return WState(
        env=env,
        w=w,
        ubarBR=ubar,
        UBR=UBR,
        eff_moment_kernel=eff_kernel,
        cov_kernel=cov_kernel,
    )


def _quadratic_no_left_conj(gamma, K):
    """
    Compute gamma^T K conj(gamma) for a candidate batch.

    gamma: [C,N]
    K:
      [N,N] or [A,N,N]

    returns:
      [C] or [C,A]
    """
    if K.ndim == 2:
        y = gamma @ K
        return torch.sum(y * gamma.conj(), dim=-1)

    if K.ndim == 3:
        # [A,N,N] @ [C,N] without materializing gamma outer products.
        # y[a,c,n] = sum_i gamma[c,i] K[a,i,n]
        y = torch.einsum("ci,ain->acn", gamma, K)
        q = torch.sum(y * gamma.conj()[None, :, :], dim=-1)
        return q.transpose(0, 1)  # [C,A]

    raise ValueError("K must be rank 2 or 3")


@torch.inference_mode()
def evaluate_gamma_batch(
    state: WState,
    gamma,
    *,
    candidate_chunk: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate C gamma patterns.

    gamma:
        [nRIS] or [C,nRIS]

    Returns:
        muFeff    [C,nR]
        sigma2Feff[C,nR]
        muSNR     [C]
        Cmat      [C,nR,nR]
        sigma2Wick[C]

    IMPORTANT:
        candidate batching does not create [C,nRIS,nRIS].
    """
    dev = state.UBR.device
    cd = state.UBR.dtype
    rd = state.env.sigma2BR.dtype

    g = torch.as_tensor(gamma, dtype=cd, device=dev)
    if g.ndim == 1:
        g = g.unsqueeze(0)
    if g.ndim != 2 or g.shape[1] != state.ubarBR.numel():
        raise ValueError(
            f"gamma must be [C,{state.ubarBR.numel()}], got {tuple(g.shape)}"
        )

    C = g.shape[0]
    if candidate_chunk is None:
        candidate_chunk = C

    outputs = {
        "muFeff": [],
        "sigma2Feff": [],
        "muSNR": [],
        "Cmat": [],
        "sigma2Wick": [],
    }

    nR = state.env.muRU.shape[0]
    Kcov_flat = state.cov_kernel.reshape(
        nR*nR,
        state.ubarBR.numel(),
        state.ubarBR.numel(),
    )

    for c0 in range(0, C, candidate_chunk):
        c1 = min(c0 + candidate_chunk, C)
        gc = g[c0:c1]

        # MATLAB:
        # muLayer = muRU * (gamma .* ubarBR)
        mu = (gc * state.ubarBR[None, :]) @ state.env.muRU.T
        # shape [Cc,nR], no conjugation, matching MATLAB multiplication.

        second_eff = _quadratic_no_left_conj(
            gc, state.eff_moment_kernel
        ).real

        sigma2 = torch.clamp(
            second_eff - torch.abs(mu) ** 2,
            min=0.0
        ).to(rd)

        muSNR = torch.sum(
            sigma2 + torch.abs(mu) ** 2,
            dim=1
        ).real

        second_cov = _quadratic_no_left_conj(
            gc, Kcov_flat
        ).reshape(-1, nR, nR)

        Cmat = (
            second_cov
            - mu[:, :, None] * mu.conj()[:, None, :]
        )
        Cmat = _hermitianize(Cmat)

        wick1 = torch.sum(torch.abs(Cmat) ** 2, dim=(1,2))

        # MATLAB: mu' * Cmat * mu
        wick2 = torch.einsum(
            "cr,crs,cs->c",
            mu.conj(), Cmat, mu
        )

        wick = torch.clamp(
            (wick1 + 2.0 * wick2.real),
            min=0.0
        ).to(rd)

        outputs["muFeff"].append(mu)
        outputs["sigma2Feff"].append(sigma2)
        outputs["muSNR"].append(muSNR)
        outputs["Cmat"].append(Cmat)
        outputs["sigma2Wick"].append(wick)

    return {
        k: torch.cat(v, dim=0)
        for k, v in outputs.items()
    }


def _rel_fro(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()), np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel()) / den)


def compare_stage3_matlab_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
) -> Dict[str, float]:
    from scipy.io import loadmat

    M = loadmat(mat_path, squeeze_me=True)
    dev = _device(device)

    def scalar(name):
        return float(np.asarray(M[name]).reshape(()))

    env = build_static_environment(
        rho_RU=M["rhoRU"],
        rho_UR=M["rhoUR"],
        rho_RB=M["rhoRB"],
        rho_BR=M["rhoBR"],
        rho_RUhop=M["rhoRUhop"],
        mu_XPR_BR=scalar("muXPR_BR"),
        sigma_XPR_BR=scalar("sigmaXPR_BR"),
        mu_XPR_RU=scalar("muXPR_RU"),
        sigma_XPR_RU=scalar("sigmaXPR_RU"),
        muBR=M["muBR"],
        sigma2BR=scalar("sigma2BR"),
        muRU=M["muRU"],
        sigma2RU=scalar("sigma2RU"),
        device=dev,
        parity=parity,
    )

    state = prepare_w_state(env, M["W"])
    out = evaluate_gamma_batch(state, M["gamma"])

    py = {
        "UBR": state.UBR.detach().cpu().numpy(),
        "muFeff": out["muFeff"][0].detach().cpu().numpy(),
        "sigma2Feff": out["sigma2Feff"][0].detach().cpu().numpy(),
        "muSNR": float(out["muSNR"][0].detach().cpu()),
        "Cmat": out["Cmat"][0].detach().cpu().numpy(),
        "sigma2Wick": float(out["sigma2Wick"][0].detach().cpu()),
    }

    metrics = {}

    for name in ("UBR","muFeff","sigma2Feff","Cmat"):
        ref = np.asarray(M[name]).squeeze()
        got = np.asarray(py[name]).squeeze()
        metrics[f"{name}_relFro"] = _rel_fro(got, ref)
        metrics[f"{name}_maxAbs"] = float(np.max(np.abs(got-ref)))

    for name in ("muSNR","sigma2Wick"):
        ref = scalar(name)
        got = float(py[name])
        metrics[f"{name}_abs"] = abs(got-ref)
        metrics[f"{name}_rel"] = abs(got-ref) / max(abs(ref), np.finfo(float).eps)

    return metrics


@torch.inference_mode()
def benchmark_gamma_candidates(
    state: WState,
    *,
    n_candidates: int = 512,
    repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmark only gamma-dependent candidate scoring.
    prepare_w_state is intentionally excluded because it runs once per (bank,W).
    """
    dev = state.UBR.device
    rd = state.env.sigma2BR.dtype
    cd = state.UBR.dtype
    nRIS = state.ubarBR.numel()

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    # Generic unit-modulus gamma benchmark. Production RIS amplitude model
    # will be ported separately; candidate scoring cost is the same order.
    phase = 2.0 * math.pi * torch.rand(
        (n_candidates,nRIS),
        generator=gen,
        device=dev,
        dtype=rd,
    )
    gamma = torch.exp(1j*phase).to(cd)

    _ = evaluate_gamma_batch(state, gamma)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = evaluate_gamma_batch(state, gamma)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        ts.append(time.perf_counter()-t0)

    med = float(np.median(ts))
    return {
        "device": str(dev),
        "nRIS": int(nRIS),
        "nR": int(state.env.muRU.shape[0]),
        "n_candidates": int(n_candidates),
        "median_seconds": med,
        "candidates_per_second": float(n_candidates/med),
    }
