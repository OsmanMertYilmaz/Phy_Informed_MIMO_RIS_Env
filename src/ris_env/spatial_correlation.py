"""
Stage 2 GPU spatial-correlation engine for the RIS project.

Ports the current MATLAB second-order / capped Gauss-Hermite functions:
    - compute_ch_rho_avg
    - compute_ch_eff_rho_avg_fast

Design goals:
    1) MATLAB parity in float64/complex128
    2) production CUDA path in float32/complex64
    3) unique-displacement cache to avoid O(nAnt^2) GH evaluation
    4) batched bank evaluation when shapes match

The current project uses:
    nGH = 20
    L = 20 ray offsets
    azimuth cap = 104 deg
    zenith cap = 52 deg
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math
import numpy as np
import torch


ALPHA = np.array([
     0.0447, -0.0447,  0.1413, -0.1413,  0.2492, -0.2492,
     0.3715, -0.3715,  0.5129, -0.5129,  0.6797, -0.6797,
     0.8844, -0.8844,  1.1481, -1.1481,  1.5195, -1.5195,
     2.1551, -2.1551
], dtype=np.float64)


@dataclass
class DisplacementCache:
    unique_lambda: torch.Tensor   # [D,3], displacement / lambda0
    pair_map: torch.Tensor        # [nAnt,nAnt], long
    n_ant: int


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool) -> torch.dtype:
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool) -> torch.dtype:
    return torch.complex128 if parity else torch.complex64


def _batch_vector(x, B: int, name: str, *, device, dtype):
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(B)
    t = t.reshape(-1)
    if t.numel() == 1:
        return t.expand(B)
    if t.numel() != B:
        raise ValueError(f"{name}: expected scalar or B={B}, got {tuple(t.shape)}")
    return t


def _batch_xyz(x, name: str, *, device, dtype):
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"{name}: expected [B,3], got {tuple(t.shape)}")
    return t


def wrap_azimuth_deg(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + 180.0, 360.0) - 180.0


def wrap_zenith_deg(x: torch.Tensor) -> torch.Tensor:
    y = torch.remainder(x, 360.0)
    return torch.where(y > 180.0, 360.0 - y, y)


def direction_angles_deg(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MATLAB cart2sph + phi=azimuth, theta=90-elevation.
    v: [B,3]
    """
    norm = torch.linalg.vector_norm(v, dim=-1)
    if torch.any(norm <= 0):
        raise ValueError("direction vector cannot be zero")
    phi = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0]))
    theta = torch.rad2deg(torch.acos(torch.clamp(v[:, 2] / norm, -1.0, 1.0)))
    return wrap_azimuth_deg(phi), wrap_zenith_deg(theta)


def gauss_hermite_rule(n: int, *, device, dtype):
    # Physicists' Hermite rule: integral exp(-x^2) f(x) dx.
    x, w = np.polynomial.hermite.hermgauss(n)
    return (
        torch.as_tensor(x, dtype=dtype, device=device),
        torch.as_tensor(w, dtype=dtype, device=device),
    )


def build_displacement_cache_from_dbar(
    dbar,
    lambda0: float,
    *,
    device=None,
    parity: bool = True,
) -> DisplacementCache:
    """
    Reproduce MATLAB buildUniqueDisplacements* semantics.

    Matrix entry (i,j) uses:
        dbar[:,i] - dbar[:,j]

    We normalize by lambda0 immediately. This removes carrier frequency
    from the downstream phase algebra because k0*d = 2*pi*(d/lambda0).
    """
    dev = _device(device)
    rd = _rdtype(parity)

    d = np.asarray(dbar, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != 3:
        raise ValueError("dbar must have shape [3,nAnt]")

    n = d.shape[1]
    disp = d.T[:, None, :] - d.T[None, :, :]  # [n,n,3]
    disp_lambda = disp / float(lambda0)

    # MATLAB tolerance = max(lambda0*1e-10,1e-12) in meters.
    # In lambda-normalized coordinates:
    tol_lambda = max(float(lambda0) * 1e-10, 1e-12) / float(lambda0)
    key = np.rint(disp_lambda / tol_lambda).astype(np.int64)

    flat_key = key.reshape(-1, 3)
    _, first, inverse = np.unique(
        flat_key,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    unique_lambda = disp_lambda.reshape(-1, 3)[first]

    return DisplacementCache(
        unique_lambda=torch.as_tensor(unique_lambda, dtype=rd, device=dev),
        pair_map=torch.as_tensor(inverse.reshape(n, n), dtype=torch.long, device=dev),
        n_ant=n,
    )


def _capped_variance_nodes(
    mu_log10,
    sig_log10,
    cap_deg: float,
    xgh: torch.Tensor,
    wgh: torch.Tensor,
):
    """
    Batched equivalent of cappedVarianceNodes*.

    We intentionally do not merge identical capped tail nodes on GPU.
    Their weights are still summed by the final quadrature reduction, so
    the mathematical result is unchanged.
    """
    ln10 = math.log(10.0)
    m = 2.0 * ln10 * mu_log10[:, None]
    v = 2.0 * ln10 * sig_log10[:, None]

    log_spread_sq = m + math.sqrt(2.0) * v * xgh[None, :]
    spread_sq = torch.minimum(
        torch.exp(log_spread_sq),
        torch.tensor(cap_deg * cap_deg, dtype=xgh.dtype, device=xgh.device),
    )

    kappa = (math.pi / 180.0 / 7.0) ** 2
    var = kappa * spread_sq
    weight = (wgh / math.sqrt(math.pi))[None, :].expand_as(var)
    return var, weight


def _second_order_corr_batch(
    cache: DisplacementCache,
    phi_deg: torch.Tensor,
    theta_deg: torch.Tensor,
    mu_az: torch.Tensor,
    sig_az: torch.Tensor,
    mu_zn: torch.Tensor,
    sig_zn: torch.Tensor,
    *,
    n_gh: int,
    parity: bool,
    gh_pair_chunk: int = 80,
) -> torch.Tensor:
    """
    Evaluate one cluster center for B banks on unique displacements.

    Returns [B,D] complex.
    Uses GH-pair chunking to bound temporary GPU memory.
    """
    dev = cache.unique_lambda.device
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    B = phi_deg.numel()
    disp = cache.unique_lambda.to(rd)  # [D,3]
    D = disp.shape[0]

    ph = torch.deg2rad(phi_deg)
    th = torch.deg2rad(theta_deg)

    sphi, cphi = torch.sin(ph), torch.cos(ph)
    sth, cth = torch.sin(th), torch.cos(th)

    r0 = torch.stack((sth*cphi, sth*sphi, cth), dim=1)
    rtheta = torch.stack((cth*cphi, cth*sphi, -sth), dim=1)
    rphi = torch.stack((-sth*sphi, sth*cphi, torch.zeros_like(sth)), dim=1)
    rtt = -r0
    rtp = torch.stack((-cth*sphi, cth*cphi, torch.zeros_like(sth)), dim=1)
    rpp = torch.stack((-sth*cphi, -sth*sphi, torch.zeros_like(sth)), dim=1)

    # k0 * displacement_m = 2*pi * displacement_lambda.
    two_pi = 2.0 * math.pi
    phase = two_pi * (r0 @ disp.T)
    a_zn = two_pi * (rtheta @ disp.T)
    a_az = two_pi * (rphi @ disp.T)
    b_zz = two_pi * (rtt @ disp.T)
    b_za = two_pi * (rtp @ disp.T)
    b_aa = two_pi * (rpp @ disp.T)

    xgh, wgh = gauss_hermite_rule(n_gh, device=dev, dtype=rd)
    var_zn, w_zn = _capped_variance_nodes(mu_zn, sig_zn, 52.0, xgh, wgh)
    var_az, w_az = _capped_variance_nodes(mu_az, sig_az, 104.0, xgh, wgh)

    # Flatten GH pairs [G*G] and process in chunks.
    G = n_gh
    sz = var_zn[:, :, None].expand(B, G, G).reshape(B, G*G)
    sa = var_az[:, None, :].expand(B, G, G).reshape(B, G*G)
    ww = (
        w_zn[:, :, None] * w_az[:, None, :]
    ).reshape(B, G*G)

    base = torch.exp(1j * phase).to(cd)
    result = torch.zeros((B, D), dtype=cd, device=dev)

    for q0 in range(0, G*G, gh_pair_chunk):
        q1 = min(q0 + gh_pair_chunk, G*G)

        sz2 = sz[:, q0:q1, None]
        sa2 = sa[:, q0:q1, None]
        weight = ww[:, q0:q1, None]

        bzz = b_zz[:, None, :]
        bza = b_za[:, None, :]
        baa = b_aa[:, None, :]
        azn = a_zn[:, None, :]
        aaz = a_az[:, None, :]

        r11 = 1.0 - 1j * sz2 * bzz
        r12 =      - 1j * sz2 * bza
        r21 =      - 1j * sa2 * bza
        r22 = 1.0 - 1j * sa2 * baa

        det = r11 * r22 - r12 * r21

        c11 =  r22 * sz2 / det
        c12 = -r12 * sa2 / det
        c21 = -r21 * sz2 / det
        c22 =  r11 * sa2 / det

        quad = (
            azn*azn*c11
            + azn*aaz*(c12+c21)
            + aaz*aaz*c22
        )

        conditional = (
            base[:, None, :]
            * torch.exp(-0.5 * quad)
            / torch.sqrt(det)
        )

        result += torch.sum(weight * conditional, dim=1)

    return result


def _reconstruct(unique_values: torch.Tensor, cache: DisplacementCache) -> torch.Tensor:
    # unique_values [B,D] -> [B,n,n]
    return unique_values[:, cache.pair_map]


def _hermitianize_and_set_diag(x: torch.Tensor) -> torch.Tensor:
    y = 0.5 * (x + x.transpose(-1, -2).conj())
    n = y.shape[-1]
    idx = torch.arange(n, device=y.device)
    y[:, idx, idx] = 1.0 + 0.0j
    return y


@torch.inference_mode()
def compute_ch_rho_avg_batch(
    *,
    cache: DisplacementCache,
    vec,
    mu_lg_az,
    sig_lg_az,
    mu_lg_zn,
    sig_lg_zn,
    c_az,
    c_zn_scale,
    mu_offset_zn,
    n_gh: int = 20,
    parity: bool = False,
    gh_pair_chunk: int = 80,
) -> torch.Tensor:
    """
    Batched GPU port of MATLAB compute_ch_rho_avg.

    Returns [B,nAnt,nAnt].
    """
    dev = cache.unique_lambda.device
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    v = _batch_xyz(vec, "vec", device=dev, dtype=rd)
    B = v.shape[0]

    muaz = _batch_vector(mu_lg_az, B, "mu_lg_az", device=dev, dtype=rd)
    sigaz = _batch_vector(sig_lg_az, B, "sig_lg_az", device=dev, dtype=rd)
    muzn = _batch_vector(mu_lg_zn, B, "mu_lg_zn", device=dev, dtype=rd)
    sigzn = _batch_vector(sig_lg_zn, B, "sig_lg_zn", device=dev, dtype=rd)
    caz = _batch_vector(c_az, B, "c_az", device=dev, dtype=rd)
    czn = _batch_vector(c_zn_scale, B, "c_zn_scale", device=dev, dtype=rd)
    off = _batch_vector(mu_offset_zn, B, "mu_offset_zn", device=dev, dtype=rd)

    phi0, theta0 = direction_angles_deg(v)

    alpha = torch.as_tensor(ALPHA, dtype=rd, device=dev)
    accum = torch.zeros(
        (B, cache.unique_lambda.shape[0]),
        dtype=cd,
        device=dev,
    )

    for ell in range(alpha.numel()):
        phi = wrap_azimuth_deg(phi0 + caz * alpha[ell])
        theta = wrap_zenith_deg(theta0 + off + czn * alpha[ell])

        accum += _second_order_corr_batch(
            cache, phi, theta,
            muaz, sigaz, muzn, sigzn,
            n_gh=n_gh, parity=parity,
            gh_pair_chunk=gh_pair_chunk,
        )

    accum /= float(alpha.numel())
    rho = _reconstruct(accum, cache)

    n = cache.n_ant
    if n % 2:
        raise ValueError("dual-pol array requires even nAnt")

    sign = torch.cat((
        torch.ones(n//2, dtype=rd, device=dev),
        -torch.ones(n//2, dtype=rd, device=dev),
    ))
    same_pol = 0.5 * (sign[:, None] * sign[None, :] + 1.0)

    rho = rho * same_pol[None, :, :]
    return _hermitianize_and_set_diag(rho)


@torch.inference_mode()
def compute_ch_eff_rho_avg_batch(
    *,
    cache_tx: DisplacementCache,
    cache_rx: DisplacementCache,
    arrival_vector,
    departure_vector,
    mu_ASA, sig_ASA,
    mu_ZSA, sig_ZSA,
    mu_ASD, sig_ASD,
    mu_ZSD, sig_ZSD,
    c_ASA, c_ZSA,
    c_ASD, c_ZSD,
    mu_offset_ZOD,
    n_gh: int = 20,
    parity: bool = False,
    gh_pair_chunk: int = 80,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched GPU port of MATLAB compute_ch_eff_rho_avg_fast.

    Returns:
        rhoRB [B,nRx,nRx,L]
        rhoBR [B,nTx,nTx,L]
    """
    if cache_tx.unique_lambda.device != cache_rx.unique_lambda.device:
        raise ValueError("Tx/Rx caches must live on the same device")

    dev = cache_tx.unique_lambda.device
    rd = _rdtype(parity)

    av = _batch_xyz(arrival_vector, "arrival_vector", device=dev, dtype=rd)
    dv = _batch_xyz(departure_vector, "departure_vector", device=dev, dtype=rd)
    if av.shape[0] != dv.shape[0]:
        raise ValueError("arrival/departure batch mismatch")
    B = av.shape[0]

    def bv(x, name):
        return _batch_vector(x, B, name, device=dev, dtype=rd)

    muASA, sigASA = bv(mu_ASA,"mu_ASA"), bv(sig_ASA,"sig_ASA")
    muZSA, sigZSA = bv(mu_ZSA,"mu_ZSA"), bv(sig_ZSA,"sig_ZSA")
    muASD, sigASD = bv(mu_ASD,"mu_ASD"), bv(sig_ASD,"sig_ASD")
    muZSD, sigZSD = bv(mu_ZSD,"mu_ZSD"), bv(sig_ZSD,"sig_ZSD")
    cASA, cZSA = bv(c_ASA,"c_ASA"), bv(c_ZSA,"c_ZSA")
    cASD, cZSD = bv(c_ASD,"c_ASD"), bv(c_ZSD,"c_ZSD")
    off = bv(mu_offset_ZOD,"mu_offset_ZOD")

    phiAOA, thetaZOA = direction_angles_deg(av)
    phiAOD, thetaZOD = direction_angles_deg(dv)

    alpha = torch.as_tensor(ALPHA, dtype=rd, device=dev)
    rx_mats = []
    tx_mats = []

    for ell in range(alpha.numel()):
        phi_r = wrap_azimuth_deg(phiAOA + cASA * alpha[ell])
        theta_r = wrap_zenith_deg(thetaZOA + cZSA * alpha[ell])

        val_r = _second_order_corr_batch(
            cache_rx, phi_r, theta_r,
            muASA, sigASA, muZSA, sigZSA,
            n_gh=n_gh, parity=parity,
            gh_pair_chunk=gh_pair_chunk,
        )
        mat_r = _hermitianize_and_set_diag(_reconstruct(val_r, cache_rx))
        rx_mats.append(mat_r)

        phi_t = wrap_azimuth_deg(phiAOD + cASD * alpha[ell])
        theta_t = wrap_zenith_deg(thetaZOD + off + cZSD * alpha[ell])

        val_t = _second_order_corr_batch(
            cache_tx, phi_t, theta_t,
            muASD, sigASD, muZSD, sigZSD,
            n_gh=n_gh, parity=parity,
            gh_pair_chunk=gh_pair_chunk,
        )
        mat_t = _hermitianize_and_set_diag(_reconstruct(val_t, cache_tx))
        tx_mats.append(mat_t)

    rhoRB = torch.stack(rx_mats, dim=-1)
    rhoBR = torch.stack(tx_mats, dim=-1)
    return rhoRB, rhoBR


def _rel_fro(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()), np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel()) / den)


def compare_rho_matlab_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
    gh_pair_chunk: int = 80,
) -> Dict[str, float]:
    """
    Compare one export_rho_parity_case.m result.
    """
    from scipy.io import loadmat

    M = loadmat(mat_path, squeeze_me=True)
    dev = _device(device)

    lambda0 = float(np.asarray(M["lambda0"]).reshape(()))

    cache_t_br = build_displacement_cache_from_dbar(
        M["dbarTBR"], lambda0, device=dev, parity=parity
    )
    cache_r_br = build_displacement_cache_from_dbar(
        M["dbarRBR"], lambda0, device=dev, parity=parity
    )
    cache_t_ru = build_displacement_cache_from_dbar(
        M["dbarTRU"], lambda0, device=dev, parity=parity
    )
    cache_r_ru = build_displacement_cache_from_dbar(
        M["dbarRRU"], lambda0, device=dev, parity=parity
    )

    def sc(name):
        return float(np.asarray(M[name]).reshape(()))

    def vec(name):
        return np.asarray(M[name], dtype=float).reshape(1,3)

    rhoRB, rhoBR = compute_ch_eff_rho_avg_batch(
        cache_tx=cache_t_br,
        cache_rx=cache_r_br,
        arrival_vector=vec("arrivalBR"),
        departure_vector=vec("departureBR"),
        mu_ASA=sc("muASA_BR"), sig_ASA=sc("sigASA_BR"),
        mu_ZSA=sc("muZSA_BR"), sig_ZSA=sc("sigZSA_BR"),
        mu_ASD=sc("muASD_BR"), sig_ASD=sc("sigASD_BR"),
        mu_ZSD=sc("muZSD_BR"), sig_ZSD=sc("sigZSD_BR"),
        c_ASA=sc("cASA_BR"), c_ZSA=sc("cZSA_BR"),
        c_ASD=sc("cASD_BR"), c_ZSD=sc("cZSD_BR"),
        mu_offset_ZOD=sc("muOffsetZOD_BR"),
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRU, rhoUR = compute_ch_eff_rho_avg_batch(
        cache_tx=cache_t_ru,
        cache_rx=cache_r_ru,
        arrival_vector=vec("arrivalRU"),
        departure_vector=vec("departureRU"),
        mu_ASA=sc("muASA_RU"), sig_ASA=sc("sigASA_RU"),
        mu_ZSA=sc("muZSA_RU"), sig_ZSA=sc("sigZSA_RU"),
        mu_ASD=sc("muASD_RU"), sig_ASD=sc("sigASD_RU"),
        mu_ZSD=sc("muZSD_RU"), sig_ZSD=sc("sigZSD_RU"),
        c_ASA=sc("cASA_RU"), c_ZSA=sc("cZSA_RU"),
        c_ASD=sc("cASD_RU"), c_ZSD=sc("cZSD_RU"),
        mu_offset_ZOD=sc("muOffsetZOD_RU"),
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rho_hop = compute_ch_rho_avg_batch(
        cache=cache_t_ru,
        vec=vec("departureRU"),
        mu_lg_az=sc("muASD_RU"),
        sig_lg_az=sc("sigASD_RU"),
        mu_lg_zn=sc("muZSD_RU"),
        sig_lg_zn=sc("sigZSD_RU"),
        c_az=sc("cASD_RU"),
        c_zn_scale=sc("cZSD_RU"),
        mu_offset_zn=sc("muOffsetZOD_RU"),
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    py = {
        "rhoRB": rhoRB[0].detach().cpu().numpy(),
        "rhoBR": rhoBR[0].detach().cpu().numpy(),
        "rhoRU": rhoRU[0].detach().cpu().numpy(),
        "rhoUR": rhoUR[0].detach().cpu().numpy(),
        "rhoRUhopBase": rho_hop[0].detach().cpu().numpy(),
    }

    metrics = {}
    for name in ["rhoRB","rhoBR","rhoRU","rhoUR","rhoRUhopBase"]:
        ref = np.asarray(M[name])
        got = py[name]
        metrics[f"{name}_relFro"] = _rel_fro(got, ref)
        metrics[f"{name}_maxAbs"] = float(np.max(np.abs(got-ref)))

    return metrics
