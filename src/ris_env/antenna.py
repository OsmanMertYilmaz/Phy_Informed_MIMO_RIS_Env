"""
Stage 1 GPU physics parity layer for the RIS project.

Scope:
    MATLAB generate_channel_moments_no_cdl
        -> muH
        -> sigma2H
        -> dbarT
        -> dbarR

Assumptions locked to the current dataset configuration:
    - array.Size = [M N 2 1 1]
    - single panel
    - dV = dH = 0.5 lambda
    - polarization angles = [+45, -45] deg
    - isotropic element
    - PolarizationModel = Model-2
    - Tx/Rx array orientations = [0,0,0]
    - array broadside = +x, matching MathWorks Rup transform
    - MATLAB linear ordering: M fastest, then N, then polarization P

This module has:
    1) exact/parity float64 path
    2) production float32/complex64 path
    3) batched GPU evaluation for many same-shape links
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import math
import time

import numpy as np
import torch


@dataclass(frozen=True)
class ArraySpec:
    M: int
    N: int
    P: int = 2
    dV_lambda: float = 0.5
    dH_lambda: float = 0.5

    def validate(self) -> None:
        if self.M <= 0 or self.N <= 0:
            raise ValueError("M and N must be positive.")
        if self.P != 2:
            raise ValueError("Stage-1 implementation is locked to P=2 dual polarization.")
        if self.dV_lambda != 0.5 or self.dH_lambda != 0.5:
            raise ValueError("Stage-1 implementation is locked to 0.5-lambda spacing.")


def _real_dtype(parity: bool) -> torch.dtype:
    return torch.float64 if parity else torch.float32


def _complex_dtype(parity: bool) -> torch.dtype:
    return torch.complex128 if parity else torch.complex64


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_dualpol_positions_lambda(
    spec: ArraySpec,
    *,
    device: str | torch.device | None = None,
    parity: bool = True,
) -> torch.Tensor:
    """
    Reproduce makeAntennaArray(...).ElementPositions for:
        Size=[M N 2 1 1], orientation=[0,0,0], single panel.

    Returns:
        positions_lambda: [3, 2*M*N] in wavelength units.

    MATLAB ordering:
        m fastest, then n, then polarization.
        First M*N ports are +45 deg, second M*N ports are -45 deg.

    MathWorks Rup:
        [x_local, y_local, 0] -> [0, y_local, -x_local].
    """
    spec.validate()
    dev = _as_device(device)
    dtype = _real_dtype(parity)

    v = (
        torch.arange(spec.M, dtype=dtype, device=dev)
        - (spec.M - 1) / 2.0
    ) * spec.dV_lambda

    h = (
        torch.arange(spec.N, dtype=dtype, device=dev)
        - (spec.N - 1) / 2.0
    ) * spec.dH_lambda

    # MATLAB linear order for M x N:
    # m changes fastest.
    v_one_pol = v.repeat(spec.N)
    h_one_pol = torch.repeat_interleave(h, spec.M)

    # P dimension comes after M,N -> duplicate whole M*N block per polarization.
    v_all = v_one_pol.repeat(spec.P)
    h_all = h_one_pol.repeat(spec.P)

    x = torch.zeros_like(v_all)
    y = h_all
    z = -v_all

    return torch.stack((x, y, z), dim=0)


def _normalize_direction(v: torch.Tensor) -> torch.Tensor:
    """
    cart2sph -> wrapped az/zenith -> getRhoHat is mathematically v/||v||.
    Input: [B,3]
    Output: [B,3]
    """
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    if torch.any(norm <= 0):
        raise ValueError("Direction vector must be nonzero.")
    return v / norm


def _location_terms(
    direction: torch.Tensor,
    positions_lambda: torch.Tensor,
    *,
    parity: bool,
) -> torch.Tensor:
    """
    Because dbar = positions_lambda * lambda0 and MATLAB divides by lambda0
    in getLocationTerm, lambda cancels exactly:
        exp(j 2pi rho^T dbar / lambda0)
      = exp(j 2pi rho^T positions_lambda)

    direction: [B,3]
    positions_lambda: [3,S]
    returns: [B,S] complex
    """
    phase = 2.0 * math.pi * (direction @ positions_lambda)
    return torch.exp(1j * phase).to(_complex_dtype(parity))


def _matlab_mean_field_scalar(
    U: int,
    S: int,
    *,
    device: torch.device,
    parity: bool,
) -> torch.Tensor:
    """
    Exact scalar convention used by the provided MATLAB
    generate_channel_moments_no_cdl / calc_mu:

        fieldScalar =
          rxField(1,1)*txField(1,1)
          - polarizationSign*rxField(2,1)*txField(2,1)

    For isotropic Model-2, orientation=0 and first (+45 deg) port:
        rxField(:,1) = txField(:,1) = [cos45; sin45].

    polarizationSign:
        -1 for cross polarization blocks
        +1 otherwise.

    This intentionally reproduces the existing dataset convention.
    """
    dtype = _real_dtype(parity)

    u_idx = torch.arange(U, device=device)
    s_idx = torch.arange(S, device=device)

    u_second = u_idx >= (U // 2)
    s_second = s_idx >= (S // 2)

    cross = torch.logical_xor(
        u_second[:, None],
        s_second[None, :]
    )

    pol_sign = torch.where(
        cross,
        torch.tensor(-1.0, dtype=dtype, device=device),
        torch.tensor(+1.0, dtype=dtype, device=device),
    )

    c45 = torch.tensor(math.sqrt(0.5), dtype=dtype, device=device)
    first_term = c45 * c45
    second_term = c45 * c45

    return first_term - pol_sign * second_term


@torch.inference_mode()
def generate_channel_moments_batch(
    *,
    tx_spec: ArraySpec,
    rx_spec: ArraySpec,
    a_vectors,
    d_vectors,
    carrier_frequency,
    K,
    mu_xpr,
    sigma_xpr,
    c0: float = 299792458.0,
    device: str | torch.device | None = None,
    parity: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Batched GPU implementation of generate_channel_moments_no_cdl.

    Same tx/rx array shape for the whole batch.
    This is deliberate: the future dataset generator will bucket banks by
    array shape / nRIS and evaluate each bucket as a GPU batch.

    Inputs:
        a_vectors         [B,3]   MATLAB aVector  (arrival-side direction vector)
        d_vectors         [B,3]   MATLAB dVector  (departure-side direction vector)
        carrier_frequency [B] or scalar
        K                 [B] or scalar, linear K
        mu_xpr            [B] or scalar
        sigma_xpr         [B] or scalar

    Returns:
        muH     [B,U,S]
        sigma2H [B]
        dbarT   [B,3,S] meters
        dbarR   [B,3,U] meters
        txLocation [B,S]
        rxLocation [B,U]
    """
    tx_spec.validate()
    rx_spec.validate()

    dev = _as_device(device)
    rdtype = _real_dtype(parity)
    cdtype = _complex_dtype(parity)

    a = torch.as_tensor(a_vectors, dtype=rdtype, device=dev)
    d = torch.as_tensor(d_vectors, dtype=rdtype, device=dev)

    if a.ndim == 1:
        a = a.unsqueeze(0)
    if d.ndim == 1:
        d = d.unsqueeze(0)

    if a.shape != d.shape or a.shape[-1] != 3:
        raise ValueError(f"a_vectors/d_vectors must both be [B,3], got {a.shape}, {d.shape}")

    B = a.shape[0]

    def batch_scalar(x, name):
        t = torch.as_tensor(x, dtype=rdtype, device=dev).reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)
        if t.numel() != B:
            raise ValueError(f"{name} must be scalar or length B={B}.")
        return t

    fc = batch_scalar(carrier_frequency, "carrier_frequency")
    Kt = batch_scalar(K, "K")
    mux = batch_scalar(mu_xpr, "mu_xpr")
    sigx = batch_scalar(sigma_xpr, "sigma_xpr")

    if torch.any(fc <= 0):
        raise ValueError("carrier_frequency must be positive.")
    if torch.any(Kt < 0):
        raise ValueError("K must be nonnegative.")

    tx_pos_lambda = build_dualpol_positions_lambda(
        tx_spec, device=dev, parity=parity
    )
    rx_pos_lambda = build_dualpol_positions_lambda(
        rx_spec, device=dev, parity=parity
    )

    S = tx_pos_lambda.shape[1]
    U = rx_pos_lambda.shape[1]

    # Physical positions in meters.
    lambda0 = (torch.tensor(c0, dtype=rdtype, device=dev) / fc)
    dbarT = tx_pos_lambda.unsqueeze(0) * lambda0[:, None, None]
    dbarR = rx_pos_lambda.unsqueeze(0) * lambda0[:, None, None]

    # MATLAB semantics:
    # aVector -> AOA -> rx direction
    # dVector -> AOD -> tx direction
    rx_direction = _normalize_direction(a)
    tx_direction = _normalize_direction(d)

    tx_location = _location_terms(
        tx_direction, tx_pos_lambda, parity=parity
    )
    rx_location = _location_terms(
        rx_direction, rx_pos_lambda, parity=parity
    )

    field_scalar = _matlab_mean_field_scalar(
        U, S, device=dev, parity=parity
    ).to(cdtype)

    los_scale = torch.sqrt(Kt / (Kt + 1.0)).to(rdtype)

    location_matrix = (
        rx_location[:, :, None] * tx_location[:, None, :]
    ).to(cdtype)

    muH = (
        los_scale[:, None, None].to(cdtype)
        * field_scalar[None, :, :]
        * location_matrix
    )

    # Same closed-form as MATLAB.
    ln10 = torch.tensor(math.log(10.0), dtype=rdtype, device=dev)
    e_inverse_xpr = torch.exp(
        -ln10 * mux / 10.0
        + (ln10 * ln10) * (sigx * sigx) / 200.0
    )

    sigma2H = (
        torch.tensor(0.5, dtype=rdtype, device=dev)
        / (Kt + 1.0)
        * (1.0 + e_inverse_xpr)
    )

    return {
        "muH": muH,
        "sigma2H": sigma2H,
        "dbarT": dbarT,
        "dbarR": dbarR,
        "txLocation": tx_location,
        "rxLocation": rx_location,
        "txPositionsLambda": tx_pos_lambda,
        "rxPositionsLambda": rx_pos_lambda,
    }


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()), np.finfo(float).eps)
    return float(np.linalg.norm((a - b).ravel()) / den)


def compare_with_matlab_mat(
    mat_path: str,
    *,
    device: str | torch.device | None = None,
    parity: bool = True,
) -> Dict[str, float]:
    """
    Load a golden .mat exported by export_no_cdl_parity_case.m and compare.
    """
    from scipy.io import loadmat

    M = loadmat(mat_path, squeeze_me=True)

    tx_size = np.asarray(M["txSize"], dtype=int).reshape(-1)
    rx_size = np.asarray(M["rxSize"], dtype=int).reshape(-1)

    tx_spec = ArraySpec(int(tx_size[0]), int(tx_size[1]))
    rx_spec = ArraySpec(int(rx_size[0]), int(rx_size[1]))

    out = generate_channel_moments_batch(
        tx_spec=tx_spec,
        rx_spec=rx_spec,
        a_vectors=np.asarray(M["aVector"], dtype=float).reshape(1, 3),
        d_vectors=np.asarray(M["dVector"], dtype=float).reshape(1, 3),
        carrier_frequency=float(np.asarray(M["fc"]).reshape(())),
        K=float(np.asarray(M["K"]).reshape(())),
        mu_xpr=float(np.asarray(M["muXPR"]).reshape(())),
        sigma_xpr=float(np.asarray(M["sigmaXPR"]).reshape(())),
        c0=float(np.asarray(M["c0"]).reshape(())),
        device=device,
        parity=parity,
    )

    py_mu = out["muH"][0].detach().cpu().numpy()
    py_sig = float(out["sigma2H"][0].detach().cpu())
    py_dt = out["dbarT"][0].detach().cpu().numpy()
    py_dr = out["dbarR"][0].detach().cpu().numpy()

    mt_mu = np.asarray(M["muH"])
    mt_sig = float(np.asarray(M["sigma2H"]).reshape(()))
    mt_dt = np.asarray(M["dbarT"])
    mt_dr = np.asarray(M["dbarR"])

    metrics = {
        "muH_relFro": _rel_fro(py_mu, mt_mu),
        "muH_maxAbs": float(np.max(np.abs(py_mu - mt_mu))),
        "sigma2H_abs": abs(py_sig - mt_sig),
        "sigma2H_rel": abs(py_sig - mt_sig) / max(abs(mt_sig), np.finfo(float).eps),
        "dbarT_relFro": _rel_fro(py_dt, mt_dt),
        "dbarT_maxAbs": float(np.max(np.abs(py_dt - mt_dt))),
        "dbarR_relFro": _rel_fro(py_dr, mt_dr),
        "dbarR_maxAbs": float(np.max(np.abs(py_dr - mt_dr))),
    }

    if "AisoProbe" in M:
        probe = np.asarray(M["AisoProbe"], dtype=float).reshape(-1)
        finite = probe[np.isfinite(probe)]
        if finite.size:
            metrics["Aiso_max_abs_from_1"] = float(np.max(np.abs(finite - 1.0)))

    return metrics


@torch.inference_mode()
def benchmark_same_shape(
    *,
    tx_spec: ArraySpec,
    rx_spec: ArraySpec,
    batch_size: int = 4096,
    repeats: int = 5,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """
    Simple throughput benchmark. Uses production float32/complex64.
    The important metric is links/s for batched evaluation.
    """
    dev = _as_device(device)

    rng = np.random.default_rng(42)
    a = rng.normal(size=(batch_size, 3)).astype(np.float32)
    d = rng.normal(size=(batch_size, 3)).astype(np.float32)
    fc = np.full(batch_size, 3.5e9, np.float32)
    K = np.full(batch_size, 4.0, np.float32)
    mux = np.full(batch_size, 8.0, np.float32)
    sigx = np.full(batch_size, 3.0, np.float32)

    # warmup
    _ = generate_channel_moments_batch(
        tx_spec=tx_spec,
        rx_spec=rx_spec,
        a_vectors=a,
        d_vectors=d,
        carrier_frequency=fc,
        K=K,
        mu_xpr=mux,
        sigma_xpr=sigx,
        device=dev,
        parity=False,
    )

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = generate_channel_moments_batch(
            tx_spec=tx_spec,
            rx_spec=rx_spec,
            a_vectors=a,
            d_vectors=d,
            carrier_frequency=fc,
            K=K,
            mu_xpr=mux,
            sigma_xpr=sigx,
            device=dev,
            parity=False,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)

    med = float(np.median(times))
    return {
        "device": str(dev),
        "batch_size": int(batch_size),
        "median_seconds": med,
        "links_per_second": float(batch_size / med),
        "tx_ports": int(2 * tx_spec.M * tx_spec.N),
        "rx_ports": int(2 * rx_spec.M * rx_spec.N),
    }
