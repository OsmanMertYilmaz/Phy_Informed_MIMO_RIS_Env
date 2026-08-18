"""
Stage 4 GPU RIS response model.

Official project configuration:
    phase levels = [45, 135] degrees
    beta_min = 0.8
    c = 0.43*pi
    k = 1.6

MATLAB reference:
    beta = (1-beta_min) .* ((sin(phi-c)+1)/2).^k + beta_min;
    gamma = beta .* exp(1j*phi);

The functions below support:
    - one pattern:            z [nRIS]
    - candidate batch:        z [C,nRIS]
    - arbitrary leading dims: z [...,nRIS]

Parity mode:
    float64 / complex128

Production mode:
    float32 / complex64
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from typing import Dict, Any, Tuple
import math
import time
import numpy as np
import torch


BETA_MIN = 0.8
C_PHASE = 0.43 * math.pi
K_EXP = 1.6
PHASE_LEVELS_DEG = (45.0, 135.0)


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool) -> torch.dtype:
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool) -> torch.dtype:
    return torch.complex128 if parity else torch.complex64


@torch.inference_mode()
def z_to_phi(
    z,
    *,
    phase_levels_deg: Tuple[float, float] = PHASE_LEVELS_DEG,
    device=None,
    parity: bool = False,
) -> torch.Tensor:
    """
    Convert binary RIS state z in {0,1} to phase in radians.

    MATLAB equivalent:
        phaseLevels = deg2rad([45 135]);
        phi = phaseLevels(z+1);
    """
    dev = _device(device)
    rd = _rdtype(parity)

    zt = torch.as_tensor(z, device=dev)

    # Strong validation without silently rounding inputs.
    if zt.dtype == torch.bool:
        zi = zt.to(torch.long)
    else:
        if torch.is_floating_point(zt):
            if not torch.all((zt == 0) | (zt == 1)):
                raise ValueError("z must contain only 0/1.")
        else:
            if not torch.all((zt == 0) | (zt == 1)):
                raise ValueError("z must contain only 0/1.")
        zi = zt.to(torch.long)

    levels = torch.tensor(
        phase_levels_deg,
        dtype=rd,
        device=dev,
    ) * (math.pi / 180.0)

    return levels[zi]


@torch.inference_mode()
def generate_ris_response_from_phi(
    phi,
    *,
    device=None,
    parity: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU port of MATLAB generate_ris_response(phi).

    Returns:
        beta  : real tensor, same shape as phi
        gamma : complex tensor, same shape as phi
    """
    dev = _device(device)
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    ph = torch.as_tensor(phi, dtype=rd, device=dev)

    # Exact project formula.
    beta = (
        (1.0 - BETA_MIN)
        * ((torch.sin(ph - C_PHASE) + 1.0) / 2.0) ** K_EXP
        + BETA_MIN
    )

    gamma = beta.to(cd) * torch.exp(1j * ph).to(cd)

    return beta, gamma


@torch.inference_mode()
def generate_ris_response_from_z(
    z,
    *,
    phase_levels_deg: Tuple[float, float] = PHASE_LEVELS_DEG,
    device=None,
    parity: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Complete binary RIS path:
        z -> phi -> beta -> gamma
    """
    phi = z_to_phi(
        z,
        phase_levels_deg=phase_levels_deg,
        device=device,
        parity=parity,
    )
    beta, gamma = generate_ris_response_from_phi(
        phi,
        device=phi.device,
        parity=parity,
    )
    return {
        "phi": phi,
        "beta": beta,
        "gamma": gamma,
    }


def _rel_fro(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()), np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel()) / den)


def compare_stage4_matlab_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
) -> Dict[str, float]:
    """
    Compare one MATLAB Stage-4 .mat file.

    Expected variables:
        Z, phi, beta, gamma
    """
    from scipy.io import loadmat

    M = loadmat(mat_path, squeeze_me=False)

    Z = np.asarray(M["Z"])
    phi_ref = np.asarray(M["phi"])
    beta_ref = np.asarray(M["beta"])
    gamma_ref = np.asarray(M["gamma"])

    out = generate_ris_response_from_z(
        Z,
        device=device,
        parity=parity,
    )

    phi = out["phi"].detach().cpu().numpy()
    beta = out["beta"].detach().cpu().numpy()
    gamma = out["gamma"].detach().cpu().numpy()

    return {
        "phi_relFro": _rel_fro(phi, phi_ref),
        "phi_maxAbs": float(np.max(np.abs(phi-phi_ref))),
        "beta_relFro": _rel_fro(beta, beta_ref),
        "beta_maxAbs": float(np.max(np.abs(beta-beta_ref))),
        "gamma_relFro": _rel_fro(gamma, gamma_ref),
        "gamma_maxAbs": float(np.max(np.abs(gamma-gamma_ref))),
    }


@torch.inference_mode()
def benchmark_ris_response(
    *,
    n_candidates: int = 4096,
    n_ris: int = 512,
    repeats: int = 20,
    device=None,
) -> Dict[str, Any]:
    """
    Production float32/complex64 throughput benchmark.
    """
    dev = _device(device)

    gen = torch.Generator(device=dev)
    gen.manual_seed(42)

    Z = torch.randint(
        0, 2,
        (n_candidates, n_ris),
        device=dev,
        generator=gen,
        dtype=torch.int8,
    )

    # warmup
    _ = generate_ris_response_from_z(
        Z,
        device=dev,
        parity=False,
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = generate_ris_response_from_z(
            Z,
            device=dev,
            parity=False,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.perf_counter()-t0)

    med = float(np.median(times))
    total_elements = n_candidates * n_ris

    return {
        "device": str(dev),
        "n_candidates": int(n_candidates),
        "nRIS": int(n_ris),
        "median_seconds": med,
        "candidates_per_second": float(n_candidates/med),
        "ris_elements_per_second": float(total_elements/med),
    }
