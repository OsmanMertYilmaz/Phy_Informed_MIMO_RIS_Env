"""Deployable Level-3 features; no teacher labels or Monte Carlo inputs.

Port order is M fastest, then N, then polarization, matching MATLAB (:).
Covariance component names always use RU first, BR second.
"""
from __future__ import annotations

import numpy as np
import torch

PATTERN_VERSION = "physical_m_n_pol_v2"
COVARIANCE_VERSION = "z_covariance_v1"
COMPONENTS = ("LL", "LN", "NL", "NN")
COVARIANCE_FEATURE_NAMES = tuple(
    [f"cov_{c}_share" for c in COMPONENTS]
    + [f"cov_{c}_offdiag_norm" for c in COMPONENTS]
)


def physical_ris_grid(bits, n1: int, n2: int):
    """Return [..., polarization, N, M], for NumPy arrays or torch tensors."""
    n1, n2 = int(n1), int(n2)
    if n1 <= 0 or n2 <= 0 or bits.shape[-1] != 2 * n1 * n2:
        raise ValueError("RIS bit length must equal 2*n1*n2 with positive sizes")
    return bits.reshape(*bits.shape[:-1], 2, n2, n1)


def physical_pattern_features(bits, n1: int, n2: int) -> np.ndarray:
    """The existing 13 pattern descriptors, on the physical RIS grid.

    Input is [candidate, port]. transitionRate retains the original flat-port
    meaning; spatial roughness, polarization and FFT use physical axes.
    """
    bits = np.asarray(bits)
    if bits.ndim != 2 or not np.isin(bits, [0, 1]).all():
        raise ValueError("bits must be a binary [candidate, port] array")
    p = physical_ris_grid(bits, n1, n2).transpose(0, 3, 2, 1).astype(np.float32)
    p0, p1 = p[..., 0], p[..., 1]
    f = np.empty((len(bits), 13), dtype=np.float32)
    f[:, 0] = p.mean(axis=(1, 2, 3))
    f[:, 1], f[:, 2] = p0.mean(axis=(1, 2)), p1.mean(axis=(1, 2))
    f[:, 3] = f[:, 1] - f[:, 2]
    f[:, 4] = (p0 == p1).mean(axis=(1, 2))
    f[:, 5] = p.mean(axis=(2, 3)).std(axis=1)
    f[:, 6] = p.mean(axis=(1, 3)).std(axis=1)
    f[:, 7] = np.abs(np.diff(p, axis=1)).mean(axis=(1, 2, 3)) if n1 > 1 else 0
    f[:, 8] = np.abs(np.diff(p, axis=2)).mean(axis=(1, 2, 3)) if n2 > 1 else 0
    f[:, 9] = np.abs(np.diff(bits.astype(np.float32), axis=1)).mean(axis=1)
    centered = p0 - p0.mean(axis=(1, 2), keepdims=True)
    mag = np.abs(np.fft.fft2(centered, axes=(1, 2)))
    total = mag.sum(axis=(1, 2)) + 1e-12
    f[:, 10] = mag[:, :min(3, n1), :min(3, n2)].sum(axis=(1, 2)) / total
    f[:, 11] = mag.max(axis=(1, 2)) / total
    h1, h2 = n1 // 2, n2 // 2
    f[:, 12] = (
        p[:, :h1, :h2].mean(axis=(1, 2, 3))
        - p[:, h1:, h2:].mean(axis=(1, 2, 3))
    ) if h1 and h2 else 0
    return f


@torch.inference_mode()
def covariance_component_features(env, ubr, ubar, gamma, *, candidate_chunk=64):
    """Return [Z,8] LL/LN/NL/NN powers and signed off-diagonal effects.

    K_ab = (sum_r R_a,r) * U_b, where * is elementwise. P_ab is
    gamma.T K_ab gamma.conj(), NOT gamma.H K_ab gamma. Diagonal power
    includes abs(gamma)**2 because the two RIS states have different gains.

    First four features: P_ab / sum_ab P_ab.
    Last four: (P_ab - D_ab) / sum_ab D_ab. A shared diagonal denominator
    avoids unstable division by an absent LOS component and retains its weight.
    This is a second-moment decomposition, not an exact variance correction.
    """
    if candidate_chunk < 1:
        raise ValueError("candidate_chunk must be positive")
    g = torch.as_tensor(gamma, dtype=ubr.dtype, device=ubr.device)
    if g.ndim != 2 or g.shape[1] != ubr.shape[0] or not len(g):
        raise ValueError("gamma must be nonempty [Z,nRIS]")
    ubar = ubar.reshape(-1)
    if ubr.shape != (ubar.numel(), ubar.numel()):
        raise ValueError("UBR/ubar shape mismatch")
    herm = lambda x: 0.5 * (x + x.conj().transpose(-2, -1))
    ul = ubar[:, None] * ubar.conj()[None, :]
    un = herm(ubr - ul)
    rl = env.muRU.T @ env.muRU.conj()
    rn = int(env.muRU.shape[0]) * env.rho_RUhop
    kernels = torch.stack([herm(r * u) for r, u in
                           ((rl, ul), (rl, un), (rn, ul), (rn, un))])
    # Normalize all kernels by one positive scale before quadratic evaluation.
    # This makes the dimensionless features robust to path-loss magnitude.
    diag = kernels.diagonal(dim1=-2, dim2=-1).real
    scale = diag.clamp_min(0).sum()
    if not torch.isfinite(kernels).all() or not torch.isfinite(g).all():
        raise ValueError("Non-finite channel moments or RIS response")
    if scale <= 0:
        if kernels.abs().max() > 0:
            raise ValueError("Nonzero covariance with nonpositive total diagonal")
        return torch.zeros((len(g), 8), dtype=ubr.real.dtype, device=ubr.device)
    kernels = kernels / scale
    diag = kernels.diagonal(dim1=-2, dim2=-1).real
    tol = 64 * torch.finfo(diag.dtype).eps
    if (diag < -tol).any():
        raise ValueError("Covariance component has a negative diagonal")
    diag = diag.clamp_min(0)
    result = []
    for start in range(0, len(g), candidate_chunk):
        chunk = g[start:start + candidate_chunk]
        power = torch.einsum("zi,aij,zj->za", chunk, kernels, chunk.conj()).real
        diagonal = chunk.abs().square() @ diag.T
        dtotal = diagonal.sum(1, keepdim=True)
        if (power < -tol * dtotal.clamp_min(torch.finfo(diag.dtype).tiny)).any():
            raise ValueError("Covariance component power is materially negative")
        power = power.clamp_min(0)
        tiny = torch.finfo(power.dtype).tiny
        shares = power / power.sum(1, keepdim=True).clamp_min(tiny)
        offdiag = (power - diagonal) / dtotal.clamp_min(tiny)
        result.append(torch.cat([shares, offdiag], dim=1))
    out = torch.cat(result, dim=0)
    if not torch.isfinite(out).all():
        raise ValueError("Non-finite covariance feature")
    return out
