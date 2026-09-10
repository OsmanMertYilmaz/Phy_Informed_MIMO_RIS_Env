"""Deployment-computable z-channel interaction features.

The features in this module are deterministic functions of the physical bank,
the Type-I precoder and the binary RIS state.  They never consume empirical
Monte-Carlo moments or teacher targets.

Two details are deliberately explicit:

* ``rho_RU`` lives on the UE receive array and is not an RIS-element profile.
  The RIS-side RU profile below is built from ``muRU`` and ``rho_RUhop``.
* ``z`` selects the project's 45/135 degree complex RIS responses.  It is not
  an on/off mask.  The exact coherent feature therefore uses complex ``gamma``;
  the remaining phase-1 shares are named as descriptive state allocations.
"""

from __future__ import annotations

from typing import Dict

import torch

from ris_env.label_engine import MeanOnlyWState, prepare_w_mean_only_state
from ris_env.ris_response import generate_ris_response_from_z
from ris_env.l3_features import physical_ris_grid


INTERACTION_FEATURE_NAMES = (
    "ix_phase1_br_exc_share",
    "ix_phase1_ru_gain_share",
    "ix_phase1_e2e_diag_share",
    "ix_corr_br_exc",
    "ix_corr_ru_gain",
    "ix_corr_e2e_diag",
    "ix_phase1_br_los_share",
    "ix_phase1_br_nlos_share",
    "ix_phase1_ll_diag_share",
    "ix_phase1_nn_diag_share",
    "ix_weighted_rough_m",
    "ix_weighted_rough_n",
    "ix_phase1_pol0_share",
    "ix_phase1_pol1_share",
    "ix_phase1_pol_contrast",
    "ix_los_coherence",
)


def _safe_share(z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fraction of non-negative profile weight assigned to phase state 1."""
    weight = torch.clamp(weight.real, min=0)
    den = weight.sum()
    tiny = torch.finfo(weight.dtype).tiny
    num = z @ weight
    return torch.where(den > tiny, num / den, torch.zeros_like(num))


def _profile_corr(z: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every binary pattern with one real profile."""
    p = profile.real
    zc = z - z.mean(dim=1, keepdim=True)
    pc = p - p.mean()
    num = zc @ pc
    den = torch.sqrt(torch.sum(zc * zc, dim=1) * torch.sum(pc * pc))
    tiny = torch.finfo(p.dtype).tiny
    return torch.where(den > tiny, num / den, torch.zeros_like(num))


def _weighted_roughness(
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    nris_x: int,
    nris_y: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Power-weighted phase transitions along physical M and N axes.

    Port order is MATLAB-compatible: M fastest, then N, then polarization.
    Consequently the physical view is [candidate, polarization, N, M].
    """
    c = z.shape[0]
    zg = physical_ris_grid(z, nris_x, nris_y)
    wg = physical_ris_grid(torch.clamp(weight.real, min=0), nris_x, nris_y)
    tiny = torch.finfo(z.dtype).tiny

    if int(nris_x) > 1:
        wm = 0.5 * (wg[..., 1:] + wg[..., :-1])
        dm = torch.abs(zg[..., 1:] - zg[..., :-1])
        den_m = wm.sum()
        rough_m = torch.where(
            den_m > tiny,
            torch.sum(dm * wm[None, ...], dim=(1, 2, 3)) / den_m,
            torch.zeros(c, dtype=z.dtype, device=z.device),
        )
    else:
        rough_m = torch.zeros(c, dtype=z.dtype, device=z.device)

    if int(nris_y) > 1:
        wn = 0.5 * (wg[:, 1:, :] + wg[:, :-1, :])
        dn = torch.abs(zg[:, :, 1:, :] - zg[:, :, :-1, :])
        den_n = wn.sum()
        rough_n = torch.where(
            den_n > tiny,
            torch.sum(dn * wn[None, ...], dim=(1, 2, 3)) / den_n,
            torch.zeros(c, dtype=z.dtype, device=z.device),
        )
    else:
        rough_n = torch.zeros(c, dtype=z.dtype, device=z.device)

    return rough_m, rough_n


@torch.inference_mode()
def analytic_z_channel_interactions(
    static_env,
    w_state: MeanOnlyWState,
    z,
    *,
    nris_x: int,
    nris_y: int,
    parity: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return 16 analytic features for every z of one ``(bank, W)``.

    Complexity is linear in the number of candidates and RIS elements except
    for the UBR construction already performed by ``prepare_w_mean_only_state``.
    No stochastic channel realization is generated.
    """
    if w_state.UBR is None:
        raise ValueError("MeanOnlyWState.UBR is required for interactions.")

    dev = w_state.ubarBR.device
    rd = torch.float64 if parity else torch.float32
    zi = torch.as_tensor(z, device=dev)
    if zi.ndim != 2:
        raise ValueError(f"z must be [candidate,nRIS], got {tuple(zi.shape)}")
    if not torch.all((zi == 0) | (zi == 1)):
        raise ValueError("z must contain only 0/1 phase-state indices.")
    zf = zi.to(rd)

    nris = int(w_state.ubarBR.numel())
    if zi.shape[1] != nris:
        raise ValueError(f"z nRIS={zi.shape[1]} != state nRIS={nris}")
    if nris != 2 * int(nris_x) * int(nris_y):
        raise ValueError(
            f"nRIS={nris} != 2*nris_x*nris_y="
            f"{2*int(nris_x)*int(nris_y)}"
        )

    gamma = generate_ris_response_from_z(
        zi, device=dev, parity=parity
    )["gamma"]

    # BS->RIS excitation profiles, conditioned on W.
    u_total = torch.clamp(torch.diagonal(w_state.UBR).real.to(rd), min=0)
    u_los = torch.abs(w_state.ubarBR).square().to(rd)
    u_nlos = torch.clamp(u_total - u_los, min=0)

    # RIS-side RU profiles.  rho_RUhop already includes sigma2RU and its
    # diagonal is the NLOS transmit-side power for one UE receive branch.
    r_los = torch.sum(torch.abs(static_env.muRU).square(), dim=0).to(rd)
    n_r = int(static_env.muRU.shape[0])
    r_nlos = torch.clamp(
        n_r * torch.diagonal(static_env.rho_RUhop).real.to(rd), min=0
    )
    r_total = r_los + r_nlos

    # Diagonal parts of the four RU/BR LOS-NLOS combinations.
    d_ll = u_los * r_los
    d_ln = u_los * r_nlos
    d_nl = u_nlos * r_los
    d_nn = u_nlos * r_nlos
    d_total = d_ll + d_ln + d_nl + d_nn

    rough_m, rough_n = _weighted_roughness(
        zf, d_total, nris_x=nris_x, nris_y=nris_y
    )

    half = nris // 2
    pol0 = _safe_share(zf[:, :half], d_total[:half])
    pol1 = _safe_share(zf[:, half:], d_total[half:])

    # Exact coherent LOS contribution of the cascaded mean field:
    #   sum_r |sum_i gamma_i * (muBR*w)_i * muRU_{r,i}|^2.
    los_field = (gamma * w_state.ubarBR[None, :]) @ static_env.muRU.T
    los_power = torch.sum(torch.abs(los_field).square(), dim=1).to(rd)
    los_diag = torch.sum(torch.abs(gamma).square().to(rd) * d_ll[None, :], dim=1)
    tiny = torch.finfo(rd).tiny
    los_coherence = torch.where(
        los_diag > tiny,
        los_power / (float(nris) * los_diag),
        torch.zeros_like(los_power),
    )

    values = (
        _safe_share(zf, u_total),
        _safe_share(zf, r_total),
        _safe_share(zf, d_total),
        _profile_corr(zf, u_total),
        _profile_corr(zf, r_total),
        _profile_corr(zf, d_total),
        _safe_share(zf, u_los),
        _safe_share(zf, u_nlos),
        _safe_share(zf, d_ll),
        _safe_share(zf, d_nn),
        rough_m,
        rough_n,
        pol0,
        pol1,
        pol0 - pol1,
        los_coherence,
    )

    out = dict(zip(INTERACTION_FEATURE_NAMES, values))
    matrix = torch.stack(values, dim=1)
    if not torch.isfinite(matrix).all():
        raise RuntimeError("Non-finite analytic z-channel interaction feature.")
    return out


@torch.inference_mode()
def analytic_z_channel_interactions_multi_w(
    static_env,
    W,
    z,
    *,
    nris_x: int,
    nris_y: int,
    parity: bool = False,
) -> torch.Tensor:
    """Return ``[W, Z, 16]`` interactions for one physical bank."""
    wt = torch.as_tensor(W, device=static_env.muBR.device)
    parts = []
    for w in wt:
        state = prepare_w_mean_only_state(static_env, w)
        features = analytic_z_channel_interactions(
            static_env,
            state,
            z,
            nris_x=nris_x,
            nris_y=nris_y,
            parity=parity,
        )
        parts.append(torch.stack(tuple(features.values()), dim=1))
    return torch.stack(parts, dim=0)
