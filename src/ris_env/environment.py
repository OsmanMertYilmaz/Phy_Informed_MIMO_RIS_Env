"""
Stage 8A — Full deterministic RIS environment integration.

This composes the already validated Stage 1–7 modules:

    geometry
      -> LSP / K
      -> channel moments
      -> spatial rho
      -> Type-I rank-1 W reconstruction from WIdx
      -> binary RIS z -> gamma
      -> effective moments
      -> Cmat / muSNR / sigma2Wick

Important scope:
    - deterministic analytical environment
    - W is represented by its Type-I codebook index WIdx=[i11,i12,i2]
    - no stochastic channel realization / pilot-SVD selection yet
    - nl=1
    - XP=2
    - RIS phase levels [45,135] deg

The purpose of Stage 8 is integration parity:
all individual blocks already passed separately; now we verify that the
complete Python call reproduces MATLAB from raw bank metadata + WIdx + Z.
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import numpy as np
import torch

from ris_env.geometry_lsp import (
    SCENARIO_TO_ID,
    generate_geometry_batch,
    generate_lsp_batch,
)
from ris_env.antenna import (
    ArraySpec,
    generate_channel_moments_batch,
)
from ris_env.spatial_correlation import (
    build_displacement_cache_from_dbar,
    compute_ch_rho_avg_batch,
    compute_ch_eff_rho_avg_batch,
)
from ris_env.ris_response import (
    generate_ris_response_from_z,
)
from ris_env.codebook import (
    generate_codebook_rank1,
)
from ris_env.snr_statistics import (
    build_static_environment,
    prepare_w_state,
    evaluate_gamma_batch,
)


C0 = 299792458.0


@dataclass
class BankInput:
    scenario_br: str
    scenario_ru: str
    fc: float

    ris: tuple[float,float,float]
    gnb: tuple[float,float,float]
    ue: tuple[float,float,float]

    nT1: int
    nT2: int
    nR1: int
    nR2: int
    nRIS_x: int
    nRIS_y: int

    # MATLAB 1-based Type-I rank-1 indices:
    # [i11, i12, i2]
    WIdx: tuple[int,int,int]


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _scalar(x):
    if torch.is_tensor(x):
        return x.reshape(()).item()
    return x


def _select_w_from_idx(codebook, WIdx):
    i11,i12,i2 = [int(x) for x in WIdx]
    if not (1 <= i11 <= codebook.i11_len):
        raise ValueError(f"i11={i11} out of range")
    if not (1 <= i12 <= codebook.i12_len):
        raise ValueError(f"i12={i12} out of range")
    if not (1 <= i2 <= codebook.i2_len):
        raise ValueError(f"i2={i2} out of range")
    return codebook.values[:,i2-1,i11-1,i12-1].contiguous()


@torch.inference_mode()
def build_deterministic_bank(
    bank: BankInput,
    *,
    device=None,
    parity: bool = False,
    gh_pair_chunk: int = 80,
) -> Dict[str,Any]:
    """
    Build all gamma-independent state for one bank/W.

    Returns a dictionary containing:
        geometry, LSPs, moments, rho, W, static_env, w_state, timings
    """
    dev = _device(device)

    timings = {}

    # ----------------------------------------------------------
    # Stage 6 + 7: geometry + LSP
    # ----------------------------------------------------------
    _sync(dev)
    t0 = time.perf_counter()

    geom = generate_geometry_batch(
        bank.ris,bank.gnb,bank.ue,
        device=dev,parity=parity,
    )

    sid_br = torch.tensor(
        SCENARIO_TO_ID[bank.scenario_br],
        dtype=torch.long,
        device=dev,
    )
    sid_ru = torch.tensor(
        SCENARIO_TO_ID[bank.scenario_ru],
        dtype=torch.long,
        device=dev,
    )

    lsp_br = generate_lsp_batch(
        sid_br,bank.fc,
        geom.ris2gnb,geom.gnb2ris,
        device=dev,parity=parity,
    )
    lsp_ru = generate_lsp_batch(
        sid_ru,bank.fc,
        geom.ue2ris,geom.ris2ue,
        device=dev,parity=parity,
    )

    _sync(dev)
    timings["geometry_lsp_s"] = time.perf_counter()-t0

    # ----------------------------------------------------------
    # Stage 1: moments
    # ----------------------------------------------------------
    _sync(dev)
    t0 = time.perf_counter()

    br = generate_channel_moments_batch(
        tx_spec=ArraySpec(bank.nT1,bank.nT2),
        rx_spec=ArraySpec(bank.nRIS_x,bank.nRIS_y),
        a_vectors=geom.ris2gnb.reshape(1,3),
        d_vectors=geom.gnb2ris.reshape(1,3),
        carrier_frequency=bank.fc,
        K=lsp_br.K_linear.reshape(1),
        mu_xpr=lsp_br.mu_XPR.reshape(1),
        sigma_xpr=lsp_br.sigma_XPR.reshape(1),
        c0=C0,
        device=dev,
        parity=parity,
    )

    ru = generate_channel_moments_batch(
        tx_spec=ArraySpec(bank.nRIS_x,bank.nRIS_y),
        rx_spec=ArraySpec(bank.nR1,bank.nR2),
        a_vectors=geom.ue2ris.reshape(1,3),
        d_vectors=geom.ris2ue.reshape(1,3),
        carrier_frequency=bank.fc,
        K=lsp_ru.K_linear.reshape(1),
        mu_xpr=lsp_ru.mu_XPR.reshape(1),
        sigma_xpr=lsp_ru.sigma_XPR.reshape(1),
        c0=C0,
        device=dev,
        parity=parity,
    )

    _sync(dev)
    timings["moments_s"] = time.perf_counter()-t0

    lambda0 = C0 / float(bank.fc)

    # ----------------------------------------------------------
    # Stage 2: rho
    # Caches are shape/static-array dependent. For this Stage-8
    # integration test they are built here; final repo will cache
    # them globally by array shape.
    # ----------------------------------------------------------
    _sync(dev)
    t0 = time.perf_counter()

    dbarTBR = br["dbarT"][0].detach().cpu().numpy()
    dbarRBR = br["dbarR"][0].detach().cpu().numpy()
    dbarTRU = ru["dbarT"][0].detach().cpu().numpy()
    dbarRRU = ru["dbarR"][0].detach().cpu().numpy()

    cache_t_br = build_displacement_cache_from_dbar(
        dbarTBR,lambda0,device=dev,parity=parity
    )
    cache_r_br = build_displacement_cache_from_dbar(
        dbarRBR,lambda0,device=dev,parity=parity
    )
    cache_t_ru = build_displacement_cache_from_dbar(
        dbarTRU,lambda0,device=dev,parity=parity
    )
    cache_r_ru = build_displacement_cache_from_dbar(
        dbarRRU,lambda0,device=dev,parity=parity
    )

    rhoRB_b,rhoBR_b = compute_ch_eff_rho_avg_batch(
        cache_tx=cache_t_br,
        cache_rx=cache_r_br,
        arrival_vector=geom.ris2gnb.reshape(1,3),
        departure_vector=geom.gnb2ris.reshape(1,3),
        mu_ASA=lsp_br.mu_ASA.reshape(1),
        sig_ASA=lsp_br.sigma_ASA.reshape(1),
        mu_ZSA=lsp_br.mu_ZSA.reshape(1),
        sig_ZSA=lsp_br.sigma_ZSA.reshape(1),
        mu_ASD=lsp_br.mu_ASD.reshape(1),
        sig_ASD=lsp_br.sigma_ASD.reshape(1),
        mu_ZSD=lsp_br.mu_ZSD.reshape(1),
        sig_ZSD=lsp_br.sigma_ZSD.reshape(1),
        c_ASA=lsp_br.c_ASA.reshape(1),
        c_ZSA=lsp_br.c_ZSA.reshape(1),
        c_ASD=lsp_br.c_ASD.reshape(1),
        c_ZSD=lsp_br.c_ZSD.reshape(1),
        mu_offset_ZOD=lsp_br.mu_offset_ZOD.reshape(1),
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRU_b,rhoUR_b = compute_ch_eff_rho_avg_batch(
        cache_tx=cache_t_ru,
        cache_rx=cache_r_ru,
        arrival_vector=geom.ue2ris.reshape(1,3),
        departure_vector=geom.ris2ue.reshape(1,3),
        mu_ASA=lsp_ru.mu_ASA.reshape(1),
        sig_ASA=lsp_ru.sigma_ASA.reshape(1),
        mu_ZSA=lsp_ru.mu_ZSA.reshape(1),
        sig_ZSA=lsp_ru.sigma_ZSA.reshape(1),
        mu_ASD=lsp_ru.mu_ASD.reshape(1),
        sig_ASD=lsp_ru.sigma_ASD.reshape(1),
        mu_ZSD=lsp_ru.mu_ZSD.reshape(1),
        sig_ZSD=lsp_ru.sigma_ZSD.reshape(1),
        c_ASA=lsp_ru.c_ASA.reshape(1),
        c_ZSA=lsp_ru.c_ZSA.reshape(1),
        c_ASD=lsp_ru.c_ASD.reshape(1),
        c_ZSD=lsp_ru.c_ZSD.reshape(1),
        mu_offset_ZOD=lsp_ru.mu_offset_ZOD.reshape(1),
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRUhopBase_b = compute_ch_rho_avg_batch(
        cache=cache_t_ru,
        vec=geom.ris2ue.reshape(1,3),
        mu_lg_az=lsp_ru.mu_ASD.reshape(1),
        sig_lg_az=lsp_ru.sigma_ASD.reshape(1),
        mu_lg_zn=lsp_ru.mu_ZSD.reshape(1),
        sig_lg_zn=lsp_ru.sigma_ZSD.reshape(1),
        c_az=lsp_ru.c_ASD.reshape(1),
        c_zn_scale=lsp_ru.c_ZSD.reshape(1),
        mu_offset_zn=lsp_ru.mu_offset_ZOD.reshape(1),
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRB = rhoRB_b[0]
    rhoBR = rhoBR_b[0]
    rhoRU = rhoRU_b[0]
    rhoUR = rhoUR_b[0]
    rhoRUhop = ru["sigma2H"][0] * rhoRUhopBase_b[0]

    _sync(dev)
    timings["rho_s"] = time.perf_counter()-t0

    # ----------------------------------------------------------
    # Stage 5: reconstruct fixed W from WIdx
    # ----------------------------------------------------------
    _sync(dev)
    t0 = time.perf_counter()

    codebook = generate_codebook_rank1(
        2,bank.nT1,bank.nT2,1,1,
        device=dev,parity=parity,
    )
    W = _select_w_from_idx(codebook,bank.WIdx)

    _sync(dev)
    timings["codebook_w_s"] = time.perf_counter()-t0

    # ----------------------------------------------------------
    # Stage 3: gamma-independent environment + W state
    # ----------------------------------------------------------
    _sync(dev)
    t0 = time.perf_counter()

    static_env = build_static_environment(
        rho_RU=rhoRU,
        rho_UR=rhoUR,
        rho_RB=rhoRB,
        rho_BR=rhoBR,
        rho_RUhop=rhoRUhop,
        mu_XPR_BR=lsp_br.mu_XPR,
        sigma_XPR_BR=lsp_br.sigma_XPR,
        mu_XPR_RU=lsp_ru.mu_XPR,
        sigma_XPR_RU=lsp_ru.sigma_XPR,
        muBR=br["muH"][0],
        sigma2BR=br["sigma2H"][0],
        muRU=ru["muH"][0],
        sigma2RU=ru["sigma2H"][0],
        device=dev,
        parity=parity,
    )

    w_state = prepare_w_state(static_env,W)

    _sync(dev)
    timings["prepare_w_state_s"] = time.perf_counter()-t0

    timings["bank_prepare_total_s"] = sum(timings.values())

    return {
        "bank": bank,
        "geometry": geom,
        "lsp_br": lsp_br,
        "lsp_ru": lsp_ru,
        "br": br,
        "ru": ru,
        "rhoRB": rhoRB,
        "rhoBR": rhoBR,
        "rhoRU": rhoRU,
        "rhoUR": rhoUR,
        "rhoRUhop": rhoRUhop,
        "W": W,
        "static_env": static_env,
        "w_state": w_state,
        "timings": timings,
    }


@torch.inference_mode()
def evaluate_z_candidates(
    prepared: Dict[str,Any],
    Z,
    *,
    parity: bool = False,
    candidate_chunk: Optional[int] = None,
) -> Dict[str,Any]:
    """
    Evaluate many binary RIS patterns for a prepared bank/W.
    """
    dev = prepared["W"].device

    _sync(dev)
    t0 = time.perf_counter()

    ris = generate_ris_response_from_z(
        Z,device=dev,parity=parity
    )

    _sync(dev)
    ris_response_s = time.perf_counter()-t0

    _sync(dev)
    t0 = time.perf_counter()

    stats = evaluate_gamma_batch(
        prepared["w_state"],
        ris["gamma"],
        candidate_chunk=candidate_chunk,
    )

    _sync(dev)
    stats_s = time.perf_counter()-t0

    return {
        **ris,
        **stats,
        "UBR": prepared["w_state"].UBR,
        "W": prepared["W"],
        "timings": {
            "ris_response_s": ris_response_s,
            "candidate_stats_s": stats_s,
            "candidate_total_s": ris_response_s + stats_s,
        },
    }


def _rel_fro(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()),np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel())/den)


def _max_abs(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def _mat_str(x):
    a = np.asarray(x)
    if a.dtype.kind in ("U","S"):
        return str(a.squeeze())
    if a.dtype == object:
        y = a.squeeze()
        if hasattr(y,"item"):
            y = y.item()
        return str(y)
    return str(a.squeeze())


def compare_stage8_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
    gh_pair_chunk: int = 80,
) -> Dict[str,Any]:
    from scipy.io import loadmat

    M = loadmat(mat_path,squeeze_me=False)

    def sc(name):
        return float(np.asarray(M[name]).reshape(()))

    def si(name):
        return int(np.asarray(M[name]).reshape(()))

    def sstr(name):
        x = np.asarray(M[name])
        # scipy char arrays may be U1/U...; flatten and join.
        if x.dtype.kind in ("U","S"):
            return "".join(x.reshape(-1).tolist()).strip()
        y = x.squeeze()
        return str(y.item() if hasattr(y,"item") else y).strip()

    bank = BankInput(
        scenario_br=sstr("scenarioBR"),
        scenario_ru=sstr("scenarioRU"),
        fc=sc("fc"),
        ris=tuple(np.asarray(M["ris"]).reshape(-1).astype(float).tolist()),
        gnb=tuple(np.asarray(M["gnb"]).reshape(-1).astype(float).tolist()),
        ue=tuple(np.asarray(M["ue"]).reshape(-1).astype(float).tolist()),
        nT1=si("nT1"), nT2=si("nT2"),
        nR1=si("nR1"), nR2=si("nR2"),
        nRIS_x=si("nRISx"), nRIS_y=si("nRISy"),
        WIdx=tuple(np.asarray(M["WIdx"]).reshape(-1).astype(int).tolist()),
    )

    prepared = build_deterministic_bank(
        bank,
        device=device,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    Z = np.asarray(M["Z"],dtype=np.int8)
    out = evaluate_z_candidates(
        prepared,Z,
        parity=parity,
    )

    py = {
        "W": out["W"].detach().cpu().numpy().reshape(-1,1),
        "gammaCandidates": out["gamma"].detach().cpu().numpy(),
        "UBR": out["UBR"].detach().cpu().numpy(),
        "muFeffCandidates": out["muFeff"].detach().cpu().numpy(),
        "sigma2FeffCandidates": out["sigma2Feff"].detach().cpu().numpy(),
        "CmatCandidates": out["Cmat"].detach().cpu().numpy(),
        "muSNRCandidates": out["muSNR"].detach().cpu().numpy(),
        "sigma2WickCandidates": out["sigma2Wick"].detach().cpu().numpy(),
    }

    # MATLAB exports candidate-major arrays where practical.
    refs = {
        "W": np.asarray(M["W"]),
        "gammaCandidates": np.asarray(M["gammaCandidates"]),
        "UBR": np.asarray(M["UBR"]),
        "muFeffCandidates": np.asarray(M["muFeffCandidates"]),
        "sigma2FeffCandidates": np.asarray(M["sigma2FeffCandidates"]),
        "CmatCandidates": np.moveaxis(np.asarray(M["CmatCandidates"]),2,0),
        "muSNRCandidates": np.asarray(M["muSNRCandidates"]).reshape(-1),
        "sigma2WickCandidates": np.asarray(M["sigma2WickCandidates"]).reshape(-1),
    }

    metrics = {}

    for name in [
        "W","gammaCandidates","UBR",
        "muFeffCandidates","sigma2FeffCandidates","CmatCandidates"
    ]:
        metrics[f"{name}_relFro"] = _rel_fro(py[name],refs[name])
        metrics[f"{name}_maxAbs"] = _max_abs(py[name],refs[name])

    for name in ["muSNRCandidates","sigma2WickCandidates"]:
        metrics[f"{name}_relFro"] = _rel_fro(py[name],refs[name])
        metrics[f"{name}_maxAbs"] = _max_abs(py[name],refs[name])

    metrics.update({
        f"time_{k}": float(v)
        for k,v in {**prepared["timings"],**out["timings"]}.items()
    })

    metrics["scenarioBR"] = bank.scenario_br
    metrics["scenarioRU"] = bank.scenario_ru
    metrics["nT"] = int(2*bank.nT1*bank.nT2)
    metrics["nR"] = int(2*bank.nR1*bank.nR2)
    metrics["nRIS"] = int(2*bank.nRIS_x*bank.nRIS_y)
    metrics["nCandidates"] = int(Z.shape[0])

    return metrics
