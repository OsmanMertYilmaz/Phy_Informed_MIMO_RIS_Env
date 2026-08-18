"""
Stage 8A optimization pass #1:
    shape-cached + shape-batched rho evaluation.

Why:
    Stage-8 profiling showed that rho generation dominates deterministic
    bank preparation. The original integration path rebuilt displacement
    caches for every bank and evaluated one bank at a time.

This module changes two things without changing the mathematics:

1) Array displacement caches are constructed directly from normalized
   antenna geometry and reused by array shape. They no longer depend on
   carrier frequency or bank geometry.

2) Banks with identical
       (nT1,nT2,nR1,nR2,nRIS_x,nRIS_y)
   are evaluated as GPU batches.

No analytical formula is changed.

The existing ris_gpu_rho_stage2.py functions are still the numerical kernel.
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, Tuple, Iterable, Optional
import math
import time

import numpy as np
import pandas as pd
import torch

from ris_env.geometry_lsp import (
    SCENARIO_TO_ID,
    generate_geometry_batch,
    generate_lsp_batch,
)
from ris_env.antenna import (
    ArraySpec,
    build_dualpol_positions_lambda,
    generate_channel_moments_batch,
)
from ris_env.spatial_correlation import (
    DisplacementCache,
    build_displacement_cache_from_dbar,
    compute_ch_rho_avg_batch,
    compute_ch_eff_rho_avg_batch,
)


C0 = 299792458.0


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def shape_key_from_row(row) -> Tuple[int,int,int,int,int,int]:
    return (
        int(row.nT1), int(row.nT2),
        int(row.nR1), int(row.nR2),
        int(row.nRIS_x), int(row.nRIS_y),
    )


def add_shape_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_shape_key"] = list(zip(
        x["nT1"].astype(int),
        x["nT2"].astype(int),
        x["nR1"].astype(int),
        x["nR2"].astype(int),
        x["nRIS_x"].astype(int),
        x["nRIS_y"].astype(int),
    ))
    return x


@dataclass
class ShapeRhoCaches:
    br_tx: DisplacementCache
    br_rx: DisplacementCache
    ru_tx: DisplacementCache
    ru_rx: DisplacementCache


class ArrayDisplacementCacheManager:
    """
    Cache normalized unique-displacement maps by array shape.

    Stage 1 positions are represented in wavelength units. Therefore
    displacement/lambda is shape-only for the locked 0.5-lambda arrays.

    We intentionally build unique-displacement maps in float64 on CPU once
    and then cast/store them on the requested device/dtype.
    """

    def __init__(self):
        self._array_cache: Dict[tuple, DisplacementCache] = {}
        self._shape_cache: Dict[tuple, ShapeRhoCaches] = {}

    @staticmethod
    def _dev_key(dev: torch.device) -> tuple:
        return (dev.type, -1 if dev.index is None else int(dev.index))

    def get_array_cache(
        self,
        M: int,
        N: int,
        *,
        device=None,
        parity: bool = False,
    ) -> DisplacementCache:
        dev = _device(device)
        key = (int(M),int(N),bool(parity),self._dev_key(dev))

        if key in self._array_cache:
            return self._array_cache[key]

        # Build canonical normalized positions in double precision.
        spec = ArraySpec(int(M),int(N))
        pos_lambda = build_dualpol_positions_lambda(
            spec,
            device="cpu",
            parity=True,
        ).cpu().numpy()

        # Pass lambda0=1: dbar/lambda0 == normalized positions.
        cache = build_displacement_cache_from_dbar(
            pos_lambda,
            1.0,
            device=dev,
            parity=parity,
        )
        self._array_cache[key] = cache
        return cache

    def get_shape_caches(
        self,
        shape_key: Tuple[int,int,int,int,int,int],
        *,
        device=None,
        parity: bool = False,
    ) -> ShapeRhoCaches:
        dev = _device(device)
        key = tuple(shape_key) + (bool(parity),) + self._dev_key(dev)

        if key in self._shape_cache:
            return self._shape_cache[key]

        nT1,nT2,nR1,nR2,nRISx,nRISy = map(int,shape_key)

        caches = ShapeRhoCaches(
            br_tx=self.get_array_cache(
                nT1,nT2,device=dev,parity=parity
            ),
            br_rx=self.get_array_cache(
                nRISx,nRISy,device=dev,parity=parity
            ),
            ru_tx=self.get_array_cache(
                nRISx,nRISy,device=dev,parity=parity
            ),
            ru_rx=self.get_array_cache(
                nR1,nR2,device=dev,parity=parity
            ),
        )
        self._shape_cache[key] = caches
        return caches

    def stats(self) -> Dict[str,int]:
        return {
            "array_cache_entries": len(self._array_cache),
            "shape_cache_entries": len(self._shape_cache),
        }


def estimate_rho_output_bytes_per_bank(
    shape_key: Tuple[int,int,int,int,int,int],
    *,
    parity: bool = False,
) -> int:
    """
    Estimate retained rho output bytes for one bank:
      rhoRB  [nRIS,nRIS,20]
      rhoBR  [nT,nT,20]
      rhoRU  [nR,nR,20]
      rhoUR  [nRIS,nRIS,20]
      rhoHop [nRIS,nRIS]

    This is NOT peak temporary memory; the auto chunker applies a safety factor.
    """
    nT1,nT2,nR1,nR2,nRISx,nRISy = map(int,shape_key)

    nT = 2*nT1*nT2
    nR = 2*nR1*nR2
    nRIS = 2*nRISx*nRISy
    L = 20

    elems = (
        nRIS*nRIS*L
        + nT*nT*L
        + nR*nR*L
        + nRIS*nRIS*L
        + nRIS*nRIS
    )

    complex_bytes = 16 if parity else 8
    return int(elems*complex_bytes)


def auto_rho_chunk_size(
    shape_key: Tuple[int,int,int,int,int,int],
    *,
    parity: bool = False,
    target_memory_mb: float = 768.0,
    safety_factor: float = 3.0,
    max_batch: int = 128,
) -> int:
    """
    Conservative chunk heuristic. GH temporaries and output tensors coexist,
    so retained rho bytes are multiplied by safety_factor.
    """
    per_bank = estimate_rho_output_bytes_per_bank(
        shape_key,parity=parity
    )
    budget = int(target_memory_mb*1024**2)
    batch = budget // max(int(per_bank*safety_factor),1)
    return max(1,min(int(batch),int(max_batch)))


def _same_shape_or_raise(df: pd.DataFrame) -> Tuple[int,int,int,int,int,int]:
    if len(df) == 0:
        raise ValueError("empty dataframe chunk")
    keys = {
        (
            int(r.nT1),int(r.nT2),
            int(r.nR1),int(r.nR2),
            int(r.nRIS_x),int(r.nRIS_y)
        )
        for r in df.itertuples(index=False)
    }
    if len(keys) != 1:
        raise ValueError(
            "compute_rho_same_shape_chunk requires exactly one array shape"
        )
    return next(iter(keys))


@torch.inference_mode()
def compute_rho_same_shape_chunk(
    df_chunk: pd.DataFrame,
    cache_manager: ArrayDisplacementCacheManager,
    *,
    device=None,
    parity: bool = False,
    gh_pair_chunk: int = 80,
    return_moments: bool = True,
) -> Dict[str,Any]:
    """
    GPU batch for one same-shape dataframe chunk.

    All banks in the chunk may have different:
        - scenario
        - fc
        - gNB/RIS/UE positions
        - LOS/NLOS
        - LSP

    They must have the same antenna dimensions.
    """
    dev = _device(device)
    shape_key = _same_shape_or_raise(df_chunk)
    nT1,nT2,nR1,nR2,nRISx,nRISy = shape_key

    ris_np = df_chunk[["ris_x","ris_y","ris_z"]].to_numpy(dtype=np.float64)
    gnb_np = df_chunk[["gnb_x","gnb_y","gnb_z"]].to_numpy(dtype=np.float64)
    ue_np  = df_chunk[["ue_x","ue_y","ue_z"]].to_numpy(dtype=np.float64)
    fc_np  = df_chunk["fc"].to_numpy(dtype=np.float64)

    geom = generate_geometry_batch(
        ris_np,gnb_np,ue_np,
        device=dev,parity=parity,
    )

    sid_br = torch.tensor(
        [
            SCENARIO_TO_ID[str(x)]
            for x in df_chunk["scenario_BR"].astype(str)
        ],
        dtype=torch.long,
        device=dev,
    )
    sid_ru = torch.tensor(
        [
            SCENARIO_TO_ID[str(x)]
            for x in df_chunk["scenario_RU"].astype(str)
        ],
        dtype=torch.long,
        device=dev,
    )

    lsp_br = generate_lsp_batch(
        sid_br,fc_np,
        geom.ris2gnb,geom.gnb2ris,
        device=dev,parity=parity,
    )
    lsp_ru = generate_lsp_batch(
        sid_ru,fc_np,
        geom.ue2ris,geom.ris2ue,
        device=dev,parity=parity,
    )

    br = generate_channel_moments_batch(
        tx_spec=ArraySpec(nT1,nT2),
        rx_spec=ArraySpec(nRISx,nRISy),
        a_vectors=geom.ris2gnb,
        d_vectors=geom.gnb2ris,
        carrier_frequency=fc_np,
        K=lsp_br.K_linear,
        mu_xpr=lsp_br.mu_XPR,
        sigma_xpr=lsp_br.sigma_XPR,
        c0=C0,
        device=dev,
        parity=parity,
    )

    ru = generate_channel_moments_batch(
        tx_spec=ArraySpec(nRISx,nRISy),
        rx_spec=ArraySpec(nR1,nR2),
        a_vectors=geom.ue2ris,
        d_vectors=geom.ris2ue,
        carrier_frequency=fc_np,
        K=lsp_ru.K_linear,
        mu_xpr=lsp_ru.mu_XPR,
        sigma_xpr=lsp_ru.sigma_XPR,
        c0=C0,
        device=dev,
        parity=parity,
    )

    caches = cache_manager.get_shape_caches(
        shape_key,
        device=dev,
        parity=parity,
    )

    rhoRB,rhoBR = compute_ch_eff_rho_avg_batch(
        cache_tx=caches.br_tx,
        cache_rx=caches.br_rx,
        arrival_vector=geom.ris2gnb,
        departure_vector=geom.gnb2ris,
        mu_ASA=lsp_br.mu_ASA,
        sig_ASA=lsp_br.sigma_ASA,
        mu_ZSA=lsp_br.mu_ZSA,
        sig_ZSA=lsp_br.sigma_ZSA,
        mu_ASD=lsp_br.mu_ASD,
        sig_ASD=lsp_br.sigma_ASD,
        mu_ZSD=lsp_br.mu_ZSD,
        sig_ZSD=lsp_br.sigma_ZSD,
        c_ASA=lsp_br.c_ASA,
        c_ZSA=lsp_br.c_ZSA,
        c_ASD=lsp_br.c_ASD,
        c_ZSD=lsp_br.c_ZSD,
        mu_offset_ZOD=lsp_br.mu_offset_ZOD,
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRU,rhoUR = compute_ch_eff_rho_avg_batch(
        cache_tx=caches.ru_tx,
        cache_rx=caches.ru_rx,
        arrival_vector=geom.ue2ris,
        departure_vector=geom.ris2ue,
        mu_ASA=lsp_ru.mu_ASA,
        sig_ASA=lsp_ru.sigma_ASA,
        mu_ZSA=lsp_ru.mu_ZSA,
        sig_ZSA=lsp_ru.sigma_ZSA,
        mu_ASD=lsp_ru.mu_ASD,
        sig_ASD=lsp_ru.sigma_ASD,
        mu_ZSD=lsp_ru.mu_ZSD,
        sig_ZSD=lsp_ru.sigma_ZSD,
        c_ASA=lsp_ru.c_ASA,
        c_ZSA=lsp_ru.c_ZSA,
        c_ASD=lsp_ru.c_ASD,
        c_ZSD=lsp_ru.c_ZSD,
        mu_offset_ZOD=lsp_ru.mu_offset_ZOD,
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRUhopBase = compute_ch_rho_avg_batch(
        cache=caches.ru_tx,
        vec=geom.ris2ue,
        mu_lg_az=lsp_ru.mu_ASD,
        sig_lg_az=lsp_ru.sigma_ASD,
        mu_lg_zn=lsp_ru.mu_ZSD,
        sig_lg_zn=lsp_ru.sigma_ZSD,
        c_az=lsp_ru.c_ASD,
        c_zn_scale=lsp_ru.c_ZSD,
        mu_offset_zn=lsp_ru.mu_offset_ZOD,
        n_gh=20,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
    )

    rhoRUhop = ru["sigma2H"][:,None,None] * rhoRUhopBase

    out = {
        "shape_key": shape_key,
        "geometry": geom,
        "lsp_br": lsp_br,
        "lsp_ru": lsp_ru,
        "rhoRB": rhoRB,
        "rhoBR": rhoBR,
        "rhoRU": rhoRU,
        "rhoUR": rhoUR,
        "rhoRUhop": rhoRUhop,
    }

    if return_moments:
        out["br"] = br
        out["ru"] = ru

    return out


def _rel_fro_torch(a: torch.Tensor,b: torch.Tensor) -> float:
    da = a.to(torch.complex128 if a.is_complex() else torch.float64)
    db = b.to(torch.complex128 if b.is_complex() else torch.float64)
    den = max(float(torch.linalg.vector_norm(db.reshape(-1)).cpu()),np.finfo(float).eps)
    num = float(torch.linalg.vector_norm((da-db).reshape(-1)).cpu())
    return num/den


@torch.inference_mode()
def validate_cached_batch_against_dbar_reference(
    df_same_shape: pd.DataFrame,
    *,
    n_rows: int = 4,
    device=None,
    parity: bool = True,
    gh_pair_chunk: int = 80,
) -> Dict[str,float]:
    """
    Validate the optimization itself.

    NEW path:
        shape-only cached displacement maps + batched banks

    REFERENCE path:
        old Stage-8 behavior: displacement cache constructed from each
        bank's dbar/lambda.

    Both use the already validated Stage-2 kernel.
    """
    dev = _device(device)
    x = df_same_shape.iloc[:min(n_rows,len(df_same_shape))].copy()
    if len(x) == 0:
        raise ValueError("no rows supplied")

    mgr = ArrayDisplacementCacheManager()
    new = compute_rho_same_shape_chunk(
        x,mgr,
        device=dev,
        parity=parity,
        gh_pair_chunk=gh_pair_chunk,
        return_moments=True,
    )

    refs = {name: [] for name in ["rhoRB","rhoBR","rhoRU","rhoUR","rhoRUhop"]}

    for b in range(len(x)):
        br = new["br"]
        ru = new["ru"]
        geom = new["geometry"]
        lsp_br = new["lsp_br"]
        lsp_ru = new["lsp_ru"]

        fc = float(x.iloc[b]["fc"])
        lambda0 = C0/fc

        cache_t_br = build_displacement_cache_from_dbar(
            br["dbarT"][b].detach().cpu().numpy(),
            lambda0,device=dev,parity=parity
        )
        cache_r_br = build_displacement_cache_from_dbar(
            br["dbarR"][b].detach().cpu().numpy(),
            lambda0,device=dev,parity=parity
        )
        cache_t_ru = build_displacement_cache_from_dbar(
            ru["dbarT"][b].detach().cpu().numpy(),
            lambda0,device=dev,parity=parity
        )
        cache_r_ru = build_displacement_cache_from_dbar(
            ru["dbarR"][b].detach().cpu().numpy(),
            lambda0,device=dev,parity=parity
        )

        rb,brho = compute_ch_eff_rho_avg_batch(
            cache_tx=cache_t_br,cache_rx=cache_r_br,
            arrival_vector=geom.ris2gnb[b:b+1],
            departure_vector=geom.gnb2ris[b:b+1],
            mu_ASA=lsp_br.mu_ASA[b:b+1],sig_ASA=lsp_br.sigma_ASA[b:b+1],
            mu_ZSA=lsp_br.mu_ZSA[b:b+1],sig_ZSA=lsp_br.sigma_ZSA[b:b+1],
            mu_ASD=lsp_br.mu_ASD[b:b+1],sig_ASD=lsp_br.sigma_ASD[b:b+1],
            mu_ZSD=lsp_br.mu_ZSD[b:b+1],sig_ZSD=lsp_br.sigma_ZSD[b:b+1],
            c_ASA=lsp_br.c_ASA[b:b+1],c_ZSA=lsp_br.c_ZSA[b:b+1],
            c_ASD=lsp_br.c_ASD[b:b+1],c_ZSD=lsp_br.c_ZSD[b:b+1],
            mu_offset_ZOD=lsp_br.mu_offset_ZOD[b:b+1],
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )

        rru,rur = compute_ch_eff_rho_avg_batch(
            cache_tx=cache_t_ru,cache_rx=cache_r_ru,
            arrival_vector=geom.ue2ris[b:b+1],
            departure_vector=geom.ris2ue[b:b+1],
            mu_ASA=lsp_ru.mu_ASA[b:b+1],sig_ASA=lsp_ru.sigma_ASA[b:b+1],
            mu_ZSA=lsp_ru.mu_ZSA[b:b+1],sig_ZSA=lsp_ru.sigma_ZSA[b:b+1],
            mu_ASD=lsp_ru.mu_ASD[b:b+1],sig_ASD=lsp_ru.sigma_ASD[b:b+1],
            mu_ZSD=lsp_ru.mu_ZSD[b:b+1],sig_ZSD=lsp_ru.sigma_ZSD[b:b+1],
            c_ASA=lsp_ru.c_ASA[b:b+1],c_ZSA=lsp_ru.c_ZSA[b:b+1],
            c_ASD=lsp_ru.c_ASD[b:b+1],c_ZSD=lsp_ru.c_ZSD[b:b+1],
            mu_offset_ZOD=lsp_ru.mu_offset_ZOD[b:b+1],
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )

        hop = compute_ch_rho_avg_batch(
            cache=cache_t_ru,
            vec=geom.ris2ue[b:b+1],
            mu_lg_az=lsp_ru.mu_ASD[b:b+1],
            sig_lg_az=lsp_ru.sigma_ASD[b:b+1],
            mu_lg_zn=lsp_ru.mu_ZSD[b:b+1],
            sig_lg_zn=lsp_ru.sigma_ZSD[b:b+1],
            c_az=lsp_ru.c_ASD[b:b+1],
            c_zn_scale=lsp_ru.c_ZSD[b:b+1],
            mu_offset_zn=lsp_ru.mu_offset_ZOD[b:b+1],
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )
        hop = ru["sigma2H"][b] * hop

        refs["rhoRB"].append(rb[0])
        refs["rhoBR"].append(brho[0])
        refs["rhoRU"].append(rru[0])
        refs["rhoUR"].append(rur[0])
        refs["rhoRUhop"].append(hop[0])

    metrics = {}
    for name in refs:
        ref = torch.stack(refs[name],dim=0)
        metrics[f"{name}_relFro"] = _rel_fro_torch(new[name],ref)
        metrics[f"{name}_maxAbs"] = float(
            torch.max(torch.abs(new[name]-ref)).detach().cpu()
        )

    return metrics


@torch.inference_mode()
def benchmark_single_bank_reference(
    df: pd.DataFrame,
    *,
    n_rows: int = 16,
    device=None,
    parity: bool = False,
    gh_pair_chunk: int = 80,
) -> Dict[str,float]:
    """
    Baseline: old per-bank cache construction + B=1 rho calls.
    Intended for a small subset only.
    """
    dev = _device(device)
    x = df.iloc[:min(n_rows,len(df))].copy()

    total_start = time.perf_counter()
    bank_times = []

    for _,one in x.iterrows():
        one_df = pd.DataFrame([one])

        # First compute batched primitives for B=1, but use no cached rho.
        shape_key = _same_shape_or_raise(one_df)
        nT1,nT2,nR1,nR2,nRISx,nRISy = shape_key

        ris = one_df[["ris_x","ris_y","ris_z"]].to_numpy(dtype=np.float64)
        gnb = one_df[["gnb_x","gnb_y","gnb_z"]].to_numpy(dtype=np.float64)
        ue  = one_df[["ue_x","ue_y","ue_z"]].to_numpy(dtype=np.float64)
        fc_np = one_df["fc"].to_numpy(dtype=np.float64)

        geom = generate_geometry_batch(ris,gnb,ue,device=dev,parity=parity)

        sid_br = torch.tensor(
            [SCENARIO_TO_ID[str(one["scenario_BR"])]],
            device=dev,dtype=torch.long
        )
        sid_ru = torch.tensor(
            [SCENARIO_TO_ID[str(one["scenario_RU"])]],
            device=dev,dtype=torch.long
        )

        lsp_br = generate_lsp_batch(
            sid_br,fc_np,geom.ris2gnb,geom.gnb2ris,
            device=dev,parity=parity
        )
        lsp_ru = generate_lsp_batch(
            sid_ru,fc_np,geom.ue2ris,geom.ris2ue,
            device=dev,parity=parity
        )

        br = generate_channel_moments_batch(
            tx_spec=ArraySpec(nT1,nT2),rx_spec=ArraySpec(nRISx,nRISy),
            a_vectors=geom.ris2gnb,d_vectors=geom.gnb2ris,
            carrier_frequency=fc_np,K=lsp_br.K_linear,
            mu_xpr=lsp_br.mu_XPR,sigma_xpr=lsp_br.sigma_XPR,
            c0=C0,device=dev,parity=parity
        )
        ru = generate_channel_moments_batch(
            tx_spec=ArraySpec(nRISx,nRISy),rx_spec=ArraySpec(nR1,nR2),
            a_vectors=geom.ue2ris,d_vectors=geom.ris2ue,
            carrier_frequency=fc_np,K=lsp_ru.K_linear,
            mu_xpr=lsp_ru.mu_XPR,sigma_xpr=lsp_ru.sigma_XPR,
            c0=C0,device=dev,parity=parity
        )

        lambda0 = C0/float(fc_np[0])

        _sync(dev)
        t0 = time.perf_counter()

        ctbr = build_displacement_cache_from_dbar(
            br["dbarT"][0].detach().cpu().numpy(),lambda0,
            device=dev,parity=parity
        )
        crbr = build_displacement_cache_from_dbar(
            br["dbarR"][0].detach().cpu().numpy(),lambda0,
            device=dev,parity=parity
        )
        ctru = build_displacement_cache_from_dbar(
            ru["dbarT"][0].detach().cpu().numpy(),lambda0,
            device=dev,parity=parity
        )
        crru = build_displacement_cache_from_dbar(
            ru["dbarR"][0].detach().cpu().numpy(),lambda0,
            device=dev,parity=parity
        )

        _ = compute_ch_eff_rho_avg_batch(
            cache_tx=ctbr,cache_rx=crbr,
            arrival_vector=geom.ris2gnb,departure_vector=geom.gnb2ris,
            mu_ASA=lsp_br.mu_ASA,sig_ASA=lsp_br.sigma_ASA,
            mu_ZSA=lsp_br.mu_ZSA,sig_ZSA=lsp_br.sigma_ZSA,
            mu_ASD=lsp_br.mu_ASD,sig_ASD=lsp_br.sigma_ASD,
            mu_ZSD=lsp_br.mu_ZSD,sig_ZSD=lsp_br.sigma_ZSD,
            c_ASA=lsp_br.c_ASA,c_ZSA=lsp_br.c_ZSA,
            c_ASD=lsp_br.c_ASD,c_ZSD=lsp_br.c_ZSD,
            mu_offset_ZOD=lsp_br.mu_offset_ZOD,
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )
        _ = compute_ch_eff_rho_avg_batch(
            cache_tx=ctru,cache_rx=crru,
            arrival_vector=geom.ue2ris,departure_vector=geom.ris2ue,
            mu_ASA=lsp_ru.mu_ASA,sig_ASA=lsp_ru.sigma_ASA,
            mu_ZSA=lsp_ru.mu_ZSA,sig_ZSA=lsp_ru.sigma_ZSA,
            mu_ASD=lsp_ru.mu_ASD,sig_ASD=lsp_ru.sigma_ASD,
            mu_ZSD=lsp_ru.mu_ZSD,sig_ZSD=lsp_ru.sigma_ZSD,
            c_ASA=lsp_ru.c_ASA,c_ZSA=lsp_ru.c_ZSA,
            c_ASD=lsp_ru.c_ASD,c_ZSD=lsp_ru.c_ZSD,
            mu_offset_ZOD=lsp_ru.mu_offset_ZOD,
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )
        _ = compute_ch_rho_avg_batch(
            cache=ctru,vec=geom.ris2ue,
            mu_lg_az=lsp_ru.mu_ASD,sig_lg_az=lsp_ru.sigma_ASD,
            mu_lg_zn=lsp_ru.mu_ZSD,sig_lg_zn=lsp_ru.sigma_ZSD,
            c_az=lsp_ru.c_ASD,c_zn_scale=lsp_ru.c_ZSD,
            mu_offset_zn=lsp_ru.mu_offset_ZOD,
            n_gh=20,parity=parity,gh_pair_chunk=gh_pair_chunk,
        )

        _sync(dev)
        bank_times.append(time.perf_counter()-t0)

    total = time.perf_counter()-total_start

    return {
        "n_banks": len(x),
        "rho_median_seconds_per_bank": float(np.median(bank_times)),
        "rho_mean_seconds_per_bank": float(np.mean(bank_times)),
        "wall_seconds": float(total),
        "banks_per_second_from_rho_median": float(1.0/np.median(bank_times)),
    }


@torch.inference_mode()
def benchmark_bucketed_rho(
    df: pd.DataFrame,
    *,
    device=None,
    parity: bool = False,
    gh_pair_chunk: int = 80,
    target_memory_mb: float = 768.0,
    max_batch: int = 128,
    safety_factor: float = 3.0,
    cache_manager: Optional[ArrayDisplacementCacheManager] = None,
) -> Dict[str,Any]:
    """
    Stream all rows grouped by identical array shape.

    Outputs are deliberately discarded after each chunk: this benchmark
    measures dataset-generation throughput without retaining all rho matrices.
    """
    dev = _device(device)
    mgr = cache_manager or ArrayDisplacementCacheManager()
    x = add_shape_key_columns(df)

    bucket_rows = []
    total_banks = 0

    _sync(dev)
    wall0 = time.perf_counter()

    # Stable ordering helps reproducibility.
    grouped = x.groupby("_shape_key",sort=True)

    for shape_key,bucket in grouped:
        chunk_size = auto_rho_chunk_size(
            shape_key,
            parity=parity,
            target_memory_mb=target_memory_mb,
            safety_factor=safety_factor,
            max_batch=max_batch,
        )

        times = []
        n_chunks = 0

        for start in range(0,len(bucket),chunk_size):
            chunk = bucket.iloc[start:start+chunk_size]

            _sync(dev)
            t0 = time.perf_counter()

            out = compute_rho_same_shape_chunk(
                chunk,mgr,
                device=dev,
                parity=parity,
                gh_pair_chunk=gh_pair_chunk,
                return_moments=False,
            )

            _sync(dev)
            dt = time.perf_counter()-t0
            times.append(dt)
            n_chunks += 1
            total_banks += len(chunk)

            # Drop large outputs before next chunk.
            del out
            if dev.type == "cuda":
                # Do not empty cache in production loops; allocator reuse is faster.
                pass

        bucket_rows.append({
            "shape_key": str(tuple(shape_key)),
            "banks": int(len(bucket)),
            "chunk_size": int(chunk_size),
            "chunks": int(n_chunks),
            "median_chunk_s": float(np.median(times)),
            "seconds_per_bank": float(np.sum(times)/len(bucket)),
            "banks_per_second": float(len(bucket)/np.sum(times)),
            "rho_output_MB_per_bank": (
                estimate_rho_output_bytes_per_bank(
                    shape_key,parity=parity
                )/1024**2
            ),
        })

    _sync(dev)
    wall = time.perf_counter()-wall0

    bucket_df = pd.DataFrame(bucket_rows).sort_values(
        ["banks_per_second","banks"],
        ascending=[True,False],
    ).reset_index(drop=True)

    return {
        "n_banks": int(total_banks),
        "n_shape_buckets": int(len(bucket_df)),
        "wall_seconds": float(wall),
        "banks_per_second": float(total_banks/wall),
        "seconds_per_bank": float(wall/total_banks),
        "cache_stats": mgr.stats(),
        "bucket_table": bucket_df,
    }
