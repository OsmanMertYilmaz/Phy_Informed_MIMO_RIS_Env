"""
Stage 6 + Stage 7 GPU environment primitives:
    - geometry
    - large-scale parameters (LSP)

This module is a direct port of the user-provided MATLAB functions:

    generate_geometry
    generate_lsp

No model/formula corrections are applied. The goal is exact parity with the
current dataset-generation convention.

Official scenario ids:
    0 = UMi-LOS
    1 = UMi-NLOS
    2 = UMa-LOS
    3 = UMa-NLOS
    4 = RMa-LOS
    5 = RMa-NLOS
    6 = Indoor-Office-LOS
    7 = Indoor-Office-NLOS

All geometry tensors use final dimension 3 and support arbitrary batch shape.
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Sequence
import math
import time
import numpy as np
import torch


SCENARIO_NAMES = (
    "UMi-LOS",
    "UMi-NLOS",
    "UMa-LOS",
    "UMa-NLOS",
    "RMa-LOS",
    "RMa-NLOS",
    "Indoor-Office-LOS",
    "Indoor-Office-NLOS",
)

SCENARIO_TO_ID = {name: i for i, name in enumerate(SCENARIO_NAMES)}


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool):
    return torch.float64 if parity else torch.float32


def scenario_names_to_ids(names: Sequence[str], *, device=None) -> torch.Tensor:
    vals = [SCENARIO_TO_ID[str(x)] for x in names]
    return torch.as_tensor(vals, dtype=torch.long, device=_device(device))


@dataclass
class GeometryBatch:
    ris: torch.Tensor
    gnb: torch.Tensor
    ue: torch.Tensor
    gnb2ris: torch.Tensor
    ris2gnb: torch.Tensor
    dist_gnb2ris: torch.Tensor
    ris2ue: torch.Tensor
    ue2ris: torch.Tensor
    dist_ris2ue: torch.Tensor


@dataclass
class LSPBatch:
    mu_K: torch.Tensor
    sigma_K: torch.Tensor
    mu_XPR: torch.Tensor
    sigma_XPR: torch.Tensor
    mu_ASA: torch.Tensor
    sigma_ASA: torch.Tensor
    mu_ZSA: torch.Tensor
    sigma_ZSA: torch.Tensor
    mu_ASD: torch.Tensor
    sigma_ASD: torch.Tensor
    mu_ZSD: torch.Tensor
    sigma_ZSD: torch.Tensor
    c_ASA: torch.Tensor
    c_ZSA: torch.Tensor
    c_ASD: torch.Tensor
    c_ZSD: torch.Tensor
    mu_offset_ZOD: torch.Tensor
    M: torch.Tensor
    L: torch.Tensor
    isLOS: torch.Tensor
    scenario_id: torch.Tensor
    K_linear: torch.Tensor


@torch.inference_mode()
def generate_geometry_batch(
    ris,
    gnb,
    ue,
    *,
    device=None,
    parity: bool = False,
) -> GeometryBatch:
    """
    Exact MATLAB mapping:

        gnb2ris = ris - gnb
        ris2gnb = gnb - ris
        dist_gnb2ris = norm(gnb2ris)

        ris2ue = ue - ris
        ue2ris = ris - ue
        dist_ris2ue = norm(ris2ue)
    """
    dev = _device(device)
    rd = _rdtype(parity)

    ris = torch.as_tensor(ris, dtype=rd, device=dev)
    gnb = torch.as_tensor(gnb, dtype=rd, device=dev)
    ue  = torch.as_tensor(ue,  dtype=rd, device=dev)

    if ris.shape[-1] != 3 or gnb.shape[-1] != 3 or ue.shape[-1] != 3:
        raise ValueError("ris/gnb/ue must have final dimension 3")

    ris, gnb, ue = torch.broadcast_tensors(ris, gnb, ue)

    gnb2ris = ris - gnb
    ris2gnb = gnb - ris
    dist_gnb2ris = torch.linalg.vector_norm(gnb2ris, dim=-1)

    ris2ue = ue - ris
    ue2ris = ris - ue
    dist_ris2ue = torch.linalg.vector_norm(ris2ue, dim=-1)

    return GeometryBatch(
        ris=ris,
        gnb=gnb,
        ue=ue,
        gnb2ris=gnb2ris,
        ris2gnb=ris2gnb,
        dist_gnb2ris=dist_gnb2ris,
        ris2ue=ris2ue,
        ue2ris=ue2ris,
        dist_ris2ue=dist_ris2ue,
    )


def _full(shape, value, *, dtype, device):
    return torch.full(shape, float(value), dtype=dtype, device=device)


@torch.inference_mode()
def generate_lsp_batch(
    scenario_id,
    fc,
    a_vector,
    d_vector,
    *,
    device=None,
    parity: bool = False,
) -> LSPBatch:
    """
    Exact vectorized port of the supplied MATLAB generate_lsp.

    Important: hUT and dh intentionally follow the current project convention:
        dh  = abs(a_vector[...,2])
        hUT = a_vector[...,2]

    We do not reinterpret these as absolute UE heights.
    """
    dev = _device(device)
    rd = _rdtype(parity)

    sid = torch.as_tensor(scenario_id, dtype=torch.long, device=dev)
    fc = torch.as_tensor(fc, dtype=rd, device=dev)
    a = torch.as_tensor(a_vector, dtype=rd, device=dev)
    d = torch.as_tensor(d_vector, dtype=rd, device=dev)

    if a.shape[-1] != 3 or d.shape[-1] != 3:
        raise ValueError("a_vector and d_vector must have final dimension 3")

    # Broadcast scalar/vector inputs to one batch shape.
    batch_shape = torch.broadcast_shapes(sid.shape, fc.shape, a.shape[:-1], d.shape[:-1])
    sid = sid.expand(batch_shape)
    fc = fc.expand(batch_shape)
    a = a.expand(*batch_shape, 3)
    d = d.expand(*batch_shape, 3)

    if torch.any((sid < 0) | (sid >= len(SCENARIO_NAMES))):
        raise ValueError("scenario_id out of range [0,7]")
    if torch.any(fc <= 0):
        raise ValueError("fc must be > 0")

    fcG = fc / 1e9
    L1 = torch.log10(1.0 + fcG)
    L0 = torch.log10(fcG)

    d2D = torch.linalg.vector_norm(d[..., :2], dim=-1)
    d2Dk = d2D / 1000.0
    dh = torch.abs(a[..., 2])
    hUT = a[..., 2]

    shp = batch_shape

    # Allocate outputs.
    mu_K = torch.zeros(shp, dtype=rd, device=dev)
    sigma_K = torch.zeros_like(mu_K)
    mu_XPR = torch.zeros_like(mu_K)
    sigma_XPR = torch.zeros_like(mu_K)
    mu_ASA = torch.zeros_like(mu_K)
    sigma_ASA = torch.zeros_like(mu_K)
    mu_ZSA = torch.zeros_like(mu_K)
    sigma_ZSA = torch.zeros_like(mu_K)
    mu_ASD = torch.zeros_like(mu_K)
    sigma_ASD = torch.zeros_like(mu_K)
    mu_ZSD = torch.zeros_like(mu_K)
    sigma_ZSD = torch.zeros_like(mu_K)
    c_ASA = torch.zeros_like(mu_K)
    c_ZSA = torch.zeros_like(mu_K)
    c_ASD = torch.zeros_like(mu_K)
    mu_offset_ZOD = torch.zeros_like(mu_K)
    M = torch.zeros(shp, dtype=torch.int64, device=dev)
    isLOS = torch.zeros(shp, dtype=torch.bool, device=dev)

    def put(mask, tensor, value):
        if torch.is_tensor(value):
            tensor[mask] = value[mask]
        else:
            tensor[mask] = value

    # 0: UMi-LOS
    m = sid == 0
    put(m,mu_K,9); put(m,sigma_K,5)
    put(m,mu_XPR,9); put(m,sigma_XPR,3)
    put(m,mu_ASA,-0.08*L1 + 1.73); put(m,sigma_ASA,0.014*L1 + 0.28)
    put(m,mu_ZSA,-0.10*L1 + 0.73); put(m,sigma_ZSA,-0.04*L1 + 0.34)
    put(m,mu_ASD,-0.05*L1 + 1.21); put(m,sigma_ASD,0.41)
    put(m,mu_ZSD,torch.maximum(_full(shp,-0.21,dtype=rd,device=dev),
                               -14.8*d2Dk + 0.01*dh + 0.83))
    put(m,sigma_ZSD,0.35)
    put(m,c_ASA,17); put(m,c_ZSA,7); put(m,c_ASD,3)
    put(m,mu_offset_ZOD,0); M[m]=12; isLOS[m]=True

    # 1: UMi-NLOS
    m = sid == 1
    put(m,mu_K,0); put(m,sigma_K,0)
    put(m,mu_XPR,8); put(m,sigma_XPR,3)
    put(m,mu_ASA,-0.08*L1 + 1.81); put(m,sigma_ASA,0.05*L1 + 0.30)
    put(m,mu_ZSA,-0.04*L1 + 0.92); put(m,sigma_ZSA,-0.07*L1 + 0.41)
    put(m,mu_ASD,-0.23*L1 + 1.53); put(m,sigma_ASD,0.11*L1 + 0.33)
    put(m,mu_ZSD,torch.maximum(_full(shp,-0.5,dtype=rd,device=dev),
                               -3.1*d2Dk + 0.01*torch.maximum(hUT,torch.zeros_like(hUT)) + 0.2))
    put(m,sigma_ZSD,0.35)
    put(m,c_ASA,22); put(m,c_ZSA,7); put(m,c_ASD,10)
    put(m,mu_offset_ZOD,
        -torch.pow(
            torch.tensor(10.0,dtype=rd,device=dev),
            -1.5*torch.log10(torch.maximum(_full(shp,10,dtype=rd,device=dev),d2D)) + 3.3
        ))
    M[m]=19

    # 2: UMa-LOS
    m = sid == 2
    put(m,mu_K,9); put(m,sigma_K,3.5)
    put(m,mu_XPR,8); put(m,sigma_XPR,4)
    put(m,mu_ASA,1.81); put(m,sigma_ASA,0.20)
    put(m,mu_ZSA,0.95); put(m,sigma_ZSA,0.16)
    put(m,mu_ASD,1.06 + 0.1114*L0); put(m,sigma_ASD,0.28)
    put(m,mu_ZSD,torch.maximum(_full(shp,-0.5,dtype=rd,device=dev),
                               -2.1*d2Dk - 0.01*(hUT-1.5) + 0.75))
    put(m,sigma_ZSD,0.40)
    put(m,c_ASA,11); put(m,c_ZSA,7); put(m,c_ASD,5)
    put(m,mu_offset_ZOD,0); M[m]=12; isLOS[m]=True

    # 3: UMa-NLOS
    m = sid == 3
    put(m,mu_K,0); put(m,sigma_K,0)
    put(m,mu_XPR,7); put(m,sigma_XPR,3)
    put(m,mu_ASA,2.08 - 0.27*L0); put(m,sigma_ASA,0.11)
    put(m,mu_ZSA,-0.3236*L0 + 1.512); put(m,sigma_ZSA,0.16)
    put(m,mu_ASD,1.5 - 0.1144*L0); put(m,sigma_ASD,0.28)
    put(m,mu_ZSD,torch.maximum(_full(shp,-0.5,dtype=rd,device=dev),
                               -2.1*d2Dk - 0.01*(hUT-1.5) + 0.9))
    put(m,sigma_ZSD,0.49)
    put(m,c_ASA,15); put(m,c_ZSA,7); put(m,c_ASD,2)
    a_f = 0.208*L0 - 0.782
    b_f = 25.0
    c_f = -0.13*L0 + 2.03
    e_f = 7.66*L0 - 5.96
    put(m,mu_offset_ZOD,
        e_f - torch.pow(
            torch.tensor(10.0,dtype=rd,device=dev),
            a_f*torch.log10(torch.maximum(_full(shp,b_f,dtype=rd,device=dev),d2D))
            + c_f - 0.07*(hUT-1.5)
        ))
    M[m]=20

    # 4: RMa-LOS
    m = sid == 4
    put(m,mu_K,7); put(m,sigma_K,4)
    put(m,mu_XPR,12); put(m,sigma_XPR,4)
    put(m,mu_ASA,1.52); put(m,sigma_ASA,0.24)
    put(m,mu_ZSA,0.47); put(m,sigma_ZSA,0.40)
    put(m,mu_ASD,0.90); put(m,sigma_ASD,0.38)
    put(m,mu_ZSD,torch.maximum(_full(shp,-1,dtype=rd,device=dev),
                               -0.17*d2Dk - 0.01*(hUT-1.5) + 0.22))
    put(m,sigma_ZSD,0.34)
    put(m,c_ASA,3); put(m,c_ZSA,3); put(m,c_ASD,2)
    put(m,mu_offset_ZOD,0); M[m]=11; isLOS[m]=True

    # 5: RMa-NLOS
    m = sid == 5
    put(m,mu_K,0); put(m,sigma_K,0)
    put(m,mu_XPR,7); put(m,sigma_XPR,3)
    put(m,mu_ASA,1.52); put(m,sigma_ASA,0.13)
    put(m,mu_ZSA,0.58); put(m,sigma_ZSA,0.37)
    put(m,mu_ASD,0.95); put(m,sigma_ASD,0.45)
    put(m,mu_ZSD,torch.maximum(_full(shp,-1,dtype=rd,device=dev),
                               -0.19*d2Dk - 0.01*(hUT-1.5) + 0.28))
    put(m,sigma_ZSD,0.30)
    put(m,c_ASA,3); put(m,c_ZSA,3); put(m,c_ASD,2)
    eps_val = torch.finfo(rd).eps
    den = torch.maximum(d2D,_full(shp,eps_val,dtype=rd,device=dev))
    put(m,mu_offset_ZOD,
        torch.atan((35.0-3.5)/den) - torch.atan((35.0-1.5)/den))
    M[m]=10

    # 6: Indoor-Office-LOS
    m = sid == 6
    put(m,mu_K,7); put(m,sigma_K,4)
    put(m,mu_XPR,11); put(m,sigma_XPR,4)
    put(m,mu_ASA,-0.19*L1 + 1.781); put(m,sigma_ASA,0.12*L1 + 0.119)
    put(m,mu_ZSA,-0.26*L1 + 1.44); put(m,sigma_ZSA,-0.04*L1 + 0.264)
    put(m,mu_ASD,1.60); put(m,sigma_ASD,0.18)
    put(m,mu_ZSD,-1.43*L1 + 2.228); put(m,sigma_ZSD,0.13*L1 + 0.30)
    put(m,c_ASA,8); put(m,c_ZSA,9); put(m,c_ASD,5)
    put(m,mu_offset_ZOD,0); M[m]=15; isLOS[m]=True

    # 7: Indoor-Office-NLOS
    m = sid == 7
    put(m,mu_K,0); put(m,sigma_K,0)
    put(m,mu_XPR,10); put(m,sigma_XPR,4)
    put(m,mu_ASA,-0.11*L1 + 1.863); put(m,sigma_ASA,0.12*L1 + 0.059)
    put(m,mu_ZSA,-0.15*L1 + 1.387); put(m,sigma_ZSA,-0.09*L1 + 0.746)
    put(m,mu_ASD,1.62); put(m,sigma_ASD,0.25)
    put(m,mu_ZSD,1.08); put(m,sigma_ZSD,0.36)
    put(m,c_ASA,11); put(m,c_ZSA,9); put(m,c_ASD,5)
    put(m,mu_offset_ZOD,0); M[m]=19

    c_ZSD = (3.0/8.0) * torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        mu_ZSD
    )

    L = torch.full(shp,20,dtype=torch.int64,device=dev)

    K_linear = torch.where(
        isLOS,
        torch.pow(torch.tensor(10.0,dtype=rd,device=dev),mu_K/10.0),
        torch.zeros_like(mu_K),
    )

    return LSPBatch(
        mu_K=mu_K, sigma_K=sigma_K,
        mu_XPR=mu_XPR, sigma_XPR=sigma_XPR,
        mu_ASA=mu_ASA, sigma_ASA=sigma_ASA,
        mu_ZSA=mu_ZSA, sigma_ZSA=sigma_ZSA,
        mu_ASD=mu_ASD, sigma_ASD=sigma_ASD,
        mu_ZSD=mu_ZSD, sigma_ZSD=sigma_ZSD,
        c_ASA=c_ASA, c_ZSA=c_ZSA, c_ASD=c_ASD, c_ZSD=c_ZSD,
        mu_offset_ZOD=mu_offset_ZOD,
        M=M, L=L, isLOS=isLOS, scenario_id=sid, K_linear=K_linear,
    )


def _rel_fro(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()),np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel())/den)


def compare_stage67_golden_csv(
    csv_path: str,
    *,
    device=None,
    parity: bool = True,
) -> Dict[str,float]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    dev = _device(device)

    ris = df[["ris_x","ris_y","ris_z"]].to_numpy()
    gnb = df[["gnb_x","gnb_y","gnb_z"]].to_numpy()
    ue  = df[["ue_x","ue_y","ue_z"]].to_numpy()

    geom = generate_geometry_batch(
        ris,gnb,ue,device=dev,parity=parity
    )

    sid_br = scenario_names_to_ids(df["scenario_BR"].astype(str).tolist(),device=dev)
    sid_ru = scenario_names_to_ids(df["scenario_RU"].astype(str).tolist(),device=dev)

    fc = df["fc"].to_numpy()

    lsp_br = generate_lsp_batch(
        sid_br,fc,geom.ris2gnb,geom.gnb2ris,
        device=dev,parity=parity
    )
    lsp_ru = generate_lsp_batch(
        sid_ru,fc,geom.ue2ris,geom.ris2ue,
        device=dev,parity=parity
    )

    metrics = {}

    geom_map = {
        "gnb2ris": geom.gnb2ris,
        "ris2gnb": geom.ris2gnb,
        "dist_gnb2ris": geom.dist_gnb2ris,
        "ris2ue": geom.ris2ue,
        "ue2ris": geom.ue2ris,
        "dist_ris2ue": geom.dist_ris2ue,
    }
    for name,t in geom_map.items():
        got = t.detach().cpu().numpy()
        if got.ndim == 2:
            ref = df[[f"{name}_x",f"{name}_y",f"{name}_z"]].to_numpy()
        else:
            ref = df[name].to_numpy()
        metrics[f"geom_{name}_relFro"] = _rel_fro(got,ref)
        metrics[f"geom_{name}_maxAbs"] = float(np.max(np.abs(got-ref)))

    float_fields = [
        "mu_K","sigma_K","mu_XPR","sigma_XPR",
        "mu_ASA","sigma_ASA","mu_ZSA","sigma_ZSA",
        "mu_ASD","sigma_ASD","mu_ZSD","sigma_ZSD",
        "c_ASA","c_ZSA","c_ASD","c_ZSD","mu_offset_ZOD","K_linear"
    ]
    int_fields = ["M","L"]
    bool_fields = ["isLOS"]

    for prefix,lsp in [("BR",lsp_br),("RU",lsp_ru)]:
        for name in float_fields:
            got = getattr(lsp,name).detach().cpu().numpy()
            ref = df[f"{prefix}_{name}"].to_numpy()
            metrics[f"{prefix}_{name}_relFro"] = _rel_fro(got,ref)
            metrics[f"{prefix}_{name}_maxAbs"] = float(np.max(np.abs(got-ref)))

        for name in int_fields:
            got = getattr(lsp,name).detach().cpu().numpy()
            ref = df[f"{prefix}_{name}"].to_numpy(dtype=np.int64)
            metrics[f"{prefix}_{name}_exact"] = float(np.mean(got==ref))

        for name in bool_fields:
            got = getattr(lsp,name).detach().cpu().numpy().astype(bool)
            ref = df[f"{prefix}_{name}"].to_numpy().astype(bool)
            metrics[f"{prefix}_{name}_exact"] = float(np.mean(got==ref))

    return metrics


@torch.inference_mode()
def benchmark_geometry_lsp(
    *,
    batch_size: int = 1_000_000,
    repeats: int = 10,
    device=None,
) -> Dict[str,Any]:
    dev = _device(device)
    rd = torch.float32

    gen = torch.Generator(device=dev)
    gen.manual_seed(42)

    ris = torch.zeros((batch_size,3),dtype=rd,device=dev)
    gnb = torch.randn((batch_size,3),generator=gen,dtype=rd,device=dev)
    ue = torch.randn((batch_size,3),generator=gen,dtype=rd,device=dev)
    gnb[:,:2] *= 300.0
    ue[:,:2] *= 150.0
    gnb[:,2] = torch.rand((batch_size,),generator=gen,dtype=rd,device=dev)*45.0 + 2.0
    ue[:,2] = torch.rand((batch_size,),generator=gen,dtype=rd,device=dev)*2.5 + 1.0

    sid_br = torch.randint(0,8,(batch_size,),generator=gen,device=dev)
    sid_ru = torch.randint(0,8,(batch_size,),generator=gen,device=dev)
    fc_choices = torch.tensor(
        [2.0e9,2.6e9,3.5e9,3.6e9,4.9e9,6.0e9],
        dtype=rd,device=dev
    )
    pick = torch.randint(0,len(fc_choices),(batch_size,),generator=gen,device=dev)
    fc = fc_choices[pick]

    def run():
        g = generate_geometry_batch(ris,gnb,ue,device=dev,parity=False)
        _ = generate_lsp_batch(sid_br,fc,g.ris2gnb,g.gnb2ris,device=dev,parity=False)
        _ = generate_lsp_batch(sid_ru,fc,g.ue2ris,g.ris2ue,device=dev,parity=False)

    run()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        ts.append(time.perf_counter()-t0)

    med = float(np.median(ts))
    return {
        "device": str(dev),
        "batch_size": int(batch_size),
        "median_seconds": med,
        "banks_per_second": float(batch_size/med),
    }
