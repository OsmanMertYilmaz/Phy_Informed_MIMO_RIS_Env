"""
Stage 8B-2 — GPU-native stochastic RNG + statistical parity.

Stage 8B-1 proved the stochastic channel mathematics with identical random
primitives. Stage 8B-2 removes the MATLAB-exported primitives completely.

PyTorch/CUDA generates:
    XPR
    ASA / ZSA / ASD / ZSD
    per-cluster angular offsets
    per-ray 2x2 polarization phases

Then the already parity-validated Stage-8B1 realization kernel generates H.

Because MATLAB and PyTorch use different RNG streams, equality is now
distributional/statistical rather than sample-by-sample.

Scope of B2:
    - one fixed (W,z) per environment
    - independent GPU random realizations
    - H_BR / H_RU statistics
    - cascaded F -> Feff -> Y statistics
    - direct empirical q05(Y)
    - realization throughput

B3 is intentionally NOT implemented here. B3 will add the production
multi-W / multi-z empirical label engine and label/s benchmark.
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
from scipy.io import loadmat

from ris_env.antenna import ArraySpec, generate_channel_moments_batch
from ris_env.channel_primitives import (
    generate_channel_from_primitives,
    generate_cascaded_ch,
    apply_precoder_empirical,
    empirical_snr_samples,
)


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _rdtype(parity: bool):
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool):
    return torch.complex128 if parity else torch.complex64


@dataclass
class LinkConfig:
    tx_spec: ArraySpec
    rx_spec: ArraySpec
    a_vector: np.ndarray
    d_vector: np.ndarray
    fc: float
    K: float
    is_los: bool
    M: int
    L: int

    mu_XPR: float
    sigma_XPR: float

    mu_ASA: float
    sigma_ASA: float
    mu_ZSA: float
    sigma_ZSA: float
    mu_ASD: float
    sigma_ASD: float
    mu_ZSD: float
    sigma_ZSD: float

    c_ASA: float
    c_ZSA: float
    c_ASD: float
    c_ZSD: float
    mu_offset_ZOD: float


@torch.inference_mode()
def sample_native_primitives(
    cfg: LinkConfig,
    N: int,
    *,
    generator: torch.Generator,
    device=None,
    parity: bool = False,
) -> Dict[str,torch.Tensor]:
    """
    Distributional port of the random draws at the beginning and inside
    generate_channel_train.m.

    RNG sequence is PyTorch-native and is NOT expected to match MATLAB.
    """
    dev = _device(device)
    rd = _rdtype(parity)

    def randn(*shape):
        return torch.randn(
            shape,dtype=rd,device=dev,generator=generator
        )

    def rand(*shape):
        return torch.rand(
            shape,dtype=rd,device=dev,generator=generator
        )

    XPR = cfg.mu_XPR + cfg.sigma_XPR * randn(N,cfg.M*cfg.L)

    ASAv = torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        cfg.mu_ASA + cfg.sigma_ASA*randn(N)
    ).clamp_max(104.0)

    ZSAv = torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        cfg.mu_ZSA + cfg.sigma_ZSA*randn(N)
    ).clamp_max(52.0)

    ASDv = torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        cfg.mu_ASD + cfg.sigma_ASD*randn(N)
    ).clamp_max(104.0)

    ZSDv = torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        cfg.mu_ZSD + cfg.sigma_ZSD*randn(N)
    ).clamp_max(52.0)

    offsets = torch.empty(
        (N,cfg.M,4),dtype=rd,device=dev
    )
    offsets[:,:,0] = randn(N,cfg.M) * ASAv[:,None] / 7.0
    offsets[:,:,1] = randn(N,cfg.M) * ZSAv[:,None] / 7.0
    offsets[:,:,2] = randn(N,cfg.M) * ASDv[:,None] / 7.0
    offsets[:,:,3] = randn(N,cfg.M) * ZSDv[:,None] / 7.0

    Phi = -math.pi + 2.0*math.pi*rand(N,cfg.M,cfg.L,2,2)

    return {
        "XPR":XPR,
        "ASAv":ASAv,
        "ZSAv":ZSAv,
        "ASDv":ASDv,
        "ZSDv":ZSDv,
        "cluster_offsets":offsets,
        "Phi":Phi,
    }


@torch.inference_mode()
def generate_native_link_chunk(
    cfg: LinkConfig,
    N: int,
    *,
    generator: torch.Generator,
    device=None,
    parity: bool = False,
) -> torch.Tensor:
    p = sample_native_primitives(
        cfg,N,generator=generator,device=device,parity=parity
    )

    return generate_channel_from_primitives(
        tx_spec=cfg.tx_spec,
        rx_spec=cfg.rx_spec,
        a_vector=cfg.a_vector,
        d_vector=cfg.d_vector,
        fc=cfg.fc,
        K=cfg.K,
        is_los=cfg.is_los,
        M=cfg.M,
        L=cfg.L,
        c_ASA=cfg.c_ASA,
        c_ZSA=cfg.c_ZSA,
        c_ASD=cfg.c_ASD,
        c_ZSD=cfg.c_ZSD,
        mu_offset_ZOD=cfg.mu_offset_ZOD,
        XPR=p["XPR"],
        ASAv=p["ASAv"],
        ZSAv=p["ZSAv"],
        ASDv=p["ASDv"],
        ZSDv=p["ZSDv"],
        cluster_offsets=p["cluster_offsets"],
        Phi=p["Phi"],
        device=device,
        parity=parity,
    )


class LinkStatsAccumulator:
    def __init__(self, mu: torch.Tensor, sigma2: float):
        self.mu = mu
        self.sigma2 = float(sigma2)
        self.count = 0
        self.sum_h = torch.zeros_like(mu,dtype=torch.complex128)
        self.sum_abs2 = 0.0
        self.sum_abs4 = 0.0
        self.sum_real2 = 0.0
        self.sum_imag2 = 0.0
        self.sum_pseudo = 0.0 + 0.0j

    @torch.inference_mode()
    def update(self,H: torch.Tensor):
        Hd = H.to(torch.complex128)
        mu = self.mu.to(torch.complex128)
        C = Hd - mu[None,:,:]

        self.count += H.shape[0]
        self.sum_h += Hd.sum(dim=0).cpu()

        self.sum_abs2 += float(
            torch.sum(torch.abs(C)**2).detach().cpu()
        )
        self.sum_abs4 += float(
            torch.sum(torch.abs(C)**4).detach().cpu()
        )
        self.sum_real2 += float(
            torch.sum(C.real**2).detach().cpu()
        )
        self.sum_imag2 += float(
            torch.sum(C.imag**2).detach().cpu()
        )
        self.sum_pseudo += complex(
            torch.sum(C*C).detach().cpu().item()
        )

    def finalize(self) -> Dict[str,float]:
        n_elem = int(np.prod(self.mu.shape))
        denom_count = self.count*n_elem

        mean_emp = self.sum_h / self.count
        mu_cpu = self.mu.to(torch.complex128).cpu()

        sig = max(self.sigma2,np.finfo(float).eps)

        mean_leak = float(
            torch.linalg.vector_norm(mean_emp-mu_cpu)
            / math.sqrt(n_elem*sig)
        )

        return {
            "meanLeakNorm":mean_leak,
            "varianceRatioMean":self.sum_abs2/(denom_count*sig),
            "fourthMomentRatio":self.sum_abs4/(denom_count*sig*sig),
            "realVarianceRatio":2.0*self.sum_real2/(denom_count*sig),
            "imagVarianceRatio":2.0*self.sum_imag2/(denom_count*sig),
            "pseudoCovAbsRatio":abs(self.sum_pseudo/denom_count)/sig,
        }


def summarize_y(y: np.ndarray) -> Dict[str,float]:
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    return {
        "Y_mean":float(np.mean(y)),
        "Y_var":float(np.var(y,ddof=0)),
        "Y_q01":float(np.quantile(y,0.01)),
        "Y_q05":float(np.quantile(y,0.05)),
        "Y_q10":float(np.quantile(y,0.10)),
        "Y_q50":float(np.quantile(y,0.50)),
        "Y_q90":float(np.quantile(y,0.90)),
        "Y_q99":float(np.quantile(y,0.99)),
    }


def _sc(M,name):
    return float(np.asarray(M[name]).reshape(()))


def _si(M,name):
    return int(np.asarray(M[name]).reshape(()))


def _sb(M,name):
    return bool(_si(M,name))


def _sstr(M,name):
    x=np.asarray(M[name])
    if x.dtype.kind in ("U","S"):
        return "".join(x.reshape(-1).tolist()).strip()
    y=x.squeeze()
    return str(y.item() if hasattr(y,"item") else y).strip()


def load_stage8b2_case(mat_path: str):
    M=loadmat(mat_path,squeeze_me=False)

    nT1=_si(M,"nT1"); nT2=_si(M,"nT2")
    nR1=_si(M,"nR1"); nR2=_si(M,"nR2")
    nRISx=_si(M,"nRISx"); nRISy=_si(M,"nRISy")
    fc=_sc(M,"fc")

    def cfg(prefix,tx_spec,rx_spec,a_name,d_name):
        return LinkConfig(
            tx_spec=tx_spec,
            rx_spec=rx_spec,
            a_vector=np.asarray(M[a_name]).reshape(-1),
            d_vector=np.asarray(M[d_name]).reshape(-1),
            fc=fc,
            K=_sc(M,f"{prefix}_K"),
            is_los=_sb(M,f"{prefix}_isLOS"),
            M=_si(M,f"{prefix}_M"),
            L=_si(M,f"{prefix}_L"),

            mu_XPR=_sc(M,f"{prefix}_mu_XPR"),
            sigma_XPR=_sc(M,f"{prefix}_sigma_XPR"),

            mu_ASA=_sc(M,f"{prefix}_mu_ASA"),
            sigma_ASA=_sc(M,f"{prefix}_sigma_ASA"),
            mu_ZSA=_sc(M,f"{prefix}_mu_ZSA"),
            sigma_ZSA=_sc(M,f"{prefix}_sigma_ZSA"),
            mu_ASD=_sc(M,f"{prefix}_mu_ASD"),
            sigma_ASD=_sc(M,f"{prefix}_sigma_ASD"),
            mu_ZSD=_sc(M,f"{prefix}_mu_ZSD"),
            sigma_ZSD=_sc(M,f"{prefix}_sigma_ZSD"),

            c_ASA=_sc(M,f"{prefix}_c_ASA"),
            c_ZSA=_sc(M,f"{prefix}_c_ZSA"),
            c_ASD=_sc(M,f"{prefix}_c_ASD"),
            c_ZSD=_sc(M,f"{prefix}_c_ZSD"),
            mu_offset_ZOD=_sc(M,f"{prefix}_mu_offset_ZOD"),
        )

    br=cfg(
        "BR",ArraySpec(nT1,nT2),ArraySpec(nRISx,nRISy),
        "ris2gnb","gnb2ris"
    )
    ru=cfg(
        "RU",ArraySpec(nRISx,nRISy),ArraySpec(nR1,nR2),
        "ue2ris","ris2ue"
    )

    reference={
        k:_sc(M,k)
        for k in [
            "BR_meanLeakNorm","BR_varianceRatioMean",
            "BR_fourthMomentRatio","BR_realVarianceRatio",
            "BR_imagVarianceRatio","BR_pseudoCovAbsRatio",
            "RU_meanLeakNorm","RU_varianceRatioMean",
            "RU_fourthMomentRatio","RU_realVarianceRatio",
            "RU_imagVarianceRatio","RU_pseudoCovAbsRatio",
            "Y_mean","Y_var","Y_q01","Y_q05","Y_q10",
            "Y_q50","Y_q90","Y_q99",
        ]
    }

    return {
        "raw":M,
        "scenario":_sstr(M,"scenario"),
        "N_ref":_si(M,"N_ref"),
        "br":br,
        "ru":ru,
        "W":np.asarray(M["W"]).reshape(-1,1),
        "gamma":np.asarray(M["gamma"]).reshape(-1),
        "reference":reference,
        "nT":2*nT1*nT2,
        "nR":2*nR1*nR2,
        "nRIS":2*nRISx*nRISy,
    }


@torch.inference_mode()
def run_native_case(
    mat_path: str,
    *,
    N_python: int = 16384,
    chunk_size: int = 1024,
    device=None,
    parity: bool = False,
    seed_br: int = 12001,
    seed_ru: int = 13001,
) -> Dict[str,Any]:
    case=load_stage8b2_case(mat_path)
    dev=_device(device)

    br=case["br"]; ru=case["ru"]

    # Analytic target means/variances are already validated in Stage 1.
    br_m=generate_channel_moments_batch(
        tx_spec=br.tx_spec,rx_spec=br.rx_spec,
        a_vectors=br.a_vector[None,:],
        d_vectors=br.d_vector[None,:],
        carrier_frequency=br.fc,
        K=br.K,
        mu_xpr=br.mu_XPR,
        sigma_xpr=br.sigma_XPR,
        device=dev,parity=parity,
    )
    ru_m=generate_channel_moments_batch(
        tx_spec=ru.tx_spec,rx_spec=ru.rx_spec,
        a_vectors=ru.a_vector[None,:],
        d_vectors=ru.d_vector[None,:],
        carrier_frequency=ru.fc,
        K=ru.K,
        mu_xpr=ru.mu_XPR,
        sigma_xpr=ru.sigma_XPR,
        device=dev,parity=parity,
    )

    br_acc=LinkStatsAccumulator(
        br_m["muH"][0],
        float(br_m["sigma2H"][0].detach().cpu())
    )
    ru_acc=LinkStatsAccumulator(
        ru_m["muH"][0],
        float(ru_m["sigma2H"][0].detach().cpu())
    )

    gen_br=torch.Generator(device=dev)
    gen_ru=torch.Generator(device=dev)
    gen_br.manual_seed(int(seed_br))
    gen_ru.manual_seed(int(seed_ru))

    W=torch.as_tensor(
        case["W"],dtype=_cdtype(parity),device=dev
    )
    gamma=torch.as_tensor(
        case["gamma"],dtype=_cdtype(parity),device=dev
    )

    y_parts=[]

    _sync(dev)
    t0=time.perf_counter()

    done=0
    while done < N_python:
        n=min(chunk_size,N_python-done)

        HBR=generate_native_link_chunk(
            br,n,generator=gen_br,
            device=dev,parity=parity
        )
        HRU=generate_native_link_chunk(
            ru,n,generator=gen_ru,
            device=dev,parity=parity
        )

        br_acc.update(HBR)
        ru_acc.update(HRU)

        F=generate_cascaded_ch(HBR,HRU,gamma)
        Feff=apply_precoder_empirical(F,W)
        Y=empirical_snr_samples(Feff)

        y_parts.append(Y.detach().cpu().numpy())

        done += n

        del HBR,HRU,F,Feff,Y

    _sync(dev)
    elapsed=time.perf_counter()-t0

    y=np.concatenate(y_parts)

    out={
        "scenario":case["scenario"],
        "N_ref":case["N_ref"],
        "N_python":int(N_python),
        "nT":case["nT"],
        "nR":case["nR"],
        "nRIS":case["nRIS"],
        "seconds":float(elapsed),
        "realization_pairs_per_second":float(N_python/elapsed),
    }

    for k,v in br_acc.finalize().items():
        out[f"BR_{k}"]=v
    for k,v in ru_acc.finalize().items():
        out[f"RU_{k}"]=v

    out.update(summarize_y(y))

    # Independent-RNG reference comparisons.
    ref=case["reference"]

    for key in [
        "BR_varianceRatioMean","BR_fourthMomentRatio",
        "BR_realVarianceRatio","BR_imagVarianceRatio",
        "BR_pseudoCovAbsRatio",
        "RU_varianceRatioMean","RU_fourthMomentRatio",
        "RU_realVarianceRatio","RU_imagVarianceRatio",
        "RU_pseudoCovAbsRatio",
        "Y_mean","Y_var","Y_q01","Y_q05","Y_q10","Y_q50","Y_q90","Y_q99",
    ]:
        den=max(abs(ref[key]),np.finfo(float).eps)
        out[f"{key}_ref"]=ref[key]
        out[f"{key}_relDiff"]=abs(out[key]-ref[key])/den

    # Mean leakage is a finite-sample quantity whose expected value shrinks
    # with N; comparing two different N values by relative difference is not
    # meaningful, so expose both directly.
    out["BR_meanLeakNorm_ref"]=ref["BR_meanLeakNorm"]
    out["RU_meanLeakNorm_ref"]=ref["RU_meanLeakNorm"]

    return out
