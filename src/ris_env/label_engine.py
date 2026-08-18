"""
Stage 8B-3B — Production-style multi-W / multi-z Symmetric Gamma-Gamma label engine.

Locked statistical target
-------------------------
For every physical candidate (environment, W_k, z_c):

    analytic muSNR(k,c)
    empirical varEmp(k,c) from N Monte-Carlo channel realizations

are mapped to the project's Symmetric Gamma-Gamma q05:

    q05GG(k,c) = GG_q05(muSNR(k,c), varEmp(k,c))

N is expected to be chosen by Stage 8B-3A. Current locked benchmark value:
    N = 64_000

Core performance idea
---------------------
The BR/RU stochastic channel realizations are generated ONCE per MC chunk
and reused for all W and z candidates.

For one MC chunk:

    HBR[n,i,t]
    HRU[n,r,i]

For a W chunk:

    u[n,k,i] = sum_t HBR[n,i,t] W[k,t]

Then define the reusable cascaded basis:

    B[n,k,r,i] = HRU[n,r,i] u[n,k,i]

For a z/gamma chunk:

    Feff[n,k,r,c] = sum_i B[n,k,r,i] gamma[c,i]

implemented as a GEMM:

    B_flat @ gamma.T

Then:

    Y[n,k,c] = sum_r |Feff[n,k,r,c]|^2

The full [N,K,C] Y tensor is NEVER stored. Only

    sum(Y), sum(Y^2)

are accumulated, so memory does not scale with total N.

Analytic mean
-------------
The old project label q05GammaGammaFit uses analytic muSNR + empirical varEmp.
We preserve that convention exactly. A lightweight W-state is used that builds
only UBR + effective second-moment kernels; Cmat/sigma2Wick are not calculated
because B3-B labels do not need them.

Lookup
------
The Symmetric-GG q05 lookup is reconstructed from the corrected dataset by the
Stage-8B3A helper. GPU interpolation matches the previous Teacher notebook's
linear interpolation in log(CV^2).
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import math
import time

import numpy as np
import pandas as pd
import torch

from ris_env.validation import (
    row_to_link_configs,
)
from ris_env.channel_realizations import generate_native_link_chunk
from ris_env.channel_primitives import (
    generate_cascaded_ch,
    apply_precoder_empirical,
    empirical_snr_samples,
)
from ris_env.codebook import (
    generate_codebook_rank1,
    flatten_codebook_matlab_loop_order,
)
from ris_env.ris_response import generate_ris_response_from_z
from ris_env.snr_statistics import prepare_w_state, evaluate_gamma_batch
from ris_env.environment import BankInput, build_deterministic_bank
from ris_env.gamma_gamma import (
    GGQ05Lookup,
    build_gg_lookup_from_dataset,
)


COMPLEX_BYTES_C64 = 8
REAL_BYTES_F32 = 4


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def row_to_bank_input(row: pd.Series) -> BankInput:
    """
    Deterministic Stage-8 bank metadata from one dataset row.

    WIdx is only a dummy valid codeword here; B3-B reuses static_env and
    evaluates the supplied W batch separately.
    """
    return BankInput(
        scenario_br=str(row.scenario_BR),
        scenario_ru=str(row.scenario_RU),
        fc=float(row.fc),
        ris=(float(row.ris_x),float(row.ris_y),float(row.ris_z)),
        gnb=(float(row.gnb_x),float(row.gnb_y),float(row.gnb_z)),
        ue=(float(row.ue_x),float(row.ue_y),float(row.ue_z)),
        nT1=int(row.nT1),
        nT2=int(row.nT2),
        nR1=int(row.nR1),
        nR2=int(row.nR2),
        nRIS_x=int(row.nRIS1),
        nRIS_y=int(row.nRIS2),
        WIdx=(1,1,1),
    )


@torch.inference_mode()
def build_w_candidate_pool(
    row: pd.Series,
    *,
    k_w: int=32,
    device=None,
    parity: bool=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic, unique Type-I rank-1 codeword pool.

    Returns:
        W     [K,nT]
        WIdx  [K,3], MATLAB 1-based [i11,i12,i2]

    The benchmark selects approximately evenly-spaced codewords from the full
    codebook so that it does not benchmark 32 near-identical entries.
    """
    dev=_device(device)

    cb=generate_codebook_rank1(
        2,
        int(row.nT1),
        int(row.nT2),
        cb_mode=1,
        nl=1,
        device=dev,
        parity=parity,
    )
    Wflat,idx=flatten_codebook_matlab_loop_order(cb)  # [nT,Ncb], [Ncb,3]
    ncb=int(Wflat.shape[1])

    if k_w > ncb:
        raise ValueError(
            f"Requested k_w={k_w}, but this array has only {ncb} unique "
            "Type-I rank-1 codewords."
        )

    if k_w == ncb:
        sel=torch.arange(ncb,device=dev,dtype=torch.long)
    else:
        # floor(j*ncb/k_w), j=0..K-1 is strictly increasing when K<=Ncb.
        sel=torch.div(
            torch.arange(k_w,device=dev,dtype=torch.long)*ncb,
            k_w,
            rounding_mode="floor",
        )

    W=Wflat[:,sel].T.contiguous()
    WIdx=idx[sel].contiguous()

    return W,WIdx


@torch.inference_mode()
def build_z_candidate_pool(
    n_ris: int,
    *,
    c_z: int=512,
    seed: int=20260818,
    device=None,
    parity: bool=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary RIS candidate pool with four deterministic anchors followed by
    pseudo-random patterns.

    Returns:
        z      [C,nRIS], int64
        gamma  [C,nRIS], complex64/128
    """
    if c_z < 1:
        raise ValueError("c_z must be >=1")
    dev=_device(device)

    # Use a CPU RNG so the candidate set is reproducible across CUDA devices.
    rng=np.random.default_rng(int(seed))
    z=np.empty((c_z,int(n_ris)),dtype=np.int64)

    anchors=[]
    anchors.append(np.zeros(n_ris,dtype=np.int64))
    anchors.append(np.ones(n_ris,dtype=np.int64))
    anchors.append((np.arange(n_ris)%2).astype(np.int64))
    anchors.append(1-(np.arange(n_ris)%2).astype(np.int64))

    used=min(c_z,len(anchors))
    for i in range(used):
        z[i]=anchors[i]

    if c_z > used:
        z[used:]=rng.integers(
            0,2,size=(c_z-used,n_ris),dtype=np.int64
        )

    zt=torch.as_tensor(z,dtype=torch.long,device=dev)
    gamma=generate_ris_response_from_z(
        zt,device=dev,parity=parity
    )["gamma"]

    return zt,gamma


def estimate_candidate_chunk_memory_mb(
    *,
    mc_chunk: int,
    w_chunk: int,
    z_chunk: int,
    n_ris: int,
    n_r: int,
    n_t: int,
) -> Dict[str,float]:
    """
    Approximate retained complex64/float32 tensor sizes for the empirical
    candidate contraction kernel. Generator-internal primitive memory is not
    included.
    """
    N=int(mc_chunk); K=int(w_chunk); C=int(z_chunk)
    I=int(n_ris); R=int(n_r); T=int(n_t)

    bytes_hbr=N*I*T*COMPLEX_BYTES_C64
    bytes_hru=N*R*I*COMPLEX_BYTES_C64
    bytes_u=N*K*I*COMPLEX_BYTES_C64
    bytes_basis=N*K*R*I*COMPLEX_BYTES_C64
    bytes_eff=N*K*R*C*COMPLEX_BYTES_C64
    bytes_y=N*K*C*REAL_BYTES_F32

    out={
        "HBR_MB":bytes_hbr/1024**2,
        "HRU_MB":bytes_hru/1024**2,
        "u_MB":bytes_u/1024**2,
        "basis_MB":bytes_basis/1024**2,
        "eff_MB":bytes_eff/1024**2,
        "Y_MB":bytes_y/1024**2,
    }
    out["approx_total_MB"]=sum(out.values())
    return out


@torch.inference_mode()
def candidate_y_from_channels_batched(
    HBR: torch.Tensor,
    HRU: torch.Tensor,
    W: torch.Tensor,
    gamma: torch.Tensor,
    *,
    w_chunk: int=8,
    z_chunk: int=64,
) -> torch.Tensor:
    """
    Materializing helper used ONLY for small correctness tests.

    Returns:
        Y [N,K,C]

    The production engine below does not materialize full Y across total N.
    """
    if HBR.ndim != 3 or HRU.ndim != 3:
        raise ValueError("HBR/HRU must be rank-3.")
    N,I,T=HBR.shape
    N2,R,I2=HRU.shape
    if N2!=N or I2!=I:
        raise ValueError("HBR/HRU shape mismatch.")

    W=torch.as_tensor(W,dtype=HBR.dtype,device=HBR.device)
    gamma=torch.as_tensor(gamma,dtype=HBR.dtype,device=HBR.device)

    K=W.shape[0]
    C=gamma.shape[0]
    Yout=torch.empty(
        (N,K,C),
        dtype=HBR.real.dtype,
        device=HBR.device,
    )

    for k0 in range(0,K,w_chunk):
        k1=min(k0+w_chunk,K)
        Wc=W[k0:k1]

        # [N,I,T] @ [T,Kw] -> [N,I,Kw] -> [N,Kw,I]
        u=torch.matmul(HBR,Wc.T).permute(0,2,1).contiguous()

        # Reuse for all z chunks:
        # [N,Kw,R,I]
        basis=HRU[:,None,:,:]*u[:,:,None,:]
        basis_flat=basis.reshape(N*(k1-k0)*R,I)

        for c0 in range(0,C,z_chunk):
            c1=min(c0+z_chunk,C)
            gc=gamma[c0:c1]

            eff=(
                basis_flat @ gc.T
            ).reshape(N,k1-k0,R,c1-c0)

            Y=torch.sum(torch.abs(eff)**2,dim=2)
            Yout[:,k0:k1,c0:c1]=Y

    return Yout


@torch.inference_mode()
def validate_batched_contraction(
    row: pd.Series,
    *,
    n_mc: int=32,
    k_w: int=3,
    c_z: int=5,
    device=None,
    parity: bool=False,
) -> Dict[str,float]:
    """
    Compare the optimized multi-W/multi-z contraction against the already
    Stage-8B1-validated naive F -> Feff -> Y path.
    """
    dev=_device(device)
    br,ru=row_to_link_configs(row)

    gen_br=torch.Generator(device=dev)
    gen_ru=torch.Generator(device=dev)
    seed0=int(row.ch_seed)
    gen_br.manual_seed(seed0+50_000_087)
    gen_ru.manual_seed(seed0+60_000_103)

    HBR=generate_native_link_chunk(
        br,n_mc,generator=gen_br,device=dev,parity=parity
    )
    HRU=generate_native_link_chunk(
        ru,n_mc,generator=gen_ru,device=dev,parity=parity
    )

    W,_=build_w_candidate_pool(
        row,k_w=k_w,device=dev,parity=parity
    )
    z,gamma=build_z_candidate_pool(
        int(row.nRIS),c_z=c_z,seed=1234,
        device=dev,parity=parity
    )

    Yfast=candidate_y_from_channels_batched(
        HBR,HRU,W,gamma,
        w_chunk=min(k_w,2),
        z_chunk=min(c_z,3),
    )

    Yref=torch.empty_like(Yfast)

    for k in range(k_w):
        wk=W[k].reshape(-1,1)
        for c in range(c_z):
            F=generate_cascaded_ch(HBR,HRU,gamma[c])
            Feff=apply_precoder_empirical(F,wk)
            Yref[:,k,c]=empirical_snr_samples(Feff).reshape(-1)

    diff=Yfast-Yref
    rel=(
        torch.linalg.vector_norm(diff.reshape(-1))
        / torch.clamp(
            torch.linalg.vector_norm(Yref.reshape(-1)),
            min=torch.finfo(Yref.dtype).tiny,
        )
    )

    return {
        "relative_fro":float(rel.detach().cpu()),
        "max_abs":float(torch.max(torch.abs(diff)).detach().cpu()),
        "mean_abs":float(torch.mean(torch.abs(diff)).detach().cpu()),
    }


@dataclass
class MeanOnlyWState:
    eff_kernel: torch.Tensor  # [nR,nRIS,nRIS]


def _hermitianize(A: torch.Tensor) -> torch.Tensor:
    return 0.5*(A+A.conj().transpose(-2,-1))


@torch.inference_mode()
def prepare_w_mean_only_state(static_env, w) -> MeanOnlyWState:
    """
    Lightweight version of Stage-3 prepare_w_state.

    It builds only what is needed for analytic muSNR:
        UBR
        effective second-moment kernels

    It deliberately skips the much larger Cmat covariance kernels.
    """
    env=static_env
    dev=env.muBR.device
    cd=env.muBR.dtype

    w=torch.as_tensor(w,dtype=cd,device=dev).reshape(-1)
    nRIS,nT=env.muBR.shape
    nR=env.muRU.shape[0]
    LBR=env.rho_RB.shape[2]

    if w.numel()!=nT:
        raise ValueError(f"w length {w.numel()} != nT {nT}")

    SRIS=env.sRIS[:,None]*env.sRIS[None,:]
    ST=env.sT[:,None]*env.sT[None,:]

    c0BR=env.sigma2BR/(2.0*(1.0+env.eKappaBR)*LBR)

    ubar=env.muBR @ w
    U_nlos=torch.zeros((nRIS,nRIS),dtype=cd,device=dev)

    for ell in range(LBR):
        Grx=env.rho_RB[:,:,ell]
        Gtx=env.rho_BR[:,:,ell]

        # Exact MATLAB convention: w.' * G * conj(w)
        s1=torch.sum((w @ Gtx)*w.conj())
        s2=torch.sum((w @ (Gtx*ST))*w.conj())

        GrxPol=Grx*SRIS
        BRdirect=Grx+env.eKappaBR*GrxPol
        BRcross=env.eKappaBR*Grx+GrxPol

        U_nlos=U_nlos+c0BR*(BRdirect*s1+BRcross*s2)

    UBR=_hermitianize(
        ubar[:,None]*ubar.conj()[None,:]+U_nlos
    )

    eff_kernel=torch.empty(
        (nR,nRIS,nRIS),dtype=cd,device=dev
    )

    for r in range(nR):
        mg=env.muRU[r,:]
        ARU=_hermitianize(
            mg[:,None]*mg.conj()[None,:]+env.rho_RUhop
        )
        eff_kernel[r]=ARU*UBR

    return MeanOnlyWState(eff_kernel=eff_kernel)


@torch.inference_mode()
def evaluate_analytic_mu_snr_mean_only(
    state: MeanOnlyWState,
    gamma: torch.Tensor,
    *,
    z_chunk: int=64,
) -> torch.Tensor:
    """
    analytic muSNR for C gamma candidates, output [C].

    muSNR = sum_r E|Feff_r|^2
    """
    K=state.eff_kernel
    dev=K.device
    cd=K.dtype

    g=torch.as_tensor(gamma,dtype=cd,device=dev)
    if g.ndim==1:
        g=g.unsqueeze(0)

    C,I=g.shape
    if K.shape[-1]!=I:
        raise ValueError("gamma / eff_kernel nRIS mismatch.")

    out=torch.empty(C,dtype=K.real.dtype,device=dev)

    for c0 in range(0,C,z_chunk):
        c1=min(c0+z_chunk,C)
        gc=g[c0:c1]

        # q[c,r] = gamma_c^T K_r conj(gamma_c)
        # Use [r,c,i] temporary, no [C,I,I] tensor.
        y=torch.einsum("ci,rin->rcn",gc,K)
        q=torch.sum(y*gc.conj()[None,:,:],dim=-1).T.real
        out[c0:c1]=torch.sum(q,dim=1)

    return out


@torch.inference_mode()
def analytic_mu_snr_multi_w_z(
    static_env,
    W: torch.Tensor,
    gamma: torch.Tensor,
    *,
    z_chunk: int=64,
) -> torch.Tensor:
    """
    Returns analytic muSNR [K,C].
    """
    dev=static_env.muBR.device
    W=torch.as_tensor(W,dtype=static_env.muBR.dtype,device=dev)
    gamma=torch.as_tensor(gamma,dtype=static_env.muBR.dtype,device=dev)

    Knum=W.shape[0]
    C=gamma.shape[0]
    out=torch.empty(
        (Knum,C),
        dtype=static_env.sigma2BR.dtype,
        device=dev,
    )

    for k in range(Knum):
        st=prepare_w_mean_only_state(static_env,W[k])
        out[k]=evaluate_analytic_mu_snr_mean_only(
            st,gamma,z_chunk=z_chunk
        )
        del st

    return out


@torch.inference_mode()
def validate_mean_only_against_stage3(
    static_env,
    W: torch.Tensor,
    gamma: torch.Tensor,
    *,
    k_check: int=2,
    c_check: int=4,
) -> Dict[str,float]:
    """
    Compare lightweight analytic muSNR against the already validated full
    Stage-3 evaluate_gamma_batch path.
    """
    K=min(int(k_check),W.shape[0])
    C=min(int(c_check),gamma.shape[0])

    fast=analytic_mu_snr_multi_w_z(
        static_env,W[:K],gamma[:C],z_chunk=C
    )

    ref=torch.empty_like(fast)
    for k in range(K):
        ws=prepare_w_state(static_env,W[k])
        full=evaluate_gamma_batch(
            ws,gamma[:C],candidate_chunk=C
        )
        ref[k]=full["muSNR"]
        del ws,full

    diff=fast-ref
    rel=(
        torch.linalg.vector_norm(diff.reshape(-1))
        / torch.clamp(
            torch.linalg.vector_norm(ref.reshape(-1)),
            min=torch.finfo(ref.dtype).tiny,
        )
    )

    return {
        "relative_fro":float(rel.detach().cpu()),
        "max_abs":float(torch.max(torch.abs(diff)).detach().cpu()),
    }


@torch.inference_mode()
def empirical_mean_variance_multi_w_z(
    row: pd.Series,
    W: torch.Tensor,
    gamma: torch.Tensor,
    *,
    n_mc: int=64_000,
    mc_chunk: int=256,
    w_chunk: int=4,
    z_chunk: int=64,
    device=None,
    parity: bool=False,
    seed_br: Optional[int]=None,
    seed_ru: Optional[int]=None,
) -> Dict[str,Any]:
    """
    Streaming Monte-Carlo engine.

    Returns meanEmp/varEmp [K,C] and timing/memory statistics.

    Population variance convention is intentionally used:

        varEmp = E[Y^2] - E[Y]^2

    matching MATLAB var(Y,1) used by the original dataset generator.
    """
    dev=_device(device)
    br,ru=row_to_link_configs(row)

    cd=torch.complex128 if parity else torch.complex64
    rd=torch.float64 if parity else torch.float32

    W=torch.as_tensor(W,dtype=cd,device=dev)
    gamma=torch.as_tensor(gamma,dtype=cd,device=dev)

    if W.ndim!=2:
        raise ValueError("W must be [K,nT].")
    if gamma.ndim!=2:
        raise ValueError("gamma must be [C,nRIS].")

    Knum,nT=W.shape
    C,nRIS=gamma.shape

    expected_nT=2*int(row.nT1)*int(row.nT2)
    if nT!=expected_nT:
        raise ValueError(f"W nT={nT}, expected {expected_nT}")
    if nRIS!=int(row.nRIS):
        raise ValueError(f"gamma nRIS={nRIS}, expected {int(row.nRIS)}")

    seed0=int(row.ch_seed)
    if seed_br is None:
        seed_br=seed0+70_000_121
    if seed_ru is None:
        seed_ru=seed0+80_000_147

    gen_br=torch.Generator(device=dev)
    gen_ru=torch.Generator(device=dev)
    gen_br.manual_seed(int(seed_br))
    gen_ru.manual_seed(int(seed_ru))

    # Accumulate in float64 even in production complex64 mode.
    sum1=torch.zeros((Knum,C),dtype=torch.float64,device=dev)
    sum2=torch.zeros((Knum,C),dtype=torch.float64,device=dev)

    if dev.type=="cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    _sync(dev)
    t0=time.perf_counter()

    done=0
    while done<n_mc:
        n=min(int(mc_chunk),int(n_mc-done))

        HBR=generate_native_link_chunk(
            br,n,generator=gen_br,device=dev,parity=parity
        )
        HRU=generate_native_link_chunk(
            ru,n,generator=gen_ru,device=dev,parity=parity
        )

        N=HBR.shape[0]
        R=HRU.shape[1]
        I=HBR.shape[1]

        for k0 in range(0,Knum,w_chunk):
            k1=min(k0+w_chunk,Knum)
            Wc=W[k0:k1]

            # [N,I,T] @ [T,Kw] -> [N,Kw,I]
            u=torch.matmul(HBR,Wc.T).permute(0,2,1).contiguous()

            # [N,Kw,R,I], reused across every z chunk.
            basis=HRU[:,None,:,:]*u[:,:,None,:]
            basis_flat=basis.reshape(N*(k1-k0)*R,I)

            for c0 in range(0,C,z_chunk):
                c1=min(c0+z_chunk,C)
                gc=gamma[c0:c1]

                # High-throughput complex GEMM.
                eff=(
                    basis_flat @ gc.T
                ).reshape(N,k1-k0,R,c1-c0)

                Y=torch.sum(torch.abs(eff)**2,dim=2)  # [N,Kw,Cz]
                Y64=Y.to(torch.float64)

                sum1[k0:k1,c0:c1]+=torch.sum(Y64,dim=0)
                sum2[k0:k1,c0:c1]+=torch.sum(Y64*Y64,dim=0)

                del eff,Y,Y64

            del u,basis,basis_flat

        del HBR,HRU
        done+=N

    _sync(dev)
    elapsed=time.perf_counter()-t0

    mean=sum1/float(n_mc)
    var=torch.clamp(
        sum2/float(n_mc)-mean*mean,
        min=0.0,
    )

    peak_mb=np.nan
    if dev.type=="cuda":
        peak_mb=float(
            torch.cuda.max_memory_allocated(dev)/1024**2
        )

    return {
        "meanEmp":mean,
        "varEmp":var,
        "seconds":float(elapsed),
        "candidate_count":int(Knum*C),
        "sample_evaluations":int(n_mc*Knum*C),
        "candidate_labels_per_second":float(Knum*C/elapsed),
        "sample_evaluations_per_second":float(n_mc*Knum*C/elapsed),
        "realization_pairs":int(n_mc),
        "peak_memory_MB":peak_mb,
        "seed_br":int(seed_br),
        "seed_ru":int(seed_ru),
    }


@dataclass
class TorchGGQ05Lookup:
    log_cv2: torch.Tensor
    qnorm: torch.Tensor
    legacy_linear_max: Optional[torch.Tensor] = None


def torch_gg_lookup(
    lookup: GGQ05Lookup,
    *,
    device=None,
    dtype=torch.float64,
) -> TorchGGQ05Lookup:
    dev=_device(device)
    bp = (
        None
        if lookup.legacy_linear_max is None
        else torch.as_tensor(lookup.legacy_linear_max,dtype=dtype,device=dev)
    )
    return TorchGGQ05Lookup(
        log_cv2=torch.as_tensor(
            lookup.log_cv2,dtype=dtype,device=dev
        ),
        qnorm=torch.as_tensor(
            lookup.qnorm,dtype=dtype,device=dev
        ),
        legacy_linear_max=bp,
    )


@torch.inference_mode()
def symmetric_gg_q05_torch(
    mu: torch.Tensor,
    var: torch.Tensor,
    lookup: TorchGGQ05Lookup,
) -> Dict[str,torch.Tensor]:
    """
    GPU linear interpolation in log(CV^2), matching the previous q05 Teacher
    notebook and Stage-8B3A reconstructed lookup.
    """
    mu=torch.as_tensor(mu,dtype=lookup.log_cv2.dtype,device=lookup.log_cv2.device)
    var=torch.as_tensor(var,dtype=lookup.log_cv2.dtype,device=lookup.log_cv2.device)

    tiny=torch.finfo(mu.dtype).tiny
    mu_safe=torch.clamp(mu,min=tiny)
    var_safe=torch.clamp(var,min=tiny)

    cv2=var_safe/(mu_safe*mu_safe)
    x=torch.log(cv2)

    grid=lookup.log_cv2
    vals=lookup.qnorm

    clamped=(x<grid[0])|(x>grid[-1])
    xc=torch.clamp(x,min=grid[0],max=grid[-1])

    hi=torch.searchsorted(grid,xc,right=False)
    hi=torch.clamp(hi,min=1,max=grid.numel()-1)
    lo=hi-1

    x0=grid[lo]
    x1=grid[hi]
    y0=vals[lo]
    y1=vals[hi]

    t=(xc-x0)/torch.clamp(x1-x0,min=tiny)
    qnorm=y0+t*(y1-y0)

    if lookup.legacy_linear_max is not None:
        ext=xc > lookup.legacy_linear_max
        logy0=torch.log(torch.clamp(y0,min=tiny))
        logy1=torch.log(torch.clamp(y1,min=tiny))
        qnorm_tail=torch.exp(logy0+t*(logy1-logy0))
        qnorm=torch.where(ext,qnorm_tail,qnorm)

    q05=mu_safe*qnorm

    # Symmetric Gamma-Gamma shape a=b.
    shape=(torch.sqrt(1.0+cv2)+1.0)/torch.clamp(cv2,min=tiny)

    return {
        "q05GG":q05,
        "shapeA":shape,
        "cv2":cv2,
        "lookupClamped":clamped,
    }


@torch.inference_mode()
def run_symmetric_gg_label_engine(
    row: pd.Series,
    static_env,
    W: torch.Tensor,
    gamma: torch.Tensor,
    lookup: GGQ05Lookup,
    *,
    n_mc: int=64_000,
    mc_chunk: int=256,
    w_chunk: int=4,
    z_chunk: int=64,
    device=None,
    parity: bool=False,
) -> Dict[str,Any]:
    """
    End-to-end B3-B label generation for one environment/bank.

    Output matrices are [K,C].
    """
    dev=_device(device)

    _sync(dev)
    t_total=time.perf_counter()

    emp=empirical_mean_variance_multi_w_z(
        row,W,gamma,
        n_mc=n_mc,
        mc_chunk=mc_chunk,
        w_chunk=w_chunk,
        z_chunk=z_chunk,
        device=dev,
        parity=parity,
    )

    _sync(dev)
    t0=time.perf_counter()

    mu=analytic_mu_snr_multi_w_z(
        static_env,W,gamma,z_chunk=z_chunk
    )

    _sync(dev)
    analytic_seconds=time.perf_counter()-t0

    t0=time.perf_counter()

    lookup_t=torch_gg_lookup(
        lookup,device=dev,dtype=torch.float64
    )
    gg=symmetric_gg_q05_torch(
        mu.to(torch.float64),
        emp["varEmp"],
        lookup_t,
    )

    _sync(dev)
    gg_seconds=time.perf_counter()-t0
    total_seconds=time.perf_counter()-t_total

    Knum,C=mu.shape
    labels=Knum*C

    return {
        "muSNR":mu,
        "meanEmp":emp["meanEmp"],
        "varEmp":emp["varEmp"],
        "q05GG":gg["q05GG"],
        "shapeA":gg["shapeA"],
        "cv2":gg["cv2"],
        "lookupClamped":gg["lookupClamped"],

        "empirical_seconds":emp["seconds"],
        "analytic_mu_seconds":float(analytic_seconds),
        "gg_lookup_seconds":float(gg_seconds),
        "total_seconds":float(total_seconds),

        "candidate_count":int(labels),
        "labels_per_second":float(labels/total_seconds),
        "empirical_candidate_labels_per_second":emp[
            "candidate_labels_per_second"
        ],
        "sample_evaluations_per_second":emp[
            "sample_evaluations_per_second"
        ],
        "peak_memory_MB":emp["peak_memory_MB"],
        "n_mc":int(n_mc),
        "seed_br":emp["seed_br"],
        "seed_ru":emp["seed_ru"],
    }


def flatten_label_result(
    result: Dict[str,Any],
    WIdx: torch.Tensor,
    z: torch.Tensor,
) -> pd.DataFrame:
    """
    Convert one bank's [K,C] label matrices to one row per (W,z) candidate.

    zString is intentionally included for downstream dataset generation.
    """
    K,C=result["q05GG"].shape

    widx=WIdx.detach().cpu().numpy()
    zcpu=z.detach().cpu().numpy().astype(np.int8)

    mu=result["muSNR"].detach().cpu().numpy()
    mean=result["meanEmp"].detach().cpu().numpy()
    var=result["varEmp"].detach().cpu().numpy()
    q=result["q05GG"].detach().cpu().numpy()
    shape=result["shapeA"].detach().cpu().numpy()
    cv2=result["cv2"].detach().cpu().numpy()
    clamp=result["lookupClamped"].detach().cpu().numpy()

    rows=[]
    for k in range(K):
        for c in range(C):
            rows.append({
                "wCandidate":k,
                "zCandidate":c,
                "WIdx_i11":int(widx[k,0]),
                "WIdx_i12":int(widx[k,1]),
                "WIdx_i2":int(widx[k,2]),
                "zString":"".join(str(int(x)) for x in zcpu[c]),
                "muSNR":float(mu[k,c]),
                "meanEmp":float(mean[k,c]),
                "varEmp":float(var[k,c]),
                "cv2":float(cv2[k,c]),
                "ggShapeA":float(shape[k,c]),
                "q05GG":float(q[k,c]),
                "lookupClamped":bool(clamp[k,c]),
            })
    return pd.DataFrame(rows)
