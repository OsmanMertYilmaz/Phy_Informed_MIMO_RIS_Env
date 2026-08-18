"""
Stage 8B-2 — Cross-validation against the existing MATLAB dataset.

Goal
----
For the SAME:
    environment / geometry / LSP
    W codeword
    RIS zString

generate an independent PyTorch/CUDA Monte-Carlo channel set and compare:

    Python mean(Y)  vs dataset meanEmp
    Python var(Y)   vs dataset varEmp

The old dataset q05 is NOT a raw empirical percentile. It is the Gamma
quantile obtained from analytic muSNR and empirical varEmp:

    shape = muSNR^2 / varEmp
    scale = varEmp / muSNR
    q05   = Gamma^{-1}(0.05; shape, scale)

Therefore we also compute the same Gamma q05 using the Python MC variance
and compare it with the dataset q05.

Separately, q05EmpPython is the true sample 5th percentile of the Python Y
samples. It is reported only as information for the future Direct-q05 work;
do not confuse it with the dataset q05.
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from typing import Dict, Any, List
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import gamma as scipy_gamma

from ris_env.antenna import ArraySpec
from ris_env.channel_realizations import LinkConfig, generate_native_link_chunk
from ris_env.channel_primitives import (
    generate_cascaded_ch,
    apply_precoder_empirical,
    empirical_snr_samples,
)
from ris_env.codebook import generate_codebook_rank1
from ris_env.ris_response import generate_ris_response_from_z


DEFAULT_CASES = [
    # nRIS, scenario_BR, scenario_RU
    (64,  "Indoor-Office-LOS",  "Indoor-Office-LOS"),
    (64,  "Indoor-Office-NLOS", "Indoor-Office-NLOS"),
    (128, "UMi-LOS",            "UMi-NLOS"),
    (128, "UMi-NLOS",           "UMi-LOS"),
    (256, "UMa-LOS",            "UMa-LOS"),
    (256, "UMa-NLOS",           "UMa-NLOS"),
    (512, "RMa-LOS",            "RMa-NLOS"),
    (512, "RMa-NLOS",           "RMa-LOS"),
]


REQUIRED_COLUMNS = [
    "bankID","splitID","pairID","nEval","ch_seed",
    "WIdx_i11","WIdx_i12","WIdx_i2","zString",
    "scenario_BR","scenario_RU",
    "K_BR","K_RU","M_BR","M_RU","L_BR","L_RU",
    "isLOS_BR","isLOS_RU",
    "c_ASA_BR","c_ZSA_BR","c_ASD_BR","c_ZSD_BR",
    "c_ASA_RU","c_ZSA_RU","c_ASD_RU","c_ZSD_RU",
    "mu_XPR_BR","sigma_XPR_BR",
    "mu_ASA_BR","sigma_ASA_BR",
    "mu_ZSA_BR","sigma_ZSA_BR",
    "mu_ASD_BR","sigma_ASD_BR",
    "mu_ZSD_BR","sigma_ZSD_BR",
    "mu_XPR_RU","sigma_XPR_RU",
    "mu_ASA_RU","sigma_ASA_RU",
    "mu_ZSA_RU","sigma_ZSA_RU",
    "mu_ASD_RU","sigma_ASD_RU",
    "mu_ZSD_RU","sigma_ZSD_RU",
    "ZODoffset_BR","ZODoffset_RU",
    "fc",
    "ris_x","ris_y","ris_z",
    "gnb_x","gnb_y","gnb_z",
    "ue_x","ue_y","ue_z",
    "nT1","nT2","nR1","nR2","nRIS1","nRIS2","nRIS",
    "muSNR","meanEmp","varEmp","q05",
]


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cdtype(parity: bool):
    return torch.complex128 if parity else torch.complex64


def load_validation_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Structural identities used by this validation.
    nris_expected = 2 * df["nRIS1"].astype(int) * df["nRIS2"].astype(int)
    if not np.array_equal(nris_expected.to_numpy(), df["nRIS"].astype(int).to_numpy()):
        raise ValueError("nRIS != 2*nRIS1*nRIS2 for at least one row.")

    return df


def select_representative_rows(
    df: pd.DataFrame,
    *,
    split: str = "test_interpolation",
    cases=DEFAULT_CASES,
) -> pd.DataFrame:
    rows = []

    for nris, sbr, sru in cases:
        m = (
            (df["splitID"].astype(str) == split)
            & (df["nRIS"].astype(int) == int(nris))
            & (df["scenario_BR"].astype(str) == sbr)
            & (df["scenario_RU"].astype(str) == sru)
        )

        s = df.loc[m].sort_values(["bankID","pairID"])
        if len(s) == 0:
            raise ValueError(
                f"No row for split={split}, nRIS={nris}, "
                f"BR={sbr}, RU={sru}"
            )

        rows.append(s.iloc[0])

    return pd.DataFrame(rows).reset_index(drop=True)


def _geometry_vectors(row: pd.Series):
    ris = np.array(
        [row.ris_x,row.ris_y,row.ris_z], dtype=np.float64
    )
    gnb = np.array(
        [row.gnb_x,row.gnb_y,row.gnb_z], dtype=np.float64
    )
    ue = np.array(
        [row.ue_x,row.ue_y,row.ue_z], dtype=np.float64
    )

    # Exact locked convention from the Stage-6 parity port.
    gnb2ris = ris - gnb
    ris2gnb = gnb - ris
    ris2ue  = ue - ris
    ue2ris  = ris - ue

    return gnb2ris, ris2gnb, ris2ue, ue2ris


def row_to_link_configs(row: pd.Series):
    gnb2ris,ris2gnb,ris2ue,ue2ris = _geometry_vectors(row)

    tx_br = ArraySpec(int(row.nT1),int(row.nT2))
    rx_br = ArraySpec(int(row.nRIS1),int(row.nRIS2))

    tx_ru = ArraySpec(int(row.nRIS1),int(row.nRIS2))
    rx_ru = ArraySpec(int(row.nR1),int(row.nR2))

    br = LinkConfig(
        tx_spec=tx_br,
        rx_spec=rx_br,
        a_vector=ris2gnb,
        d_vector=gnb2ris,
        fc=float(row.fc),
        K=float(row.K_BR),
        is_los=bool(row.isLOS_BR),
        M=int(row.M_BR),
        L=int(row.L_BR),

        mu_XPR=float(row.mu_XPR_BR),
        sigma_XPR=float(row.sigma_XPR_BR),

        mu_ASA=float(row.mu_ASA_BR),
        sigma_ASA=float(row.sigma_ASA_BR),
        mu_ZSA=float(row.mu_ZSA_BR),
        sigma_ZSA=float(row.sigma_ZSA_BR),
        mu_ASD=float(row.mu_ASD_BR),
        sigma_ASD=float(row.sigma_ASD_BR),
        mu_ZSD=float(row.mu_ZSD_BR),
        sigma_ZSD=float(row.sigma_ZSD_BR),

        c_ASA=float(row.c_ASA_BR),
        c_ZSA=float(row.c_ZSA_BR),
        c_ASD=float(row.c_ASD_BR),
        c_ZSD=float(row.c_ZSD_BR),
        mu_offset_ZOD=float(row.ZODoffset_BR),
    )

    ru = LinkConfig(
        tx_spec=tx_ru,
        rx_spec=rx_ru,
        a_vector=ue2ris,
        d_vector=ris2ue,
        fc=float(row.fc),
        K=float(row.K_RU),
        is_los=bool(row.isLOS_RU),
        M=int(row.M_RU),
        L=int(row.L_RU),

        mu_XPR=float(row.mu_XPR_RU),
        sigma_XPR=float(row.sigma_XPR_RU),

        mu_ASA=float(row.mu_ASA_RU),
        sigma_ASA=float(row.sigma_ASA_RU),
        mu_ZSA=float(row.mu_ZSA_RU),
        sigma_ZSA=float(row.sigma_ZSA_RU),
        mu_ASD=float(row.mu_ASD_RU),
        sigma_ASD=float(row.sigma_ASD_RU),
        mu_ZSD=float(row.mu_ZSD_RU),
        sigma_ZSD=float(row.sigma_ZSD_RU),

        c_ASA=float(row.c_ASA_RU),
        c_ZSA=float(row.c_ZSA_RU),
        c_ASD=float(row.c_ASD_RU),
        c_ZSD=float(row.c_ZSD_RU),
        mu_offset_ZOD=float(row.ZODoffset_RU),
    )

    return br,ru


@torch.inference_mode()
def row_to_w_gamma(
    row: pd.Series,
    *,
    device=None,
    parity: bool=False,
):
    dev = _device(device)

    cb = generate_codebook_rank1(
        2,
        int(row.nT1),
        int(row.nT2),
        cb_mode=1,
        nl=1,
        device=dev,
        parity=parity,
    )

    i11=int(row.WIdx_i11)
    i12=int(row.WIdx_i12)
    i2=int(row.WIdx_i2)

    # Dataset indices are MATLAB 1-based.
    W = cb.values[:,i2-1,i11-1,i12-1].reshape(-1,1)

    zstr=str(row.zString).strip()
    if len(zstr) != int(row.nRIS):
        raise ValueError(
            f"zString length {len(zstr)} != nRIS {int(row.nRIS)}"
        )
    if set(zstr) - {"0","1"}:
        raise ValueError("zString must contain only 0/1.")

    z=np.fromiter((ord(c)-48 for c in zstr),dtype=np.int64)
    ris=generate_ris_response_from_z(
        z,device=dev,parity=parity
    )

    return W,ris["gamma"]


def gamma_q05(mu: float, var: float, p: float=0.05) -> float:
    mu=float(mu); var=float(var)
    if mu <= 0 or var <= 0:
        return np.nan
    shape=mu*mu/var
    scale=var/mu
    return float(scipy_gamma.ppf(p,a=shape,scale=scale))


@torch.inference_mode()
def validate_dataset_row(
    row: pd.Series,
    *,
    N_python: int=100_000,
    chunk_size: int=1024,
    device=None,
    parity: bool=False,
    store_y: bool=True,
) -> Dict[str,Any]:
    dev=_device(device)
    br,ru=row_to_link_configs(row)
    W,gamma=row_to_w_gamma(
        row,device=dev,parity=parity
    )

    # PyTorch streams are intentionally independent of MATLAB.
    seed0=int(row.ch_seed)
    gen_br=torch.Generator(device=dev)
    gen_ru=torch.Generator(device=dev)
    gen_br.manual_seed(seed0 + 10_000_019)
    gen_ru.manual_seed(seed0 + 20_000_033)

    sum1=0.0
    sum2=0.0
    count=0
    y_parts=[] if store_y else None

    if dev.type=="cuda":
        torch.cuda.synchronize(dev)
    t0=time.perf_counter()

    done=0
    while done < N_python:
        n=min(int(chunk_size),int(N_python-done))

        HBR=generate_native_link_chunk(
            br,n,generator=gen_br,
            device=dev,parity=parity
        )
        HRU=generate_native_link_chunk(
            ru,n,generator=gen_ru,
            device=dev,parity=parity
        )

        F=generate_cascaded_ch(HBR,HRU,gamma)
        Feff=apply_precoder_empirical(F,W)
        Y=empirical_snr_samples(Feff).reshape(-1)

        y64=Y.to(torch.float64)
        sum1 += float(y64.sum().detach().cpu())
        sum2 += float((y64*y64).sum().detach().cpu())
        count += int(Y.numel())

        if store_y:
            y_parts.append(Y.detach().cpu().numpy())

        done += n
        del HBR,HRU,F,Feff,Y,y64

    if dev.type=="cuda":
        torch.cuda.synchronize(dev)
    elapsed=time.perf_counter()-t0

    mean_py=sum1/count
    # MATLAB var(...) default normalization: N-1.
    var_py=max(
        (sum2 - count*mean_py*mean_py)/(count-1),
        0.0
    )

    q05_emp_py=np.nan
    if store_y:
        y=np.concatenate(y_parts)
        q05_emp_py=float(np.quantile(y,0.05))

    mean_ds=float(row.meanEmp)
    var_ds=float(row.varEmp)
    mu_analytic=float(row.muSNR)
    q05_ds=float(row.q05)

    q05_gamma_py=gamma_q05(mu_analytic,var_py,0.05)

    def rel(a,b):
        return abs(float(a)-float(b))/max(abs(float(b)),np.finfo(float).eps)

    return {
        "bankID":int(row.bankID),
        "pairID":int(row.pairID),
        "splitID":str(row.splitID),
        "scenario_BR":str(row.scenario_BR),
        "scenario_RU":str(row.scenario_RU),
        "nT":2*int(row.nT1)*int(row.nT2),
        "nR":2*int(row.nR1)*int(row.nR2),
        "nRIS":int(row.nRIS),
        "N_dataset":int(row.nEval),
        "N_python":int(N_python),

        "meanEmp_dataset":mean_ds,
        "meanPython":mean_py,
        "meanRelErr_vs_dataset":rel(mean_py,mean_ds),

        "muSNR_analytic":mu_analytic,
        "meanRelErr_vs_muSNR":rel(mean_py,mu_analytic),

        "varEmp_dataset":var_ds,
        "varPython":var_py,
        "varRelErr_vs_dataset":rel(var_py,var_ds),

        "q05Gamma_dataset":q05_ds,
        "q05Gamma_from_pythonVar":q05_gamma_py,
        "q05GammaRelErr":rel(q05_gamma_py,q05_ds),

        "q05EmpPython_direct":q05_emp_py,
        "gammaVsDirectPython_relGap":(
            rel(q05_gamma_py,q05_emp_py)
            if np.isfinite(q05_emp_py) and q05_emp_py != 0
            else np.nan
        ),

        "seconds":float(elapsed),
        "realization_pairs_per_second":float(N_python/elapsed),
    }


def summarize_results(results: pd.DataFrame) -> Dict[str,float]:
    out={}
    for col in [
        "meanRelErr_vs_dataset",
        "meanRelErr_vs_muSNR",
        "varRelErr_vs_dataset",
        "q05GammaRelErr",
    ]:
        x=results[col].to_numpy(dtype=np.float64)
        out[col+"_median"]=float(np.median(x))
        out[col+"_p90"]=float(np.quantile(x,0.90))
        out[col+"_max"]=float(np.max(x))

    out["total_seconds"]=float(results["seconds"].sum())
    out["overall_pairs_per_second"]=float(
        results["N_python"].sum()/results["seconds"].sum()
    )
    return out
