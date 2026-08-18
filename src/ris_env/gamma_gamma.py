"""
Stage 8B-3A — Symmetric Gamma-Gamma label convergence.

IMPORTANT
---------
The project target is NOT the raw empirical 5th percentile of Y.

The locked statistical assumption is symmetric Gamma-Gamma:

    Y_GG = c * X1 * X2,
    X1, X2 ~ Gamma(a, scale=1), iid.

For each (environment, W, z):

    analytic muSNR
    empirical varEmp from Monte Carlo

define the symmetric-GG fit.

Let:
    cv2 = varEmp / muSNR^2

Then:
    a = (sqrt(1 + cv2) + 1) / cv2

and the q05 is obtained from the project's existing normalized
Gamma-Gamma q05 lookup.

The lookup is reconstructed from the existing corrected dataset using the
same relationship used by the previous q05 Teacher notebook:

    logCV2 = log(varEmp / muSNR^2)
    qNorm  = q05GammaGammaFit / muSNR
    q05GG  = muSNR * interp(logCV2)

Stage 8B-3A generates ONE native CUDA Monte-Carlo stream up to N=100000
for each representative physical configuration and evaluates prefix values

    N = 1k, 2k, 4k, 8k, 10k, 16k, 32k, 64k, 100k

against the N=100k reference.

Variance uses population normalization exactly like MATLAB var(Y,1).
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Sequence, Optional
import time

import numpy as np
import pandas as pd
import torch

from ris_env.validation import (
    load_validation_dataset,
    row_to_link_configs,
    row_to_w_gamma,
)
from ris_env.channel_realizations import generate_native_link_chunk
from ris_env.channel_primitives import (
    generate_cascaded_ch,
    apply_precoder_empirical,
    empirical_snr_samples,
)


DEFAULT_N_GRID = (
    1_000,
    2_000,
    4_000,
    8_000,
    10_000,
    16_000,
    32_000,
    64_000,
    100_000,
)

DEFAULT_CASES = (
    (64,  "Indoor-Office-LOS",  "Indoor-Office-LOS"),
    (64,  "Indoor-Office-NLOS", "Indoor-Office-NLOS"),
    (128, "UMi-LOS",            "UMi-NLOS"),
    (128, "UMi-NLOS",           "UMi-LOS"),
    (256, "UMa-LOS",            "UMa-LOS"),
    (256, "UMa-NLOS",           "UMa-NLOS"),
    (512, "RMa-LOS",            "RMa-NLOS"),
    (512, "RMa-NLOS",           "RMa-LOS"),
)


@dataclass
class GGQ05Lookup:
    log_cv2: np.ndarray
    qnorm: np.ndarray
    # None => legacy interpolation is linear in qNorm over the whole grid.
    # Packaged v2 lookup sets this to the old maximum log(CV^2): values above
    # that breakpoint use log(qNorm) interpolation for stable tail accuracy.
    legacy_linear_max: Optional[float] = None

    def __post_init__(self):
        self.log_cv2 = np.asarray(self.log_cv2,dtype=np.float64).reshape(-1)
        self.qnorm = np.asarray(self.qnorm,dtype=np.float64).reshape(-1)
        if self.log_cv2.size != self.qnorm.size or self.log_cv2.size < 2:
            raise ValueError("Invalid GG lookup sizes.")
        if not np.all(np.diff(self.log_cv2) > 0):
            raise ValueError("log_cv2 must be strictly increasing.")
        if not np.all(np.isfinite(self.qnorm)) or np.any(self.qnorm <= 0):
            raise ValueError("qnorm must be finite and positive.")
        if self.legacy_linear_max is not None:
            self.legacy_linear_max = float(self.legacy_linear_max)
            if not (self.log_cv2[0] <= self.legacy_linear_max <= self.log_cv2[-1]):
                raise ValueError("legacy_linear_max must lie inside lookup grid.")

    @property
    def cv2_min(self):
        return float(np.exp(self.log_cv2[0]))

    @property
    def cv2_max(self):
        return float(np.exp(self.log_cv2[-1]))


def build_gg_lookup_from_dataset(
    df_full: pd.DataFrame,
    *,
    round_decimals: int=12,
) -> GGQ05Lookup:
    """
    Reconstruct the project's symmetric-GG q05 lookup from the corrected
    dataset, matching the previous q05 Teacher notebook.
    """
    needed = ["muSNR","varEmp","q05GammaGammaFit"]
    missing=[c for c in needed if c not in df_full.columns]
    if missing:
        raise ValueError(f"Missing GG lookup columns: {missing}")

    mu=pd.to_numeric(df_full["muSNR"],errors="coerce").to_numpy(np.float64)
    var=pd.to_numeric(df_full["varEmp"],errors="coerce").to_numpy(np.float64)
    q=pd.to_numeric(
        df_full["q05GammaGammaFit"],errors="coerce"
    ).to_numpy(np.float64)

    ok=(
        np.isfinite(mu) & np.isfinite(var) & np.isfinite(q)
        & (mu>0) & (var>0) & (q>=0)
    )
    if ok.sum() < 100:
        raise ValueError("Not enough valid rows to reconstruct GG lookup.")

    log_cv2=np.log(var[ok]/(mu[ok]**2))
    qnorm=q[ok]/mu[ok]

    tab=pd.DataFrame({
        "logCV2Rounded":np.round(log_cv2,round_decimals),
        "qNorm":qnorm,
    })
    tab=(
        tab.groupby("logCV2Rounded",as_index=False)["qNorm"]
        .median()
        .sort_values("logCV2Rounded")
        .reset_index(drop=True)
    )

    x=tab["logCV2Rounded"].to_numpy(np.float64)
    y=tab["qNorm"].to_numpy(np.float64)

    # Same numerical monotonicity repair used in the previous Teacher notebook.
    y=np.minimum.accumulate(y)

    return GGQ05Lookup(x,y)


def gg_shape_from_mean_variance(mu,var):
    mu=np.asarray(mu,dtype=np.float64)
    var=np.asarray(var,dtype=np.float64)
    cv2=var/np.maximum(mu*mu,np.finfo(float).tiny)
    return (np.sqrt(1.0+cv2)+1.0)/np.maximum(cv2,np.finfo(float).tiny)


def symmetric_gg_q05_numpy(
    mu,
    var,
    lookup: GGQ05Lookup,
):
    mu=np.asarray(mu,dtype=np.float64)
    var=np.asarray(var,dtype=np.float64)

    valid=np.isfinite(mu)&np.isfinite(var)&(mu>0)&(var>0)
    q=np.full(np.broadcast(mu,var).shape,np.nan,dtype=np.float64)
    mu_b,var_b=np.broadcast_arrays(mu,var)

    logcv=np.full(mu_b.shape,np.nan,dtype=np.float64)
    logcv[valid]=np.log(var_b[valid]/(mu_b[valid]**2))

    xv=logcv[valid]
    norm=np.interp(
        xv,
        lookup.log_cv2,
        lookup.qnorm,
        left=lookup.qnorm[0],
        right=lookup.qnorm[-1],
    )

    # Preserve the old Teacher exactly on its validated support.  The new
    # high-CV^2 tail spans many decades in qNorm, so linear interpolation in
    # qNorm is numerically poor there.  Above the legacy breakpoint we
    # interpolate log(qNorm) versus log(CV^2).
    if lookup.legacy_linear_max is not None:
        ext=xv > lookup.legacy_linear_max
        if np.any(ext):
            norm[ext]=np.exp(np.interp(
                xv[ext],
                lookup.log_cv2,
                np.log(lookup.qnorm),
                left=np.log(lookup.qnorm[0]),
                right=np.log(lookup.qnorm[-1]),
            ))

    q[valid]=mu_b[valid]*norm

    clamped=valid & (
        (logcv < lookup.log_cv2[0])
        | (logcv > lookup.log_cv2[-1])
    )

    return q,clamped


def load_full_dataset_for_b3(csv_path: str):
    # The B2 loader intentionally did not include q05GammaGammaFit.
    # Read the full file once here because this column is needed to reconstruct
    # the exact project GG lookup.
    return pd.read_csv(csv_path)


def select_convergence_rows(
    df: pd.DataFrame,
    *,
    split: str="test_interpolation",
    cases=DEFAULT_CASES,
    rows_per_case: int=2,
) -> pd.DataFrame:
    """
    Pick multiple physical configurations across nRIS/scenario branches.
    Prefer distinct banks so the convergence test is not one-environment-only.
    """
    chosen=[]

    for nris,sbr,sru in cases:
        m=(
            (df["splitID"].astype(str)==split)
            & (df["nRIS"].astype(int)==int(nris))
            & (df["scenario_BR"].astype(str)==sbr)
            & (df["scenario_RU"].astype(str)==sru)
        )
        s=df.loc[m].sort_values(["bankID","pairID"]).copy()
        if s.empty:
            raise ValueError(
                f"No rows: split={split}, nRIS={nris}, BR={sbr}, RU={sru}"
            )

        # First configuration from distinct banks.
        first_per_bank=s.groupby("bankID",sort=False,as_index=False).head(1)
        take=first_per_bank.head(rows_per_case)

        # Fallback if this branch has fewer banks than requested.
        if len(take) < rows_per_case:
            need=rows_per_case-len(take)
            already=set(zip(take.bankID,take.pairID))
            extra=s[
                ~pd.Series(
                    list(zip(s.bankID,s.pairID)),
                    index=s.index,
                ).isin(already)
            ].head(need)
            take=pd.concat([take,extra],ignore_index=False)

        chosen.append(take.head(rows_per_case))

    return pd.concat(chosen,ignore_index=True)


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.inference_mode()
def run_prefix_convergence_row(
    row: pd.Series,
    lookup: GGQ05Lookup,
    *,
    n_grid: Sequence[int]=DEFAULT_N_GRID,
    chunk_size: int=1000,
    device=None,
    parity: bool=False,
) -> pd.DataFrame:
    """
    Generate a single native-CUDA stream up to max(n_grid), and calculate
    population mean/variance + symmetric-GG q05 at every prefix.

    Exact old-dataset variance convention:
        varEmp = var(Y,1)
               = E[Y^2] - E[Y]^2
    """
    n_grid=tuple(sorted(set(int(x) for x in n_grid)))
    if n_grid[0] <= 0:
        raise ValueError("n_grid must be positive.")
    n_max=n_grid[-1]

    # For exact prefixes, use a chunk size that divides all requested N.
    if any(n % chunk_size != 0 for n in n_grid):
        raise ValueError(
            "chunk_size must divide every N in n_grid for exact prefix capture."
        )

    dev=_device(device)
    br,ru=row_to_link_configs(row)
    W,gamma=row_to_w_gamma(row,device=dev,parity=parity)

    seed0=int(row.ch_seed)
    gen_br=torch.Generator(device=dev)
    gen_ru=torch.Generator(device=dev)
    gen_br.manual_seed(seed0 + 30_000_053)
    gen_ru.manual_seed(seed0 + 40_000_079)

    sum1=0.0
    sum2=0.0
    count=0
    next_idx=0
    records=[]

    if dev.type=="cuda":
        torch.cuda.synchronize(dev)
    t0=time.perf_counter()

    while count < n_max:
        n=min(chunk_size,n_max-count)

        HBR=generate_native_link_chunk(
            br,n,generator=gen_br,device=dev,parity=parity
        )
        HRU=generate_native_link_chunk(
            ru,n,generator=gen_ru,device=dev,parity=parity
        )

        F=generate_cascaded_ch(HBR,HRU,gamma)
        Feff=apply_precoder_empirical(F,W)
        Y=empirical_snr_samples(Feff).reshape(-1).to(torch.float64)

        sum1 += float(Y.sum().detach().cpu())
        sum2 += float((Y*Y).sum().detach().cpu())
        count += int(Y.numel())

        while next_idx < len(n_grid) and count == n_grid[next_idx]:
            mean=sum1/count
            var=max(sum2/count - mean*mean,0.0)

            mu=float(row.muSNR)
            q05,clamped=symmetric_gg_q05_numpy(mu,var,lookup)
            a=gg_shape_from_mean_variance(mu,var)

            if dev.type=="cuda":
                torch.cuda.synchronize(dev)
            elapsed=time.perf_counter()-t0

            records.append({
                "bankID":int(row.bankID),
                "pairID":int(row.pairID),
                "scenario_BR":str(row.scenario_BR),
                "scenario_RU":str(row.scenario_RU),
                "nT":2*int(row.nT1)*int(row.nT2),
                "nR":2*int(row.nR1)*int(row.nR2),
                "nRIS":int(row.nRIS),
                "N":int(count),
                "muSNR":mu,
                "meanEmpPrefix":float(mean),
                "varEmpPrefix":float(var),
                "ggShapeA":float(np.asarray(a).reshape(())),
                "q05GGPrefix":float(np.asarray(q05).reshape(())),
                "lookupClamped":bool(np.asarray(clamped).reshape(())),
                "elapsedPrefix_s":float(elapsed),
            })
            next_idx += 1

        del HBR,HRU,F,Feff,Y

    if next_idx != len(n_grid):
        raise RuntimeError("Not all prefix checkpoints were captured.")

    out=pd.DataFrame(records)

    # Nmax is the within-stream reference.
    ref=out.loc[out["N"]==n_max].iloc[0]
    var_ref=float(ref.varEmpPrefix)
    q_ref=float(ref.q05GGPrefix)

    out["varRelErr_vs_Nmax"]=(
        np.abs(out["varEmpPrefix"]-var_ref)
        / max(abs(var_ref),np.finfo(float).eps)
    )
    out["q05GGRelErr_vs_Nmax"]=(
        np.abs(out["q05GGPrefix"]-q_ref)
        / max(abs(q_ref),np.finfo(float).eps)
    )

    # Independent old-dataset reference (about 1e4 samples).
    out["N_dataset"]=int(row.nEval)
    out["varEmp_dataset"]=float(row.varEmp)
    out["q05GG_dataset"]=float(row.q05GammaGammaFit)

    out["varRelErr_vs_dataset"]=(
        np.abs(out["varEmpPrefix"]-float(row.varEmp))
        / max(abs(float(row.varEmp)),np.finfo(float).eps)
    )
    out["q05GGRelErr_vs_dataset"]=(
        np.abs(out["q05GGPrefix"]-float(row.q05GammaGammaFit))
        / max(abs(float(row.q05GammaGammaFit)),np.finfo(float).eps)
    )

    return out


def summarize_n_convergence(all_rows: pd.DataFrame) -> pd.DataFrame:
    def q(x,p):
        return float(np.quantile(np.asarray(x,dtype=np.float64),p))

    rows=[]
    for N,g in all_rows.groupby("N",sort=True):
        rows.append({
            "N":int(N),
            "configs":int(len(g)),
            "varMdAPE_pct":100*float(np.median(g.varRelErr_vs_Nmax)),
            "varP90APE_pct":100*q(g.varRelErr_vs_Nmax,.90),
            "varMaxAPE_pct":100*float(np.max(g.varRelErr_vs_Nmax)),
            "q05GGMdAPE_pct":100*float(np.median(g.q05GGRelErr_vs_Nmax)),
            "q05GGP90APE_pct":100*q(g.q05GGRelErr_vs_Nmax,.90),
            "q05GGMaxAPE_pct":100*float(np.max(g.q05GGRelErr_vs_Nmax)),
            "lookupClamp_pct":100*float(np.mean(g.lookupClamped)),
        })
    return pd.DataFrame(rows)


def recommend_n(
    summary: pd.DataFrame,
    *,
    q05_median_limit_pct: float=1.0,
    q05_p90_limit_pct: float=2.0,
    var_median_limit_pct: float=2.0,
    var_p90_limit_pct: float=5.0,
):
    ok=(
        (summary.q05GGMdAPE_pct <= q05_median_limit_pct)
        & (summary.q05GGP90APE_pct <= q05_p90_limit_pct)
        & (summary.varMdAPE_pct <= var_median_limit_pct)
        & (summary.varP90APE_pct <= var_p90_limit_pct)
    )
    if not ok.any():
        return None
    return int(summary.loc[ok,"N"].iloc[0])
