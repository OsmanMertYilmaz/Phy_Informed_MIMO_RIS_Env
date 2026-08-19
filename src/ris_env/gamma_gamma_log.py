from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class GGLogQ05Lookup:
    log_cv2: np.ndarray
    log_qnorm: np.ndarray
    legacy_linear_max: Optional[float] = None

    def __post_init__(self):
        self.log_cv2 = np.asarray(self.log_cv2,dtype=np.float64).reshape(-1)
        self.log_qnorm = np.asarray(self.log_qnorm,dtype=np.float64).reshape(-1)

        if self.log_cv2.size != self.log_qnorm.size or self.log_cv2.size < 2:
            raise ValueError("Invalid GG log lookup sizes.")
        if not np.all(np.diff(self.log_cv2) > 0):
            raise ValueError("log_cv2 must be strictly increasing.")
        if not np.all(np.isfinite(self.log_qnorm)):
            raise ValueError("log_qnorm must be finite.")

        if self.legacy_linear_max is not None:
            self.legacy_linear_max = float(self.legacy_linear_max)
            if not (self.log_cv2[0] <= self.legacy_linear_max <= self.log_cv2[-1]):
                raise ValueError("legacy_linear_max outside grid.")

    @property
    def cv2_min(self):
        return float(np.exp(self.log_cv2[0]))

    @property
    def cv2_max(self):
        return float(np.exp(self.log_cv2[-1]))


def load_log_lookup_npz(path) -> GGLogQ05Lookup:
    a = np.load(Path(path),allow_pickle=False)
    need = {"log_cv2","log_qnorm"}
    missing = need - set(a.files)
    if missing:
        raise ValueError(f"GG log lookup missing keys: {sorted(missing)}")

    bp = None
    if "legacy_linear_max" in a.files:
        bp = float(np.asarray(a["legacy_linear_max"]).reshape(()))

    return GGLogQ05Lookup(
        log_cv2=a["log_cv2"],
        log_qnorm=a["log_qnorm"],
        legacy_linear_max=bp,
    )


def _interp_log_qnorm_numpy(logcv, lookup):
    x = np.asarray(logcv,dtype=np.float64)

    clamped = (x < lookup.log_cv2[0]) | (x > lookup.log_cv2[-1])
    xc = np.clip(x,lookup.log_cv2[0],lookup.log_cv2[-1])

    hi = np.searchsorted(lookup.log_cv2,xc,side="left")
    hi = np.clip(hi,1,len(lookup.log_cv2)-1)
    lo = hi-1

    x0 = lookup.log_cv2[lo]
    x1 = lookup.log_cv2[hi]
    l0 = lookup.log_qnorm[lo]
    l1 = lookup.log_qnorm[hi]

    t = (xc-x0)/np.maximum(x1-x0,np.finfo(np.float64).tiny)

    logq = l0+t*(l1-l0)

    if lookup.legacy_linear_max is not None:
        legacy = xc <= lookup.legacy_linear_max
        if np.any(legacy):
            q0 = np.exp(l0[legacy])
            q1 = np.exp(l1[legacy])
            qlin = q0+t[legacy]*(q1-q0)
            logq[legacy] = np.log(qlin)

    return logq,clamped


def symmetric_gg_logq05_numpy(mean,var,lookup):
    mean = np.asarray(mean,dtype=np.float64)
    var = np.asarray(var,dtype=np.float64)
    mean,var = np.broadcast_arrays(mean,var)

    valid = np.isfinite(mean)&np.isfinite(var)&(mean>0)&(var>0)

    log_q05 = np.full(mean.shape,np.nan,dtype=np.float64)
    q05 = np.full(mean.shape,np.nan,dtype=np.float64)
    cv2 = np.full(mean.shape,np.nan,dtype=np.float64)
    shape = np.full(mean.shape,np.nan,dtype=np.float64)
    clamped = np.zeros(mean.shape,dtype=bool)

    if np.any(valid):
        cv = var[valid]/(mean[valid]*mean[valid])
        lx = np.log(cv)

        log_qnorm,c = _interp_log_qnorm_numpy(lx,lookup)
        lq = np.log(mean[valid])+log_qnorm

        cv2[valid] = cv
        log_q05[valid] = lq
        q05[valid] = np.exp(lq)  # diagnostic; may underflow to 0
        clamped[valid] = c

        tiny = np.finfo(np.float64).tiny
        shape[valid] = (np.sqrt(1+cv)+1)/np.maximum(cv,tiny)

    return {
        "logQ05GG":log_q05,
        "q05GG":q05,
        "shapeA":shape,
        "cv2":cv2,
        "lookupClamped":clamped,
    }


@dataclass
class TorchGGLogQ05Lookup:
    log_cv2: torch.Tensor
    log_qnorm: torch.Tensor
    legacy_linear_max: Optional[torch.Tensor] = None


def torch_log_lookup(lookup,*,device=None,dtype=torch.float64):
    dev = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    bp = None
    if lookup.legacy_linear_max is not None:
        bp = torch.as_tensor(lookup.legacy_linear_max,dtype=dtype,device=dev)

    return TorchGGLogQ05Lookup(
        log_cv2=torch.as_tensor(lookup.log_cv2,dtype=dtype,device=dev),
        log_qnorm=torch.as_tensor(lookup.log_qnorm,dtype=dtype,device=dev),
        legacy_linear_max=bp,
    )


@torch.inference_mode()
def symmetric_gg_logq05_torch(mean,var,lookup):
    mean = torch.as_tensor(
        mean,dtype=lookup.log_cv2.dtype,device=lookup.log_cv2.device
    )
    var = torch.as_tensor(
        var,dtype=lookup.log_cv2.dtype,device=lookup.log_cv2.device
    )

    tiny = torch.finfo(mean.dtype).tiny
    mean_safe = torch.clamp(mean,min=tiny)
    var_safe = torch.clamp(var,min=tiny)

    cv2 = var_safe/(mean_safe*mean_safe)
    x = torch.log(cv2)

    grid = lookup.log_cv2
    vals = lookup.log_qnorm

    clamped = (x<grid[0]) | (x>grid[-1])
    xc = torch.clamp(x,min=grid[0],max=grid[-1])

    hi = torch.searchsorted(grid,xc,right=False)
    hi = torch.clamp(hi,min=1,max=grid.numel()-1)
    lo = hi-1

    x0 = grid[lo]
    x1 = grid[hi]
    l0 = vals[lo]
    l1 = vals[hi]

    t = (xc-x0)/torch.clamp(x1-x0,min=tiny)
    log_qnorm = l0+t*(l1-l0)

    if lookup.legacy_linear_max is not None:
        legacy = xc <= lookup.legacy_linear_max
        q0 = torch.exp(l0)
        q1 = torch.exp(l1)
        qlin = q0+t*(q1-q0)
        log_legacy = torch.log(torch.clamp(qlin,min=tiny))
        log_qnorm = torch.where(legacy,log_legacy,log_qnorm)

    log_q05 = torch.log(mean_safe)+log_qnorm
    q05 = torch.exp(log_q05)  # diagnostic; may underflow to 0

    shape = (torch.sqrt(1+cv2)+1)/torch.clamp(cv2,min=tiny)

    return {
        "logQ05GG":log_q05,
        "q05GG":q05,
        "shapeA":shape,
        "cv2":cv2,
        "lookupClamped":clamped,
    }
