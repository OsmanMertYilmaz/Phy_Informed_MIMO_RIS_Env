"""
Stage 5 GPU Type-I rank-1 codebook and precoder selection.

Official current project scope:
    XP      = 2
    cb_mode = 1
    nl      = 1
    O1      = 4
    O2      = 4 if N2 > 1 else 1

This ports the rank-1 branch of MATLAB:
    generate_codebook
    selection_precoder

The current RIS pipeline uses nl=1, so higher-rank codebook branches are
intentionally NOT included in this parity module.

MATLAB rank-1 codeword:
    w(i2,i11,i12) = 1/sqrt(nPorts) * [v_lm ; phi_n v_lm]
    phi_n = exp(j*pi*n/2)

with:
    l = i11
    m = i12

and:
    v_lm = reshape((u_l * u_m).', [], 1)

Selection exactly follows the supplied MATLAB logic:
    1) choose (i11,i12) from the first polarization half of V(:,1)
    2) choose i2 from the full V(:,1)
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math
import time
import numpy as np
import torch


@dataclass
class Rank1Codebook:
    # canonical shape [nPorts, 4, i11Len, i12Len]
    values: torch.Tensor
    N1: int
    N2: int
    XP: int
    O1: int
    O2: int

    @property
    def n_ports(self):
        return int(self.values.shape[0])

    @property
    def i2_len(self):
        return int(self.values.shape[1])

    @property
    def i11_len(self):
        return int(self.values.shape[2])

    @property
    def i12_len(self):
        return int(self.values.shape[3])


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool):
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool):
    return torch.complex128 if parity else torch.complex64


def _get_vlm(N1, N2, O1, O2, l, m, *, device, parity):
    """
    Exact ordering of MATLAB:
        um = exp(2*pi*j*m*(0:N2-1)/(O2*N2));
        ul = exp(2*pi*j*l*(0:N1-1)/(O1*N1)).';
        vlm = reshape((ul.*um).',[],1);

    Output ordering is N2-fast inside N1.
    """
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    n1 = torch.arange(N1, dtype=rd, device=device)
    n2 = torch.arange(N2, dtype=rd, device=device)

    l = torch.as_tensor(l, dtype=rd, device=device)
    m = torch.as_tensor(m, dtype=rd, device=device)

    ul = torch.exp(
        2j * math.pi * l[..., None] * n1 / (O1 * N1)
    ).to(cd)
    um = torch.exp(
        2j * math.pi * m[..., None] * n2 / (O2 * N2)
    ).to(cd)

    # [...,N1,N2], then row-major flatten -> N2-fast.
    outer = ul[..., :, None] * um[..., None, :]
    return outer.reshape(*outer.shape[:-2], N1*N2)


@torch.inference_mode()
def generate_codebook_rank1(
    XP: int,
    N1: int,
    N2: int,
    cb_mode: int = 1,
    nl: int = 1,
    *,
    device=None,
    parity: bool = False,
) -> Rank1Codebook:
    """
    Port of the nl=1 branch of generate_codebook.m.

    Returns canonical:
        values [nPorts, 4, i11Len, i12Len]
    """
    if XP != 2:
        raise ValueError("Current project Type-I implementation assumes XP=2.")
    if nl != 1:
        raise ValueError("Stage 5 official scope is nl=1.")
    if cb_mode != 1:
        raise ValueError("Current project official cb_mode is 1.")
    if N1 < 1 or N2 < 1:
        raise ValueError("N1 and N2 must be >= 1.")

    dev = _device(device)
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    O1 = 4
    O2 = 4 if N2 > 1 else 1

    n_ports = XP * N1 * N2

    # Special MATLAB branch for exactly 2 ports.
    if n_ports == 2:
        vals = torch.empty((2,4,1,1), dtype=cd, device=dev)
        s = torch.tensor(1.0/math.sqrt(2.0), dtype=rd, device=dev)
        vals[:,0,0,0] = s * torch.tensor([1,1], dtype=cd, device=dev)
        vals[:,1,0,0] = s * torch.tensor([1,1j], dtype=cd, device=dev)
        vals[:,2,0,0] = s * torch.tensor([1,-1], dtype=cd, device=dev)
        vals[:,3,0,0] = s * torch.tensor([1,-1j], dtype=cd, device=dev)
        return Rank1Codebook(vals,N1,N2,XP,O1,O2)

    i11_len = N1 * O1
    i12_len = N2 * O2
    i2_len = 4

    # Vectorize all (i11,i12) beams.
    i11 = torch.arange(i11_len, dtype=rd, device=dev)
    i12 = torch.arange(i12_len, dtype=rd, device=dev)

    L, M = torch.meshgrid(i11, i12, indexing="ij")
    vlm = _get_vlm(
        N1,N2,O1,O2,
        L.reshape(-1),
        M.reshape(-1),
        device=dev,
        parity=parity,
    )
    # [I11*I12, half]
    half = N1*N2
    vlm = vlm.reshape(i11_len,i12_len,half)

    phases = torch.exp(
        1j * math.pi/2
        * torch.arange(i2_len, dtype=rd, device=dev)
    ).to(cd)

    norm = torch.tensor(
        1.0/math.sqrt(n_ports),
        dtype=rd,
        device=dev,
    )

    # Build canonical [ports,i2,i11,i12].
    top = vlm.permute(2,0,1)[:,None,:,:].expand(
        half,i2_len,i11_len,i12_len
    )
    bottom = top * phases[None,:,None,None]

    values = norm.to(cd) * torch.cat((top,bottom),dim=0)
    return Rank1Codebook(values,N1,N2,XP,O1,O2)


@torch.inference_mode()
def flatten_codebook_matlab_loop_order(
    codebook: Rank1Codebook,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten codewords in explicit MATLAB validation loop order:

        for i11
          for i12
            for i2

    Returns:
        Wflat [nPorts,nCodewords]
        idx   [nCodewords,3], 1-based [i11,i12,i2]
    """
    cb = codebook.values
    P,I2,I11,I12 = cb.shape

    # [I11,I12,I2,P] -> flatten first three in row-major:
    # i2 fastest, then i12, then i11.
    Wflat = cb.permute(2,3,1,0).reshape(I11*I12*I2,P).T

    i11 = torch.arange(1,I11+1,device=cb.device)
    i12 = torch.arange(1,I12+1,device=cb.device)
    i2 = torch.arange(1,I2+1,device=cb.device)
    A,B,C = torch.meshgrid(i11,i12,i2,indexing="ij")
    idx = torch.stack(
        (A.reshape(-1),B.reshape(-1),C.reshape(-1)),
        dim=1
    ).to(torch.long)

    return Wflat, idx


@torch.inference_mode()
def select_precoder_rank1_batch_from_v1(
    codebook: Rank1Codebook,
    v1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version of selection_precoder(..., nl=1, V).

    v1:
        [nPorts] or [B,nPorts], corresponding to V(:,1).

    Returns:
        W    [B,nPorts]
        WIdx [B,3], MATLAB 1-based [i11,i12,i2]
    """
    cb = codebook.values
    dev = cb.device
    cd = cb.dtype

    v = torch.as_tensor(v1,dtype=cd,device=dev)
    if v.ndim == 1:
        v = v.unsqueeze(0)
    if v.ndim != 2 or v.shape[1] != codebook.n_ports:
        raise ValueError(
            f"v1 must be [B,{codebook.n_ports}], got {tuple(v.shape)}"
        )

    B = v.shape[0]
    half = codebook.n_ports // 2
    I11 = codebook.i11_len
    I12 = codebook.i12_len

    # MATLAB selection_i11_i12 uses:
    # codebook(firstHalf,1,1,i11,i12)' * wopt
    spatial = cb[:half,0,:,:]  # [half,I11,I12]

    # [B,I11,I12]
    corr_spatial = torch.abs(
        torch.einsum("hij,bh->bij", spatial.conj(), v[:,:half])
    )

    # MATLAB loop order i11 outer / i12 inner with strict max update.
    # torch.argmax returns the first maximum in flattened order.
    flat_idx = torch.argmax(
        corr_spatial.reshape(B,I11*I12),
        dim=1
    )

    i11_0 = torch.div(flat_idx,I12,rounding_mode="floor")
    i12_0 = flat_idx % I12

    spatial_flat = spatial.permute(1,2,0).reshape(I11*I12,half)
    selected_top = spatial_flat[flat_idx]  # normalized top half

    # MATLAB selection_i2:
    # temp = [vlm, phi_i*vlm].'
    # corr = abs(temp' * wopt)
    # Here selected_top already includes 1/sqrt(nPorts), a common factor
    # across i2, so argmax is identical.
    a = torch.sum(selected_top.conj() * v[:,:half],dim=1)
    b = torch.sum(selected_top.conj() * v[:,half:],dim=1)

    rd = v.real.dtype
    phases = torch.exp(
        1j * math.pi/2
        * torch.arange(4,dtype=rd,device=dev)
    ).to(cd)

    corr_i2 = torch.abs(
        a[:,None] + phases.conj()[None,:] * b[:,None]
    )
    i2_0 = torch.argmax(corr_i2,dim=1)

    # Gather exact codebook W.
    W = cb[:,i2_0,i11_0,i12_0].T.contiguous()

    WIdx = torch.stack(
        (i11_0+1,i12_0+1,i2_0+1),
        dim=1
    ).to(torch.long)

    return W,WIdx


@torch.inference_mode()
def select_precoder_rank1(
    codebook: Rank1Codebook,
    V,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-case MATLAB-compatible wrapper.
    V may be [nPorts,k]; only V(:,1) is used for nl=1.
    """
    vt = torch.as_tensor(V,dtype=codebook.values.dtype,device=codebook.values.device)
    if vt.ndim == 1:
        v1 = vt
    elif vt.ndim == 2:
        v1 = vt[:,0]
    else:
        raise ValueError("V must be rank 1 or 2")

    W,idx = select_precoder_rank1_batch_from_v1(codebook,v1)
    return W[0],idx[0]


def _rel_fro(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = max(np.linalg.norm(b.ravel()),np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel())/den)


def _phase_invariant_error(v,vref):
    v = np.asarray(v).reshape(-1)
    r = np.asarray(vref).reshape(-1)
    den = np.linalg.norm(v)*np.linalg.norm(r)
    if den == 0:
        return float("nan")
    overlap = abs(np.vdot(r,v))/den
    return float(max(0.0,1.0-overlap))


def compare_stage5_matlab_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
) -> Dict[str,Any]:
    from scipy.io import loadmat

    M = loadmat(mat_path,squeeze_me=False)
    dev = _device(device)

    def sc(name):
        return int(np.asarray(M[name]).reshape(()))

    N1 = sc("N1")
    N2 = sc("N2")
    XP = sc("XP")

    cb = generate_codebook_rank1(
        XP,N1,N2,1,1,
        device=dev,
        parity=parity,
    )

    Wflat,idxflat = flatten_codebook_matlab_loop_order(cb)

    ref_flat = np.asarray(M["CBflat"])
    ref_idx = np.asarray(M["CBidx"],dtype=np.int64)

    got_flat = Wflat.detach().cpu().numpy()
    got_idx = idxflat.detach().cpu().numpy()

    # Selection using MATLAB's V(:,1): isolates codebook/selector parity.
    V1mat = np.asarray(M["V1MAT"])
    Wref = np.asarray(M["WMAT"])
    Iref = np.asarray(M["WIdxMAT"],dtype=np.int64)

    W_from_matV, I_from_matV = select_precoder_rank1_batch_from_v1(
        cb,
        V1mat.T,
    )
    W_from_matV = W_from_matV.detach().cpu().numpy().T
    I_from_matV = I_from_matV.detach().cpu().numpy()

    # End-to-end SVD integration using same H.
    Hmat = np.asarray(M["Hstack"])
    # MATLAB saved [nT,nT,nTrials].
    H = torch.as_tensor(
        np.moveaxis(Hmat,2,0),
        dtype=_cdtype(parity),
        device=dev,
    )
    _,_,Vh = torch.linalg.svd(H,full_matrices=False)
    V1py = Vh.conj().transpose(-2,-1)[:,:,0]

    W_from_svd,I_from_svd = select_precoder_rank1_batch_from_v1(
        cb,V1py
    )
    I_from_svd_np = I_from_svd.detach().cpu().numpy()

    v1py_np = V1py.detach().cpu().numpy().T

    phase_errors = [
        _phase_invariant_error(v1py_np[:,k],V1mat[:,k])
        for k in range(V1mat.shape[1])
    ]

    return {
        "codebook_relFro": _rel_fro(got_flat,ref_flat),
        "codebook_maxAbs": float(np.max(np.abs(got_flat-ref_flat))),
        "codebook_index_exact": bool(np.array_equal(got_idx,ref_idx)),
        "selector_matV_W_relFro": _rel_fro(W_from_matV,Wref),
        "selector_matV_W_maxAbs": float(np.max(np.abs(W_from_matV-Wref))),
        "selector_matV_index_match_pct": float(
            100*np.mean(np.all(I_from_matV==Iref,axis=1))
        ),
        "svd_V1_phaseInvariant_max": float(np.max(phase_errors)),
        "svd_selector_index_match_pct": float(
            100*np.mean(np.all(I_from_svd_np==Iref,axis=1))
        ),
    }


@torch.inference_mode()
def benchmark_selector(
    *,
    N1: int = 4,
    N2: int = 2,
    batch_size: int = 65536,
    repeats: int = 10,
    device=None,
) -> Dict[str,Any]:
    dev = _device(device)
    cb = generate_codebook_rank1(
        2,N1,N2,1,1,
        device=dev,
        parity=False,
    )

    rd = torch.float32
    cd = torch.complex64

    gen = torch.Generator(device=dev)
    gen.manual_seed(42)

    v = (
        torch.randn(
            (batch_size,cb.n_ports),
            generator=gen,
            device=dev,
            dtype=rd,
        )
        + 1j*torch.randn(
            (batch_size,cb.n_ports),
            generator=gen,
            device=dev,
            dtype=rd,
        )
    ).to(cd)

    v = v / torch.linalg.vector_norm(v,dim=1,keepdim=True)

    _ = select_precoder_rank1_batch_from_v1(cb,v)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = select_precoder_rank1_batch_from_v1(cb,v)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        ts.append(time.perf_counter()-t0)

    med = float(np.median(ts))
    return {
        "device": str(dev),
        "N1": int(N1),
        "N2": int(N2),
        "nT": int(cb.n_ports),
        "batch_size": int(batch_size),
        "median_seconds": med,
        "vectors_per_second": float(batch_size/med),
    }
