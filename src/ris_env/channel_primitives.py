"""
Stage 8B-1 — stochastic channel realization parity from fixed random primitives.

This ports the current MATLAB generate_channel_train realization path without
trying to reproduce MATLAB's RNG. MATLAB exports the actual random primitives
(XPR, spread samples, per-cluster offsets, and per-ray polarization phases).
Python consumes exactly the same primitives.

Locked project assumptions already validated in Stage 1:
    - dual-polarized Size=[M N 2 1 1]
    - isotropic elements
    - polarization angles [+45,-45] degrees
    - zero Tx/Rx array orientation
    - 0.5-lambda spatial spacing
    - MATLAB port order: M fastest, then N, then polarization

The mathematical realization path is the current generate_channel_train.m:
    ray angles -> spatial phase -> 2x2 polarization matrix -> ray sum
    -> NLOS normalization -> optional LOS Rician combination.
"""

from __future__ import annotations

# NOTE: This module is a production-name refactor of a MATLAB-parity-validated
# implementation. Numerical behavior is intentionally preserved. See
# docs/VALIDATION_STATUS.md before changing equations or tensor conventions.

from typing import Dict, Any
import math
import numpy as np
import torch

from ris_env.antenna import (
    ArraySpec,
    build_dualpol_positions_lambda,
    generate_channel_moments_batch,
)

ALPHA = (
    0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492,
    0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797,
    0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195,
    2.1551, -2.1551,
)


def _device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rdtype(parity: bool):
    return torch.float64 if parity else torch.float32


def _cdtype(parity: bool):
    return torch.complex128 if parity else torch.complex64


def wrap_azimuth_deg(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + 180.0, 360.0) - 180.0


def wrap_zenith_deg(x: torch.Tensor) -> torch.Tensor:
    y = torch.remainder(x, 360.0)
    return torch.where(y > 180.0, 360.0-y, y)


def vector_center_angles_deg(v: torch.Tensor):
    """
    MATLAB:
        [az,el] = cart2sph(...)
        phi_hat = rad2deg(az)
        theta_hat = 90-rad2deg(el)
    """
    x,y,z = v.unbind(-1)
    az = torch.atan2(y,x)
    rxy = torch.sqrt(x*x+y*y)
    el = torch.atan2(z,rxy)
    phi = torch.rad2deg(az)
    theta = 90.0 - torch.rad2deg(el)
    return phi,theta


def rhohat_deg(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    pr = torch.deg2rad(phi)
    tr = torch.deg2rad(theta)
    st = torch.sin(tr)
    return torch.stack(
        (st*torch.cos(pr), st*torch.sin(pr), torch.cos(tr)),
        dim=-1,
    )


def dualpol_isotropic_field(n_ports: int, *, device, parity: bool):
    """
    For the locked isotropic Model-2 arrays with zero orientation,
    getFieldTerm is angle independent:
        first polarization block  (+45 deg): [cos45; +sin45]
        second polarization block (-45 deg): [cos45; -sin45]
    """
    if n_ports % 2:
        raise ValueError("dual-polarized port count must be even")
    rd = _rdtype(parity)
    c = torch.tensor(math.sqrt(0.5),dtype=rd,device=device)
    half = n_ports//2
    f = torch.empty((2,n_ports),dtype=rd,device=device)
    f[0,:] = c
    f[1,:half] = c
    f[1,half:] = -c
    return f


def location_terms_from_angles(
    phi: torch.Tensor,
    theta: torch.Tensor,
    positions_lambda: torch.Tensor,
    *,
    parity: bool,
) -> torch.Tensor:
    rhat = rhohat_deg(phi,theta)
    phase = 2.0*math.pi*torch.einsum(
        "...d,dp->...p",rhat,positions_lambda
    )
    return torch.exp(1j*phase).to(_cdtype(parity))


@torch.inference_mode()
def generate_channel_from_primitives(
    *,
    tx_spec: ArraySpec,
    rx_spec: ArraySpec,
    a_vector,
    d_vector,
    fc: float,
    K: float,
    is_los: bool,
    M: int,
    L: int,
    c_ASA: float,
    c_ZSA: float,
    c_ASD: float,
    c_ZSD: float,
    mu_offset_ZOD: float,
    XPR,
    ASAv,
    ZSAv,
    ASDv,
    ZSDv,
    cluster_offsets,
    Phi,
    device=None,
    parity: bool = True,
) -> torch.Tensor:
    """
    Generate H[N,U,S] from fixed MATLAB-exported random primitives.

    Shapes:
        XPR             [N,M*L]
        ASAv...         [N]
        cluster_offsets [N,M,4]
            order = [dphi_AOA,dtheta_ZOA,dphi_AOD,dtheta_ZOD]
        Phi             [N,M,L,2,2] actual phase values in radians
    """
    if L != 20:
        raise ValueError("current project ray count L must be 20")
    if M <= 0:
        raise ValueError("M must be positive")

    dev = _device(device)
    rd = _rdtype(parity)
    cd = _cdtype(parity)

    a = torch.as_tensor(a_vector,dtype=rd,device=dev).reshape(3)
    d = torch.as_tensor(d_vector,dtype=rd,device=dev).reshape(3)

    XPR = torch.as_tensor(XPR,dtype=rd,device=dev)
    ASAv = torch.as_tensor(ASAv,dtype=rd,device=dev).reshape(-1)
    ZSAv = torch.as_tensor(ZSAv,dtype=rd,device=dev).reshape(-1)
    ASDv = torch.as_tensor(ASDv,dtype=rd,device=dev).reshape(-1)
    ZSDv = torch.as_tensor(ZSDv,dtype=rd,device=dev).reshape(-1)
    offsets = torch.as_tensor(cluster_offsets,dtype=rd,device=dev)
    Phi = torch.as_tensor(Phi,dtype=rd,device=dev)

    N = ASAv.numel()
    if XPR.shape != (N,M*L):
        raise ValueError(f"XPR expected {(N,M*L)}, got {tuple(XPR.shape)}")
    if offsets.shape != (N,M,4):
        raise ValueError(
            f"cluster_offsets expected {(N,M,4)}, got {tuple(offsets.shape)}"
        )
    if Phi.shape != (N,M,L,2,2):
        raise ValueError(f"Phi expected {(N,M,L,2,2)}, got {tuple(Phi.shape)}")

    S = 2*tx_spec.M*tx_spec.N
    U = 2*rx_spec.M*rx_spec.N

    tx_pos = build_dualpol_positions_lambda(
        tx_spec,device=dev,parity=parity
    )
    rx_pos = build_dualpol_positions_lambda(
        rx_spec,device=dev,parity=parity
    )

    tx_field = dualpol_isotropic_field(S,device=dev,parity=parity)
    rx_field = dualpol_isotropic_field(U,device=dev,parity=parity)

    phi_aoa_hat,theta_zoa_hat = vector_center_angles_deg(a)
    phi_aod_hat,theta_zod_hat = vector_center_angles_deg(d)

    # ------------------------------------------------------------
    # LOS component — exact current generate_channel_train branch.
    # ------------------------------------------------------------
    phi_aoa0 = wrap_azimuth_deg(phi_aoa_hat)
    phi_aod0 = wrap_azimuth_deg(phi_aod_hat)
    theta_zoa0 = wrap_zenith_deg(theta_zoa_hat)
    theta_zod0 = wrap_zenith_deg(theta_zod_hat)

    tx_loc0 = location_terms_from_angles(
        phi_aod0,theta_zod0,tx_pos,parity=parity
    ).reshape(S)
    rx_loc0 = location_terms_from_angles(
        phi_aoa0,theta_zoa0,rx_pos,parity=parity
    ).reshape(U)

    pol_los = torch.tensor(
        [[1.0,0.0],[0.0,-1.0]],
        dtype=rd,device=dev
    )

    field_los = torch.einsum(
        "au,ab,bs->us",rx_field,pol_los,tx_field
    ).to(cd)
    location_los = rx_loc0[:,None] * tx_loc0[None,:]
    H_LOS = field_los * location_los

    # ------------------------------------------------------------
    # NLOS ray angles.
    # ------------------------------------------------------------
    alpha = torch.tensor(ALPHA,dtype=rd,device=dev)
    if alpha.numel() != L:
        raise ValueError("alpha/L mismatch")

    dphi_aoa = offsets[:,:,0]
    dtheta_zoa = offsets[:,:,1]
    dphi_aod = offsets[:,:,2]
    dtheta_zod = offsets[:,:,3]

    phi_aoa = (
        phi_aoa_hat
        + dphi_aoa[:,:,None]
        + torch.as_tensor(c_ASA,dtype=rd,device=dev)*alpha[None,None,:]
    )
    theta_zoa = (
        theta_zoa_hat
        + dtheta_zoa[:,:,None]
        + torch.as_tensor(c_ZSA,dtype=rd,device=dev)*alpha[None,None,:]
    )
    phi_aod = (
        phi_aod_hat
        + dphi_aod[:,:,None]
        + torch.as_tensor(c_ASD,dtype=rd,device=dev)*alpha[None,None,:]
    )
    theta_zod = (
        theta_zod_hat
        + torch.as_tensor(mu_offset_ZOD,dtype=rd,device=dev)
        + dtheta_zod[:,:,None]
        + torch.as_tensor(c_ZSD,dtype=rd,device=dev)*alpha[None,None,:]
    )

    phi_aoa = wrap_azimuth_deg(phi_aoa)
    phi_aod = wrap_azimuth_deg(phi_aod)
    theta_zoa = wrap_zenith_deg(theta_zoa)
    theta_zod = wrap_zenith_deg(theta_zod)

    tx_loc = location_terms_from_angles(
        phi_aod,theta_zod,tx_pos,parity=parity
    )  # [N,M,L,S]
    rx_loc = location_terms_from_angles(
        phi_aoa,theta_zoa,rx_pos,parity=parity
    )  # [N,M,L,U]

    # Current MATLAB:
    # kappa=10^(XPR/10)
    # P = [[1,1/sqrt(kappa)],[1/sqrt(kappa),1]] .* exp(j Phi)
    xpr = XPR.reshape(N,M,L)
    cross = torch.pow(
        torch.tensor(10.0,dtype=rd,device=dev),
        -xpr/20.0
    )
    amp = torch.ones((N,M,L,2,2),dtype=rd,device=dev)
    amp[:,:,:,0,1] = cross
    amp[:,:,:,1,0] = cross
    pol = amp.to(cd) * torch.exp(1j*Phi).to(cd)

    # Avoid materializing [N,M,L,U,S].
    #
    # MATLAB ray:
    #   fieldMatrix    = rxField.' * P * txField
    #   locationMatrix = rxLoc.'   * txLoc
    #   H += fieldMatrix .* locationMatrix
    #
    # Since fields are real here, .' is the same raw transpose.
    tx_weighted = (
        tx_loc[:,:,:,None,:]
        * tx_field[None,None,None,:,:].to(cd)
    )                                  # [N,M,L,2,S]
    temp = torch.einsum(
        "nmlab,nmlbs->nmlas",pol,tx_weighted
    )                                  # [N,M,L,2,S]

    rx_weighted = (
        rx_loc[:,:,:,None,:]
        * rx_field[None,None,None,:,:].to(cd)
    )                                  # [N,M,L,2,U]

    H_NLOS = torch.einsum(
        "nmlau,nmlas->nus",rx_weighted,temp
    )
    H_NLOS = H_NLOS / math.sqrt(M*L)

    Kt = torch.as_tensor(K,dtype=rd,device=dev)
    if is_los:
        H = (
            torch.sqrt(1.0/(Kt+1.0)).to(cd)*H_NLOS
            + torch.sqrt(Kt/(Kt+1.0)).to(cd)*H_LOS[None,:,:]
        )
    else:
        H = H_NLOS

    return H


@torch.inference_mode()
def generate_cascaded_ch(
    h_gnb2ris: torch.Tensor,
    h_ris2ue: torch.Tensor,
    gamma,
) -> torch.Tensor:
    """
    Exact vectorized port:
        F[n] = H_RU[n] diag(gamma) H_BR[n]
    """
    g = torch.as_tensor(
        gamma,dtype=h_gnb2ris.dtype,device=h_gnb2ris.device
    ).reshape(-1)
    return torch.einsum("nri,i,nit->nrt",h_ris2ue,g,h_gnb2ris)


@torch.inference_mode()
def apply_precoder_empirical(F: torch.Tensor, W) -> torch.Tensor:
    """
    Empirical part of generate_eff_ch:
        Feff[n] = F[n] W
    nl=1 in the current project.
    """
    w = torch.as_tensor(W,dtype=F.dtype,device=F.device)
    if w.ndim == 1:
        w = w[:,None]
    return torch.einsum("nrt,tv->nrv",F,w)


@torch.inference_mode()
def empirical_snr_samples(Feff: torch.Tensor) -> torch.Tensor:
    """
    MATLAB:
        Y = squeeze(sum(abs(Feff).^2,2))
    """
    return torch.sum(torch.abs(Feff)**2,dim=1).squeeze(-1).real


def _rel_fro(a,b):
    a=np.asarray(a)
    b=np.asarray(b)
    den=max(np.linalg.norm(b.ravel()),np.finfo(float).eps)
    return float(np.linalg.norm((a-b).ravel())/den)


def _max_abs(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


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


@torch.inference_mode()
def compare_stage8b1_case(
    mat_path: str,
    *,
    device=None,
    parity: bool = True,
) -> Dict[str,Any]:
    from scipy.io import loadmat

    M = loadmat(mat_path,squeeze_me=False)
    dev=_device(device)

    nT1=_si(M,"nT1"); nT2=_si(M,"nT2")
    nR1=_si(M,"nR1"); nR2=_si(M,"nR2")
    nRISx=_si(M,"nRISx"); nRISy=_si(M,"nRISy")
    fc=_sc(M,"fc")

    results={}

    link_cfgs = {
        "BR":{
            "tx":ArraySpec(nT1,nT2),
            "rx":ArraySpec(nRISx,nRISy),
            "a":np.asarray(M["ris2gnb"]).reshape(-1),
            "d":np.asarray(M["gnb2ris"]).reshape(-1),
        },
        "RU":{
            "tx":ArraySpec(nRISx,nRISy),
            "rx":ArraySpec(nR1,nR2),
            "a":np.asarray(M["ue2ris"]).reshape(-1),
            "d":np.asarray(M["ris2ue"]).reshape(-1),
        },
    }

    pyH={}

    for prefix,cfg in link_cfgs.items():
        H = generate_channel_from_primitives(
            tx_spec=cfg["tx"],
            rx_spec=cfg["rx"],
            a_vector=cfg["a"],
            d_vector=cfg["d"],
            fc=fc,
            K=_sc(M,f"{prefix}_K"),
            is_los=_sb(M,f"{prefix}_isLOS"),
            M=_si(M,f"{prefix}_M"),
            L=_si(M,f"{prefix}_L"),
            c_ASA=_sc(M,f"{prefix}_c_ASA"),
            c_ZSA=_sc(M,f"{prefix}_c_ZSA"),
            c_ASD=_sc(M,f"{prefix}_c_ASD"),
            c_ZSD=_sc(M,f"{prefix}_c_ZSD"),
            mu_offset_ZOD=_sc(M,f"{prefix}_mu_offset_ZOD"),
            XPR=np.asarray(M[f"{prefix}_XPR"]),
            ASAv=np.asarray(M[f"{prefix}_ASAv"]).reshape(-1),
            ZSAv=np.asarray(M[f"{prefix}_ZSAv"]).reshape(-1),
            ASDv=np.asarray(M[f"{prefix}_ASDv"]).reshape(-1),
            ZSDv=np.asarray(M[f"{prefix}_ZSDv"]).reshape(-1),
            cluster_offsets=np.asarray(M[f"{prefix}_clusterOffsets"]),
            Phi=np.asarray(M[f"{prefix}_Phi"]),
            device=dev,
            parity=parity,
        )

        py = H.detach().cpu().numpy()
        ref=np.asarray(M[f"H{prefix}"])
        pyH[prefix]=H

        results[f"{prefix}_H_relFro"]=_rel_fro(py,ref)
        results[f"{prefix}_H_maxAbs"]=_max_abs(py,ref)

        # Stage-1 deterministic outputs are rechecked here as integration guards.
        moments=generate_channel_moments_batch(
            tx_spec=cfg["tx"],rx_spec=cfg["rx"],
            a_vectors=cfg["a"][None,:],
            d_vectors=cfg["d"][None,:],
            carrier_frequency=fc,
            K=_sc(M,f"{prefix}_K"),
            mu_xpr=_sc(M,f"{prefix}_mu_XPR"),
            sigma_xpr=_sc(M,f"{prefix}_sigma_XPR"),
            device=dev,parity=parity,
        )
        mu=moments["muH"][0].detach().cpu().numpy()
        sig=float(moments["sigma2H"][0].detach().cpu())
        dT=moments["dbarT"][0].detach().cpu().numpy()
        dR=moments["dbarR"][0].detach().cpu().numpy()

        results[f"{prefix}_muH_relFro"]=_rel_fro(
            mu,np.asarray(M[f"mu{prefix}"])
        )
        results[f"{prefix}_sigma2H_rel"] = abs(
            sig-_sc(M,f"sigma2{prefix}")
        )/max(abs(_sc(M,f"sigma2{prefix}")),np.finfo(float).eps)
        results[f"{prefix}_dbarT_relFro"]=_rel_fro(
            dT,np.asarray(M[f"dbarT{prefix}"])
        )
        results[f"{prefix}_dbarR_relFro"]=_rel_fro(
            dR,np.asarray(M[f"dbarR{prefix}"])
        )

    # Optional full stochastic-chain integration outputs exported by MATLAB.
    if "gamma" in M and "F" in M:
        g=torch.as_tensor(
            np.asarray(M["gamma"]).reshape(-1),
            dtype=_cdtype(parity),device=dev
        )
        F=generate_cascaded_ch(pyH["BR"],pyH["RU"],g)
        Fnp=F.detach().cpu().numpy()
        results["F_relFro"]=_rel_fro(Fnp,np.asarray(M["F"]))
        results["F_maxAbs"]=_max_abs(Fnp,np.asarray(M["F"]))

        if "W" in M and "Feff" in M:
            W=np.asarray(M["W"]).reshape(-1,1)
            Feff=apply_precoder_empirical(F,W)
            Y=empirical_snr_samples(Feff)
            results["Feff_relFro"]=_rel_fro(
                Feff.detach().cpu().numpy(),
                np.asarray(M["Feff"])
            )
            results["Y_relFro"]=_rel_fro(
                Y.detach().cpu().numpy().reshape(-1),
                np.asarray(M["Y"]).reshape(-1)
            )

    results["scenario"]= _sstr(M,"scenario")
    results["nT"]=2*nT1*nT2
    results["nR"]=2*nR1*nR2
    results["nRIS"]=2*nRISx*nRISy
    results["N"]=_si(M,"N")
    return results
