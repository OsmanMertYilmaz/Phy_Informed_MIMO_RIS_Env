
from __future__ import annotations

import math
import numpy as np
import torch

from ris_env.antenna import ArraySpec
from ris_env.channel_realizations import LinkConfig

from ris_env import channel_primitives as cp

from ris_env.case1 import (
    generate_case1_native_link_chunk,
)


# ======================================================================
# Single-hop analytical Case-1 moments
# ======================================================================

@torch.inference_mode()
def case1_single_hop_moments(
    cfg: LinkConfig,
    *,
    device,
    parity=False,
):
    """
    Returns

        muH    [U,S]
        varH   [U,S]

    for the Case-1 channel.

    With the currently locked dual-pol isotropic array,
    the marginal NLOS variance is identical for every antenna pair.
    """

    dev = torch.device(device)

    rd = (
        torch.float64
        if parity
        else torch.float32
    )

    cd = (
        torch.complex128
        if parity
        else torch.complex64
    )

    S = 2 * cfg.tx_spec.M * cfg.tx_spec.N
    U = 2 * cfg.rx_spec.M * cfg.rx_spec.N

    # --------------------------------------------------------------
    # E[kappa^{-1}]
    #
    # XPR_dB ~ N(mu_XPR, sigma_XPR^2)
    #
    # kappa^{-1} = 10^{-XPR/10}
    # --------------------------------------------------------------

    a = math.log(10.0) / 10.0

    E_inv_kappa = math.exp(
        -a * cfg.mu_XPR
        + 0.5 * a * a * cfg.sigma_XPR**2
    )

    # Dual-pol isotropic:
    #
    # |F_rx,a F_tx,b|^2 = 1/4
    #
    # two co-pol terms    -> 2 * 1/4
    # two cross-pol terms -> 2 * 1/4 * E[1/kappa]
    #
    sigma2_nlos = 0.5 * (
        1.0 + E_inv_kappa
    )

    # --------------------------------------------------------------
    # Mean
    # --------------------------------------------------------------

    if not cfg.is_los:

        muH = torch.zeros(
            (U, S),
            dtype=cd,
            device=dev,
        )

        sigma2 = sigma2_nlos

    else:

        av = torch.as_tensor(
            cfg.a_vector,
            dtype=rd,
            device=dev,
        ).reshape(3)

        dv = torch.as_tensor(
            cfg.d_vector,
            dtype=rd,
            device=dev,
        ).reshape(3)

        tx_pos = cp.build_dualpol_positions_lambda(
            cfg.tx_spec,
            device=dev,
            parity=parity,
        )

        rx_pos = cp.build_dualpol_positions_lambda(
            cfg.rx_spec,
            device=dev,
            parity=parity,
        )

        tx_field = cp.dualpol_isotropic_field(
            S,
            device=dev,
            parity=parity,
        )

        rx_field = cp.dualpol_isotropic_field(
            U,
            device=dev,
            parity=parity,
        )

        phi_aoa_hat, theta_zoa_hat = (
            cp.vector_center_angles_deg(av)
        )

        phi_aod_hat, theta_zod_hat = (
            cp.vector_center_angles_deg(dv)
        )

        phi_aoa0 = cp.wrap_azimuth_deg(
            phi_aoa_hat
        )

        phi_aod0 = cp.wrap_azimuth_deg(
            phi_aod_hat
        )

        theta_zoa0 = cp.wrap_zenith_deg(
            theta_zoa_hat
        )

        theta_zod0 = cp.wrap_zenith_deg(
            theta_zod_hat
        )

        tx_loc0 = cp.location_terms_from_angles(
            phi_aod0,
            theta_zod0,
            tx_pos,
            parity=parity,
        ).reshape(S)

        rx_loc0 = cp.location_terms_from_angles(
            phi_aoa0,
            theta_zoa0,
            rx_pos,
            parity=parity,
        ).reshape(U)

        pol_los = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=rd,
            device=dev,
        )

        field_los = torch.einsum(
            "au,ab,bs->us",
            rx_field,
            pol_los,
            tx_field,
        ).to(cd)

        H_LOS = (
            field_los
            * rx_loc0[:, None]
            * tx_loc0[None, :]
        )

        K = float(cfg.K)

        muH = (
            math.sqrt(K / (K + 1.0))
            * H_LOS
        )

        sigma2 = (
            sigma2_nlos
            / (K + 1.0)
        )

    varH = torch.full(
        (U, S),
        float(sigma2),
        dtype=rd,
        device=dev,
    )

    return muH, varH


# ======================================================================
# Cascaded Case-1 analytical moments
# ======================================================================

@torch.inference_mode()
def case1_feff_moments(
    mu_h,
    var_h,
    mu_g,
    var_g,
    W,
    gamma,
):
    """
    BR:
        h[i,t]

    RU:
        g[r,i]

    Feff:
        f[r] = sum_i gamma[i] g[r,i] sum_t W[t] h[i,t]

    Case-1 pair independence is assumed.
    """

    cd = mu_h.dtype
    rd = mu_h.real.dtype

    W = W.to(
        device=mu_h.device,
        dtype=cd,
    )

    gamma = gamma.to(
        device=mu_h.device,
        dtype=cd,
    )

    # --------------------------------------------------------------
    # BR after precoder:
    #
    # q_i = sum_t W_t h_it
    # --------------------------------------------------------------

    m = torch.einsum(
        "it,t->i",
        mu_h,
        W,
    )

    v = torch.einsum(
        "it,t->i",
        var_h,
        torch.abs(W)**2,
    ).real

    # --------------------------------------------------------------
    # Feff mean
    # --------------------------------------------------------------

    mu_f = torch.einsum(
        "i,ri,i->r",
        gamma,
        mu_g,
        m,
    )

    R = mu_g.shape[0]
    I = mu_g.shape[1]

    abs_gamma2 = (
        torch.abs(gamma)**2
    ).to(rd)

    # --------------------------------------------------------------
    # First covariance contribution:
    #
    # sum_i |gamma_i|² v_i
    #       mu_g[:,i] mu_g[:,i]^H
    # --------------------------------------------------------------

    weight = (
        abs_gamma2 * v
    )

    Sigma = torch.einsum(
        "i,ri,si->rs",
        weight.to(cd),
        mu_g,
        mu_g.conj(),
    )

    # --------------------------------------------------------------
    # Diagonal contribution from RU randomness:
    #
    # sum_i |gamma_i|² var_g[r,i]
    #       ( |m_i|² + v_i )
    # --------------------------------------------------------------

    second_moment_q = (
        torch.abs(m)**2
        + v
    ).real

    diag_extra = torch.einsum(
        "i,ri,i->r",
        abs_gamma2,
        var_g,
        second_moment_q,
    )

    Sigma = (
        Sigma
        + torch.diag(
            diag_extra.to(cd)
        )
    )

    # Numerical Hermitian cleanup.
    Sigma = 0.5 * (
        Sigma
        + Sigma.conj().T
    )

    return mu_f, Sigma


# ======================================================================
# Metrics
# ======================================================================

def relative_vector_error(x, ref):
    den = torch.linalg.vector_norm(
        ref
    ).clamp_min(1e-20)

    return float(
        (
            torch.linalg.vector_norm(x - ref)
            / den
        ).cpu()
    )


def relative_fro_error(x, ref):
    den = torch.linalg.matrix_norm(
        ref
    ).clamp_min(1e-20)

    return float(
        (
            torch.linalg.matrix_norm(x - ref)
            / den
        ).cpu()
    )


# ======================================================================
# Main
# ======================================================================

@torch.inference_mode()
def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ==============================================================
    # Synthetic but realistic two-hop topology
    #
    # gNB:
    #   ArraySpec(1,1) -> 2 ports
    #
    # RIS:
    #   ArraySpec(4,4) -> 32 dual-pol ports
    #
    # UE:
    #   ArraySpec(1,1) -> 2 ports
    # ==============================================================

    gnb = ArraySpec(1, 1)
    ris = ArraySpec(4, 4)
    ue  = ArraySpec(1, 1)

    # --------------------------------------------------------------
    # BR — LOS example
    # --------------------------------------------------------------

    cfg_br = LinkConfig(
        tx_spec=gnb,
        rx_spec=ris,

        a_vector=np.array(
            [-1.0, 0.0, 0.0]
        ),

        d_vector=np.array(
            [1.0, 0.0, 0.0]
        ),

        fc=3.5e9,

        K=5.0,
        is_los=True,

        M=4,
        L=20,

        mu_XPR=8.0,
        sigma_XPR=3.0,

        mu_ASA=1.30,
        sigma_ASA=0.10,

        mu_ZSA=1.00,
        sigma_ZSA=0.10,

        mu_ASD=1.30,
        sigma_ASD=0.10,

        mu_ZSD=1.00,
        sigma_ZSD=0.10,

        c_ASA=1.0,
        c_ZSA=1.0,
        c_ASD=1.0,
        c_ZSD=1.0,

        mu_offset_ZOD=0.0,
    )

    # --------------------------------------------------------------
    # RU — LOS example
    # --------------------------------------------------------------

    cfg_ru = LinkConfig(
        tx_spec=ris,
        rx_spec=ue,

        a_vector=np.array(
            [-1.0, 0.0, 0.0]
        ),

        d_vector=np.array(
            [1.0, 0.0, 0.0]
        ),

        fc=3.5e9,

        K=3.0,
        is_los=True,

        M=4,
        L=20,

        mu_XPR=7.0,
        sigma_XPR=3.0,

        mu_ASA=1.30,
        sigma_ASA=0.10,

        mu_ZSA=1.00,
        sigma_ZSA=0.10,

        mu_ASD=1.30,
        sigma_ASD=0.10,

        mu_ZSD=1.00,
        sigma_ZSD=0.10,

        c_ASA=1.0,
        c_ZSA=1.0,
        c_ASD=1.0,
        c_ZSD=1.0,

        mu_offset_ZOD=0.0,
    )

    # ==============================================================
    # Fixed normalized precoder
    # ==============================================================

    nT = 2 * gnb.M * gnb.N
    nRIS = 2 * ris.M * ris.N
    nR = 2 * ue.M * ue.N

    idx = torch.arange(
        nT,
        device=device,
        dtype=torch.float32,
    )

    W = torch.exp(
        1j * 2.0 * math.pi
        * idx / max(nT, 1)
    ).to(torch.complex64)

    W = W / torch.linalg.vector_norm(W)

    # ==============================================================
    # Fixed RIS pattern
    #
    # Alternating 45 / 135 deg.
    # Unit amplitude for this moment-validation test.
    # ==============================================================

    z = torch.arange(
        nRIS,
        device=device,
    ) % 2

    phi = torch.where(
        z == 0,
        torch.tensor(
            math.pi / 4,
            device=device,
        ),
        torch.tensor(
            3 * math.pi / 4,
            device=device,
        ),
    )

    gamma = torch.exp(
        1j * phi
    ).to(torch.complex64)

    # ==============================================================
    # Analytical single-hop moments
    # ==============================================================

    mu_h, var_h = (
        case1_single_hop_moments(
            cfg_br,
            device=device,
            parity=False,
        )
    )

    mu_g, var_g = (
        case1_single_hop_moments(
            cfg_ru,
            device=device,
            parity=False,
        )
    )

    print("\nShapes")
    print("mu_h :", tuple(mu_h.shape))
    print("mu_g :", tuple(mu_g.shape))
    print("W    :", tuple(W.shape))
    print("gamma:", tuple(gamma.shape))

    assert mu_h.shape == (
        nRIS,
        nT,
    )

    assert mu_g.shape == (
        nR,
        nRIS,
    )

    # ==============================================================
    # Analytical Feff moments
    # ==============================================================

    mu_f_analytic, Sigma_analytic = (
        case1_feff_moments(
            mu_h,
            var_h,
            mu_g,
            var_g,
            W,
            gamma,
        )
    )

    muY_analytic = (
        torch.trace(
            Sigma_analytic
        ).real
        + torch.sum(
            torch.abs(mu_f_analytic)**2
        )
    )

    # Gaussian/Wick prediction for Y variance.
    varY_wick = (
        torch.trace(
            Sigma_analytic
            @ Sigma_analytic
        ).real
        +
        2.0
        * torch.real(
            torch.vdot(
                mu_f_analytic,
                Sigma_analytic
                @ mu_f_analytic,
            )
        )
    )

    # ==============================================================
    # Monte Carlo
    #
    # Do this in chunks so we never hold the full dataset.
    # ==============================================================

    N_TOTAL = 32768
    MC_CHUNK = 1024
    PAIR_CHUNK = 8

    gen_br = torch.Generator(
        device=device
    )

    gen_ru = torch.Generator(
        device=device
    )

    gen_br.manual_seed(112233)
    gen_ru.manual_seed(998877)

    sum_f = torch.zeros(
        nR,
        dtype=torch.complex128,
        device=device,
    )

    sum_ff = torch.zeros(
        (nR, nR),
        dtype=torch.complex128,
        device=device,
    )

    sum_y = 0.0
    sum_y2 = 0.0

    count = 0

    for start in range(
        0,
        N_TOTAL,
        MC_CHUNK,
    ):

        n = min(
            MC_CHUNK,
            N_TOTAL - start,
        )

        HBR = generate_case1_native_link_chunk(
            cfg_br,
            n,
            generator=gen_br,
            pair_chunk=PAIR_CHUNK,
            device=device,
            parity=False,
        )

        HRU = generate_case1_native_link_chunk(
            cfg_ru,
            n,
            generator=gen_ru,
            pair_chunk=PAIR_CHUNK,
            device=device,
            parity=False,
        )

        # BR after W:
        #
        # q[n,i] = sum_t HBR[n,i,t] W[t]
        q = torch.einsum(
            "nit,t->ni",
            HBR,
            W,
        )

        # Cascaded Feff directly:
        #
        # Feff[n,r]
        # = sum_i HRU[n,r,i] gamma[i] q[n,i]
        Feff = torch.einsum(
            "nri,i,ni->nr",
            HRU,
            gamma,
            q,
        )

        Y = torch.sum(
            torch.abs(Feff)**2,
            dim=1,
        ).to(torch.float64)

        Fd = Feff.to(
            torch.complex128
        )

        sum_f += Fd.sum(
            dim=0
        )

        sum_ff += torch.einsum(
            "nr,ns->rs",
            Fd,
            Fd.conj(),
        )

        sum_y += float(
            Y.sum().cpu()
        )

        sum_y2 += float(
            torch.sum(Y*Y).cpu()
        )

        count += n

        del (
            HBR,
            HRU,
            q,
            Feff,
            Fd,
            Y,
        )

    # ==============================================================
    # MC moments
    # ==============================================================

    mu_f_mc = (
        sum_f / count
    )

    E_ff = (
        sum_ff / count
    )

    Sigma_mc = (
        E_ff
        - mu_f_mc[:, None]
        * mu_f_mc.conj()[None, :]
    )

    Sigma_mc = 0.5 * (
        Sigma_mc
        + Sigma_mc.conj().T
    )

    meanY_mc = (
        sum_y / count
    )

    varY_mc = (
        sum_y2 / count
        - meanY_mc**2
    )

    # ==============================================================
    # Errors
    # ==============================================================

    mu_scale = torch.sqrt(
        torch.trace(
            Sigma_analytic.to(
                torch.complex128
            )
        ).real
    ).clamp_min(1e-20)

    mean_f_abs_error = float(
        (
            torch.linalg.vector_norm(
                mu_f_mc
                - mu_f_analytic.to(
                    torch.complex128
                )
            )
            / mu_scale
        ).cpu()
    )

    covariance_rel_error = (
        relative_fro_error(
            Sigma_mc,
            Sigma_analytic.to(
                torch.complex128
            ),
        )
    )

    meanY_ape = (
        abs(
            meanY_mc
            - float(muY_analytic.cpu())
        )
        / max(
            abs(meanY_mc),
            1e-20,
        )
    )

    wick_ratio = (
        varY_mc
        / float(varY_wick.cpu())
    )

    wick_ape = (
        abs(
            float(varY_wick.cpu())
            - varY_mc
        )
        / max(
            abs(varY_mc),
            1e-20,
        )
    )

    # ==============================================================
    # Report
    # ==============================================================

    print("\n" + "=" * 78)
    print("CASE 1 CASCADED MOMENT VALIDATION")
    print("=" * 78)

    print(f"MC samples        : {count}")
    print(f"nT                : {nT}")
    print(f"nRIS              : {nRIS}")
    print(f"nR                : {nR}")

    print("\n--- Feff first/second moments ---")

    print(
        "Mean vector normalized error : "
        f"{100*mean_f_abs_error:.3f}%"
    )

    print(
        "Covariance relative Fro error: "
        f"{100*covariance_rel_error:.3f}%"
    )

    print("\n--- Y mean ---")

    print(
        f"Analytic meanY : "
        f"{float(muY_analytic.cpu()):.8e}"
    )

    print(
        f"MC meanY       : "
        f"{meanY_mc:.8e}"
    )

    print(
        f"Mean APE       : "
        f"{100*meanY_ape:.3f}%"
    )

    print("\n--- Y variance ---")

    print(
        f"Wick varY      : "
        f"{float(varY_wick.cpu()):.8e}"
    )

    print(
        f"MC varY        : "
        f"{varY_mc:.8e}"
    )

    print(
        f"MC / Wick      : "
        f"{wick_ratio:.6f}"
    )

    print(
        f"Wick APE       : "
        f"{100*wick_ape:.3f}%"
    )

    print("\nAnalytic Feff mean:")
    print(
        mu_f_analytic.detach().cpu().numpy()
    )

    print("\nMC Feff mean:")
    print(
        mu_f_mc.detach().cpu().numpy()
    )

    print("\nAnalytic |Sigma|:")
    print(
        np.abs(
            Sigma_analytic.detach()
            .cpu()
            .numpy()
        )
    )

    print("\nMC |Sigma|:")
    print(
        np.abs(
            Sigma_mc.detach()
            .cpu()
            .numpy()
        )
    )

    print("\n" + "=" * 78)

    # Exact second-moment check.
    if meanY_ape < 0.02:
        print(
            "PASS meanY: Case-1 analytic "
            "second moment agrees with MC."
        )
    else:
        print(
            "CHECK meanY: error is larger "
            "than expected."
        )

    # Wick is NOT asserted here.
    # This is deliberately a measurement:
    #
    # product channel may still be non-Gaussian.
    if abs(wick_ratio - 1.0) < 0.10:
        print(
            "WICK RESULT: close to Gaussian "
            "(within 10%)."
        )
    else:
        print(
            "WICK RESULT: measurable "
            "non-Gaussian fourth-moment gap remains."
        )


if __name__ == "__main__":
    main()
