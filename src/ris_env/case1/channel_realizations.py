
from __future__ import annotations

import math
from typing import Dict

import torch

from ris_env import channel_primitives as cp
from ris_env.channel_realizations import LinkConfig


# ============================================================================
# CASE 1
#
# Random NLOS polarization phases are independent for every antenna pair:
#
#   Phi[n,m,l,u,s,a,b]
#
# where
#   u = Rx port
#   s = Tx port
#   a,b = polarization indices
#
# We NEVER materialize the full tensor above.
# Antenna pairs are processed in chunks.
# ============================================================================


@torch.inference_mode()
def sample_case1_primitives(
    cfg: LinkConfig,
    N: int,
    *,
    generator: torch.Generator,
    device=None,
    parity: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Sample all Case-1 random primitives EXCEPT polarization phase.

    Shared with the current model:
        XPR
        ASA / ZSA / ASD / ZSD
        cluster angular offsets

    Case-1 difference:
        polarization phases are NOT sampled here as [N,M,L,2,2].

        They are generated later independently for each
        Rx/Tx antenna pair inside
        generate_case1_channel_from_primitives().
    """

    if N <= 0:
        raise ValueError("N must be positive")

    dev = cp._device(device)
    rd = cp._rdtype(parity)

    def randn(*shape):
        return torch.randn(
            shape,
            dtype=rd,
            device=dev,
            generator=generator,
        )

    XPR = (
        cfg.mu_XPR
        + cfg.sigma_XPR * randn(N, cfg.M * cfg.L)
    )

    ASAv = torch.pow(
        torch.tensor(10.0, dtype=rd, device=dev),
        cfg.mu_ASA + cfg.sigma_ASA * randn(N),
    ).clamp_max(104.0)

    ZSAv = torch.pow(
        torch.tensor(10.0, dtype=rd, device=dev),
        cfg.mu_ZSA + cfg.sigma_ZSA * randn(N),
    ).clamp_max(52.0)

    ASDv = torch.pow(
        torch.tensor(10.0, dtype=rd, device=dev),
        cfg.mu_ASD + cfg.sigma_ASD * randn(N),
    ).clamp_max(104.0)

    ZSDv = torch.pow(
        torch.tensor(10.0, dtype=rd, device=dev),
        cfg.mu_ZSD + cfg.sigma_ZSD * randn(N),
    ).clamp_max(52.0)

    offsets = torch.empty(
        (N, cfg.M, 4),
        dtype=rd,
        device=dev,
    )

    offsets[:, :, 0] = (
        randn(N, cfg.M) * ASAv[:, None] / 7.0
    )
    offsets[:, :, 1] = (
        randn(N, cfg.M) * ZSAv[:, None] / 7.0
    )
    offsets[:, :, 2] = (
        randn(N, cfg.M) * ASDv[:, None] / 7.0
    )
    offsets[:, :, 3] = (
        randn(N, cfg.M) * ZSDv[:, None] / 7.0
    )

    return {
        "XPR": XPR,
        "ASAv": ASAv,
        "ZSAv": ZSAv,
        "ASDv": ASDv,
        "ZSDv": ZSDv,
        "cluster_offsets": offsets,
    }


@torch.inference_mode()
def generate_case1_channel_from_primitives(
    *,
    tx_spec,
    rx_spec,
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
    generator: torch.Generator,
    pair_chunk: int = 8,
    device=None,
    parity: bool = False,
) -> torch.Tensor:
    """
    Generate Case-1 H[N,U,S].

    Main Case-1 assumption
    ----------------------
    For every NLOS ray and every Rx/Tx antenna pair (u,s),

        Phi[n,m,l,u,s,a,b]

    is independently drawn from Uniform[-pi, pi].

    Importantly, the following remain shared exactly as in the
    current channel model:
        - realization LSPs
        - XPR
        - ray/cluster angles
        - antenna geometry
        - spatial steering/location terms
        - LOS component

    Only the NLOS random polarization phase is antenna-pair-specific.

    The complete Phi[N,M,L,U,S,2,2] tensor is never stored.
    """

    if L != 20:
        raise ValueError(
            "current project ray count L must be 20"
        )

    if M <= 0:
        raise ValueError("M must be positive")

    if pair_chunk <= 0:
        raise ValueError("pair_chunk must be positive")

    dev = cp._device(device)
    rd = cp._rdtype(parity)
    cd = cp._cdtype(parity)

    a = torch.as_tensor(
        a_vector,
        dtype=rd,
        device=dev,
    ).reshape(3)

    d = torch.as_tensor(
        d_vector,
        dtype=rd,
        device=dev,
    ).reshape(3)

    XPR = torch.as_tensor(
        XPR,
        dtype=rd,
        device=dev,
    )

    ASAv = torch.as_tensor(
        ASAv,
        dtype=rd,
        device=dev,
    ).reshape(-1)

    ZSAv = torch.as_tensor(
        ZSAv,
        dtype=rd,
        device=dev,
    ).reshape(-1)

    ASDv = torch.as_tensor(
        ASDv,
        dtype=rd,
        device=dev,
    ).reshape(-1)

    ZSDv = torch.as_tensor(
        ZSDv,
        dtype=rd,
        device=dev,
    ).reshape(-1)

    offsets = torch.as_tensor(
        cluster_offsets,
        dtype=rd,
        device=dev,
    )

    N = ASAv.numel()

    if XPR.shape != (N, M * L):
        raise ValueError(
            f"XPR expected {(N, M * L)}, "
            f"got {tuple(XPR.shape)}"
        )

    if offsets.shape != (N, M, 4):
        raise ValueError(
            f"cluster_offsets expected {(N, M, 4)}, "
            f"got {tuple(offsets.shape)}"
        )

    # Number of dual-pol ports.
    S = 2 * tx_spec.M * tx_spec.N
    U = 2 * rx_spec.M * rx_spec.N

    # ----------------------------------------------------------------------
    # Array positions / fields
    # ----------------------------------------------------------------------

    tx_pos = cp.build_dualpol_positions_lambda(
        tx_spec,
        device=dev,
        parity=parity,
    )

    rx_pos = cp.build_dualpol_positions_lambda(
        rx_spec,
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
        cp.vector_center_angles_deg(a)
    )

    phi_aod_hat, theta_zod_hat = (
        cp.vector_center_angles_deg(d)
    )

    # ======================================================================
    # LOS
    #
    # IDENTICAL to current Case-2 implementation.
    # ======================================================================

    phi_aoa0 = cp.wrap_azimuth_deg(phi_aoa_hat)
    phi_aod0 = cp.wrap_azimuth_deg(phi_aod_hat)

    theta_zoa0 = cp.wrap_zenith_deg(theta_zoa_hat)
    theta_zod0 = cp.wrap_zenith_deg(theta_zod_hat)

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

    location_los = (
        rx_loc0[:, None]
        * tx_loc0[None, :]
    )

    H_LOS = (
        field_los
        * location_los
    )

    # ======================================================================
    # NLOS ray angles
    #
    # ALSO identical to Case 2.
    # ======================================================================

    alpha = torch.tensor(
        cp.ALPHA,
        dtype=rd,
        device=dev,
    )

    if alpha.numel() != L:
        raise ValueError("alpha/L mismatch")

    dphi_aoa = offsets[:, :, 0]
    dtheta_zoa = offsets[:, :, 1]

    dphi_aod = offsets[:, :, 2]
    dtheta_zod = offsets[:, :, 3]

    phi_aoa = (
        phi_aoa_hat
        + dphi_aoa[:, :, None]
        + torch.as_tensor(
            c_ASA,
            dtype=rd,
            device=dev,
        ) * alpha[None, None, :]
    )

    theta_zoa = (
        theta_zoa_hat
        + dtheta_zoa[:, :, None]
        + torch.as_tensor(
            c_ZSA,
            dtype=rd,
            device=dev,
        ) * alpha[None, None, :]
    )

    phi_aod = (
        phi_aod_hat
        + dphi_aod[:, :, None]
        + torch.as_tensor(
            c_ASD,
            dtype=rd,
            device=dev,
        ) * alpha[None, None, :]
    )

    theta_zod = (
        theta_zod_hat
        + torch.as_tensor(
            mu_offset_ZOD,
            dtype=rd,
            device=dev,
        )
        + dtheta_zod[:, :, None]
        + torch.as_tensor(
            c_ZSD,
            dtype=rd,
            device=dev,
        ) * alpha[None, None, :]
    )

    phi_aoa = cp.wrap_azimuth_deg(phi_aoa)
    phi_aod = cp.wrap_azimuth_deg(phi_aod)

    theta_zoa = cp.wrap_zenith_deg(theta_zoa)
    theta_zod = cp.wrap_zenith_deg(theta_zod)

    tx_loc = cp.location_terms_from_angles(
        phi_aod,
        theta_zod,
        tx_pos,
        parity=parity,
    )
    # [N,M,L,S]

    rx_loc = cp.location_terms_from_angles(
        phi_aoa,
        theta_zoa,
        rx_pos,
        parity=parity,
    )
    # [N,M,L,U]

    # ======================================================================
    # XPR amplitudes
    #
    # Same ray-level XPR as Case 2.
    # ======================================================================

    xpr = XPR.reshape(N, M, L)

    cross = torch.pow(
        torch.tensor(
            10.0,
            dtype=rd,
            device=dev,
        ),
        -xpr / 20.0,
    )

    # ======================================================================
    # CASE-1 NLOS
    # ======================================================================

    H_NLOS = torch.zeros(
        (N, U, S),
        dtype=cd,
        device=dev,
    )

    total_pairs = U * S

    for p0 in range(0, total_pairs, pair_chunk):

        p1 = min(
            p0 + pair_chunk,
            total_pairs,
        )

        pair_id = torch.arange(
            p0,
            p1,
            dtype=torch.long,
            device=dev,
        )

        # Flattening convention:
        #
        # p = u*S + s
        #
        # so each pair appears exactly once.
        u_idx = torch.div(
            pair_id,
            S,
            rounding_mode="floor",
        )

        s_idx = torch.remainder(
            pair_id,
            S,
        )

        P = int(pair_id.numel())

        # --------------------------------------------------------------
        # Spatial term for this set of antenna pairs.
        #
        # [N,M,L,P]
        # --------------------------------------------------------------

        rx_pair_loc = torch.index_select(
            rx_loc,
            dim=-1,
            index=u_idx,
        )

        tx_pair_loc = torch.index_select(
            tx_loc,
            dim=-1,
            index=s_idx,
        )

        location_pair = (
            rx_pair_loc
            * tx_pair_loc
        )

        # H contribution of the current pair chunk.
        h_pair = torch.zeros(
            (N, P),
            dtype=cd,
            device=dev,
        )

        # --------------------------------------------------------------
        # Four polarization terms.
        #
        # Instead of allocating
        #
        #   Phi[N,M,L,P,2,2]
        #
        # we generate ONE polarization term at a time:
        #
        #   phase[N,M,L,P].
        #
        # This keeps GPU memory substantially lower.
        # --------------------------------------------------------------

        for pol_rx in range(2):

            rx_coeff = torch.index_select(
                rx_field[pol_rx],
                dim=0,
                index=u_idx,
            )

            for pol_tx in range(2):

                tx_coeff = torch.index_select(
                    tx_field[pol_tx],
                    dim=0,
                    index=s_idx,
                )

                pair_field_coeff = (
                    rx_coeff
                    * tx_coeff
                ).to(cd)
                # [P]

                # ------------------------------------------------------
                # THE CASE-1 DIFFERENCE
                #
                # Independent phase for every:
                #
                # realization n
                # cluster m
                # ray l
                # antenna pair p=(u,s)
                # polarization entry
                # ------------------------------------------------------

                phase = (
                    -math.pi
                    + 2.0
                    * math.pi
                    * torch.rand(
                        (N, M, L, P),
                        dtype=rd,
                        device=dev,
                        generator=generator,
                    )
                )

                exp_phase = torch.exp(
                    1j * phase
                ).to(cd)

                # Co-polar amplitude = 1
                # Cross-polar amplitude = 1/sqrt(kappa)
                if pol_rx == pol_tx:

                    ray_term = (
                        exp_phase
                        * pair_field_coeff[
                            None,
                            None,
                            None,
                            :
                        ]
                    )

                else:

                    ray_term = (
                        exp_phase
                        * cross[
                            :,
                            :,
                            :,
                            None,
                        ].to(cd)
                        * pair_field_coeff[
                            None,
                            None,
                            None,
                            :
                        ]
                    )

                # Sum rays/clusters immediately.
                # We do not keep the large ray tensor.
                h_pair += torch.sum(
                    ray_term * location_pair,
                    dim=(1, 2),
                )

        h_pair = (
            h_pair
            / math.sqrt(M * L)
        )

        # [N,P] -> corresponding [N,u,s]
        H_NLOS[:, u_idx, s_idx] = h_pair

    # ======================================================================
    # Rician combination
    # ======================================================================

    Kt = torch.as_tensor(
        K,
        dtype=rd,
        device=dev,
    )

    if is_los:

        H = (
            torch.sqrt(
                1.0 / (Kt + 1.0)
            ).to(cd)
            * H_NLOS
            +
            torch.sqrt(
                Kt / (Kt + 1.0)
            ).to(cd)
            * H_LOS[None, :, :]
        )

    else:

        H = H_NLOS

    return H


@torch.inference_mode()
def generate_case1_native_link_chunk(
    cfg: LinkConfig,
    N: int,
    *,
    generator: torch.Generator,
    pair_chunk: int = 8,
    device=None,
    parity: bool = False,
) -> torch.Tensor:
    """
    GPU-native Case-1 channel realization wrapper.

    Output:
        H[N,U,S]

    with antenna-pair-specific independent NLOS
    random polarization phases.
    """

    p = sample_case1_primitives(
        cfg,
        N,
        generator=generator,
        device=device,
        parity=parity,
    )

    return generate_case1_channel_from_primitives(
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
        generator=generator,
        pair_chunk=pair_chunk,
        device=device,
        parity=parity,
    )
