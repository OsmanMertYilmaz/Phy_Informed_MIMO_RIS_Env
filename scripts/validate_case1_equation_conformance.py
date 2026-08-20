
from __future__ import annotations

import math
import numpy as np
import torch

from ris_env.antenna import ArraySpec
from ris_env import channel_primitives as cp
from ris_env.case1.channel_realizations import (
    generate_case1_channel_from_primitives,
)


# =====================================================================
# Materialize the conceptual Case-1 phase tensor
#
# Phi[n,m,l,u,s,a,b]
#
# IMPORTANT:
# RNG draw order deliberately matches the optimized Case-1 generator.
# =====================================================================

@torch.inference_mode()
def materialize_case1_phi(
    *,
    N,
    M,
    L,
    U,
    S,
    pair_chunk,
    generator,
    device,
    parity,
):
    dev = torch.device(device)
    rd = cp._rdtype(parity)

    Phi = torch.empty(
        (N, M, L, U, S, 2, 2),
        dtype=rd,
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

        # Same flattening convention as optimized code:
        #
        # p = u*S + s
        u_idx = torch.div(
            pair_id,
            S,
            rounding_mode="floor",
        )

        s_idx = torch.remainder(
            pair_id,
            S,
        )

        P = pair_id.numel()

        # Same RNG loop order as optimized generator.
        for pol_rx in range(2):

            for pol_tx in range(2):

                phase = (
                    -math.pi
                    + 2.0 * math.pi
                    * torch.rand(
                        (N, M, L, P),
                        dtype=rd,
                        device=dev,
                        generator=generator,
                    )
                )

                # Fill the conceptual full tensor.
                for j in range(P):

                    u = int(u_idx[j])
                    s = int(s_idx[j])

                    Phi[
                        :, :, :, u, s,
                        pol_rx, pol_tx
                    ] = phase[:, :, :, j]

    return Phi


# =====================================================================
# Literal brute-force implementation of the mathematical Case-1 model
# =====================================================================

@torch.inference_mode()
def brute_force_case1_reference(
    *,
    tx_spec,
    rx_spec,
    a_vector,
    d_vector,
    K,
    is_los,
    M,
    L,
    c_ASA,
    c_ZSA,
    c_ASD,
    c_ZSD,
    mu_offset_ZOD,
    XPR,
    ASAv,
    ZSAv,
    ASDv,
    ZSDv,
    cluster_offsets,
    Phi,
    device,
    parity,
):

    dev = torch.device(device)

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

    offsets = torch.as_tensor(
        cluster_offsets,
        dtype=rd,
        device=dev,
    )

    N = ASAv.numel()

    S = 2 * tx_spec.M * tx_spec.N
    U = 2 * rx_spec.M * rx_spec.N

    assert Phi.shape == (
        N, M, L, U, S, 2, 2
    )

    # ===============================================================
    # Array geometry and dual-polarization fields
    # ===============================================================

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

    # ===============================================================
    # LOS component
    # ===============================================================

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

    # Literal LOS polarization summation.
    H_LOS = torch.zeros(
        (U, S),
        dtype=cd,
        device=dev,
    )

    for u in range(U):

        for s in range(S):

            pol_sum = torch.zeros(
                (),
                dtype=cd,
                device=dev,
            )

            for pol_rx in range(2):

                for pol_tx in range(2):

                    pol_sum += (
                        rx_field[pol_rx, u]
                        * pol_los[
                            pol_rx,
                            pol_tx
                        ]
                        * tx_field[
                            pol_tx,
                            s
                        ]
                    ).to(cd)

            H_LOS[u, s] = (
                rx_loc0[u]
                * tx_loc0[s]
                * pol_sum
            )

    # ===============================================================
    # NLOS ray angles
    # ===============================================================

    alpha = torch.tensor(
        cp.ALPHA,
        dtype=rd,
        device=dev,
    )

    assert alpha.numel() == L

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

    phi_aoa = cp.wrap_azimuth_deg(
        phi_aoa
    )

    phi_aod = cp.wrap_azimuth_deg(
        phi_aod
    )

    theta_zoa = cp.wrap_zenith_deg(
        theta_zoa
    )

    theta_zod = cp.wrap_zenith_deg(
        theta_zod
    )

    rx_loc = cp.location_terms_from_angles(
        phi_aoa,
        theta_zoa,
        rx_pos,
        parity=parity,
    )
    # [N,M,L,U]

    tx_loc = cp.location_terms_from_angles(
        phi_aod,
        theta_zod,
        tx_pos,
        parity=parity,
    )
    # [N,M,L,S]

    # ===============================================================
    # XPR
    #
    # XPR is in dB.
    #
    # Cross-polar complex AMPLITUDE:
    #
    #     1/sqrt(kappa)
    #       = 10^(-XPR_dB/20)
    #
    # ===============================================================

    xpr = XPR.reshape(
        N,
        M,
        L,
    )

    cross = torch.pow(
        torch.tensor(
            10.0,
            dtype=rd,
            device=dev,
        ),
        -xpr / 20.0,
    )

    # ===============================================================
    # Literal Case-1 NLOS equation
    #
    #
    # H[n,u,s] =
    #
    #   1/sqrt(M L)
    #   sum_m sum_l
    #   L_rx[n,m,l,u] L_tx[n,m,l,s]
    #
    #   * sum_a sum_b
    #       F_rx[a,u]
    #       A[n,m,l,a,b]
    #       exp(j Phi[n,m,l,u,s,a,b])
    #       F_tx[b,s]
    #
    # ===============================================================

    H_NLOS = torch.zeros(
        (N, U, S),
        dtype=cd,
        device=dev,
    )

    for n in range(N):

        for u in range(U):

            for s in range(S):

                total = torch.zeros(
                    (),
                    dtype=cd,
                    device=dev,
                )

                for m in range(M):

                    for l in range(L):

                        pol_sum = torch.zeros(
                            (),
                            dtype=cd,
                            device=dev,
                        )

                        for pol_rx in range(2):

                            for pol_tx in range(2):

                                if pol_rx == pol_tx:
                                    amp = torch.ones(
                                        (),
                                        dtype=rd,
                                        device=dev,
                                    )
                                else:
                                    amp = cross[
                                        n, m, l
                                    ]

                                pol_sum += (
                                    rx_field[
                                        pol_rx,
                                        u
                                    ]
                                    * amp
                                    * torch.exp(
                                        1j
                                        * Phi[
                                            n, m, l,
                                            u, s,
                                            pol_rx,
                                            pol_tx
                                        ]
                                    )
                                    * tx_field[
                                        pol_tx,
                                        s
                                    ]
                                ).to(cd)

                        total += (
                            rx_loc[n,m,l,u]
                            * tx_loc[n,m,l,s]
                            * pol_sum
                        )

                H_NLOS[n,u,s] = (
                    total
                    / math.sqrt(M * L)
                )

    # ===============================================================
    # Rician combination
    # ===============================================================

    if is_los:

        Kt = torch.as_tensor(
            K,
            dtype=rd,
            device=dev,
        )

        H = (
            torch.sqrt(
                1.0 / (Kt + 1.0)
            ).to(cd)
            * H_NLOS
            +
            torch.sqrt(
                Kt / (Kt + 1.0)
            ).to(cd)
            * H_LOS[None,:,:]
        )

    else:

        H = H_NLOS.clone()

    return {
        "H": H,
        "H_NLOS": H_NLOS,
        "H_LOS": H_LOS,
        "cross": cross,
    }


# =====================================================================
# One exact test
# =====================================================================

@torch.inference_mode()
def run_one(pair_chunk):

    # CPU + float64 / complex128:
    # tiny test, but maximal numerical accuracy.
    device = torch.device("cpu")
    parity = True

    rd = cp._rdtype(parity)

    # ---------------------------------------------------------------
    # Small dual-pol channel:
    #
    # Tx ArraySpec(1,1) -> S=2 ports
    # Rx ArraySpec(1,1) -> U=2 ports
    #
    # Therefore 4 distinct antenna pairs.
    # ---------------------------------------------------------------

    tx_spec = ArraySpec(1,1)
    rx_spec = ArraySpec(1,1)

    S = 2
    U = 2

    N = 2
    M = 2
    L = 20

    # Nontrivial geometry vectors.
    a_vector = np.array(
        [-0.8, 0.4, 0.3],
        dtype=np.float64,
    )
    a_vector /= np.linalg.norm(a_vector)

    d_vector = np.array(
        [0.7, 0.5, 0.4],
        dtype=np.float64,
    )
    d_vector /= np.linalg.norm(d_vector)

    K = 4.25

    # ---------------------------------------------------------------
    # Deterministic primitives.
    #
    # This deliberately avoids another random sampling layer:
    # only the pair-specific polarization phases are stochastic here.
    # ---------------------------------------------------------------

    XPR = torch.linspace(
        3.0,
        12.0,
        N*M*L,
        dtype=rd,
        device=device,
    ).reshape(N, M*L)

    ASAv = torch.tensor(
        [8.0, 12.0],
        dtype=rd,
        device=device,
    )

    ZSAv = torch.tensor(
        [4.0, 7.0],
        dtype=rd,
        device=device,
    )

    ASDv = torch.tensor(
        [9.0, 14.0],
        dtype=rd,
        device=device,
    )

    ZSDv = torch.tensor(
        [5.0, 8.0],
        dtype=rd,
        device=device,
    )

    offsets = torch.tensor(
        [
            [
                [ 1.50, -0.80,  2.20,  0.70],
                [-2.00,  1.10, -1.30, -0.60],
            ],
            [
                [ 0.40,  1.70, -2.50,  1.20],
                [ 2.10, -1.40,  0.80, -1.10],
            ],
        ],
        dtype=rd,
        device=device,
    )

    common = dict(
        tx_spec=tx_spec,
        rx_spec=rx_spec,

        a_vector=a_vector,
        d_vector=d_vector,

        fc=3.5e9,

        K=K,

        M=M,
        L=L,

        c_ASA=1.17,
        c_ZSA=0.83,
        c_ASD=1.31,
        c_ZSD=0.91,

        mu_offset_ZOD=2.35,

        XPR=XPR,

        ASAv=ASAv,
        ZSAv=ZSAv,
        ASDv=ASDv,
        ZSDv=ZSDv,

        cluster_offsets=offsets,

        pair_chunk=pair_chunk,

        device=device,
        parity=parity,
    )

    # ===============================================================
    # PHASE SEED
    # ===============================================================

    PHASE_SEED = 20260820

    # ---------------------------------------------------------------
    # FAST NLOS
    # ---------------------------------------------------------------

    g_fast_nlos = torch.Generator(
        device=device
    )

    g_fast_nlos.manual_seed(
        PHASE_SEED
    )

    H_fast_nlos = (
        generate_case1_channel_from_primitives(
            **common,
            is_los=False,
            generator=g_fast_nlos,
        )
    )

    # ---------------------------------------------------------------
    # Materialize exactly the same conceptual Phi tensor.
    # ---------------------------------------------------------------

    g_ref = torch.Generator(
        device=device
    )

    g_ref.manual_seed(
        PHASE_SEED
    )

    Phi = materialize_case1_phi(
        N=N,
        M=M,
        L=L,
        U=U,
        S=S,
        pair_chunk=pair_chunk,
        generator=g_ref,
        device=device,
        parity=parity,
    )

    # ---------------------------------------------------------------
    # Literal reference NLOS
    # ---------------------------------------------------------------

    ref_nlos = brute_force_case1_reference(
        **{
            k: v
            for k, v in common.items()
            if k not in {
                "pair_chunk",
                "fc",
                "K",
            }
        },

        K=K,
        is_los=False,
        Phi=Phi,
    )

    # ---------------------------------------------------------------
    # FAST RICIAN
    #
    # Reset seed -> exactly same NLOS phase realization.
    # ---------------------------------------------------------------

    g_fast_los = torch.Generator(
        device=device
    )

    g_fast_los.manual_seed(
        PHASE_SEED
    )

    H_fast_rician = (
        generate_case1_channel_from_primitives(
            **common,
            is_los=True,
            generator=g_fast_los,
        )
    )

    # ---------------------------------------------------------------
    # Literal reference Rician
    # ---------------------------------------------------------------

    ref_los = brute_force_case1_reference(
        **{
            k: v
            for k, v in common.items()
            if k not in {
                "pair_chunk",
                "fc",
                "K",
            }
        },

        K=K,
        is_los=True,
        Phi=Phi,
    )

    H_ref_nlos = ref_nlos["H_NLOS"]
    H_ref_los = ref_los["H_LOS"]
    H_ref_rician = ref_los["H"]

    # ===============================================================
    # ERRORS
    # ===============================================================

    max_nlos = float(
        torch.max(
            torch.abs(
                H_fast_nlos
                - H_ref_nlos
            )
        )
    )

    max_rician = float(
        torch.max(
            torch.abs(
                H_fast_rician
                - H_ref_rician
            )
        )
    )

    rel_nlos = float(
        torch.linalg.vector_norm(
            H_fast_nlos
            - H_ref_nlos
        )
        /
        torch.linalg.vector_norm(
            H_ref_nlos
        )
    )

    rel_rician = float(
        torch.linalg.vector_norm(
            H_fast_rician
            - H_ref_rician
        )
        /
        torch.linalg.vector_norm(
            H_ref_rician
        )
    )

    # ---------------------------------------------------------------
    # Independently verify Rician composition.
    # ---------------------------------------------------------------

    Kt = torch.tensor(
        K,
        dtype=rd,
        device=device,
    )

    H_manual_rician = (
        torch.sqrt(
            1.0/(Kt+1.0)
        ).to(H_fast_nlos.dtype)
        * H_fast_nlos
        +
        torch.sqrt(
            Kt/(Kt+1.0)
        ).to(H_fast_nlos.dtype)
        * H_ref_los[None,:,:]
    )

    max_rician_composition = float(
        torch.max(
            torch.abs(
                H_fast_rician
                - H_manual_rician
            )
        )
    )

    # ===============================================================
    # Direct pair-phase structural checks
    # ===============================================================

    pair_blocks = []

    for u in range(U):
        for s in range(S):
            pair_blocks.append(
                Phi[:,:,:,u,s,:,:]
            )

    identical_pairs = 0
    total_comparisons = 0

    for i in range(len(pair_blocks)):
        for j in range(i+1, len(pair_blocks)):

            total_comparisons += 1

            if torch.equal(
                pair_blocks[i],
                pair_blocks[j],
            ):
                identical_pairs += 1

    phase_min = float(Phi.min())
    phase_max = float(Phi.max())

    # ---------------------------------------------------------------
    # XPR amplitude sanity check.
    # ---------------------------------------------------------------

    first_xpr_db = float(
        XPR.reshape(N,M,L)[0,0,0]
    )

    first_cross = float(
        ref_los["cross"][0,0,0]
    )

    expected_cross = (
        10.0 ** (
            -first_xpr_db / 20.0
        )
    )

    xpr_amp_error = abs(
        first_cross
        - expected_cross
    )

    return {
        "pair_chunk": pair_chunk,

        "max_nlos": max_nlos,
        "rel_nlos": rel_nlos,

        "max_rician": max_rician,
        "rel_rician": rel_rician,

        "max_rician_composition":
            max_rician_composition,

        "identical_pairs":
            identical_pairs,

        "pair_comparisons":
            total_comparisons,

        "phase_min": phase_min,
        "phase_max": phase_max,

        "xpr_db": first_xpr_db,
        "cross": first_cross,
        "cross_expected":
            expected_cross,

        "xpr_amp_error":
            xpr_amp_error,
    }


# =====================================================================
# Main
# =====================================================================

def main():

    print("="*100)
    print("CASE 1 — EQUATION-LEVEL CONFORMANCE")
    print("="*100)

    # Test several chunk sizes.
    #
    # NOTE:
    # Same seed does NOT need to produce the same H across
    # different pair_chunk values because RNG draw ordering changes.
    #
    # Every pair_chunk is therefore compared to its OWN exact
    # equation-level Phi realization.
    pair_chunks = [
        1,
        2,
        4,
    ]

    results = []

    for pc in pair_chunks:

        print(
            f"\nTesting pair_chunk={pc}"
        )

        r = run_one(pc)
        results.append(r)

        print(
            f"max |H_NLOS fast-ref|    = "
            f"{r['max_nlos']:.3e}"
        )

        print(
            f"relative NLOS error       = "
            f"{r['rel_nlos']:.3e}"
        )

        print(
            f"max |H_Rician fast-ref|  = "
            f"{r['max_rician']:.3e}"
        )

        print(
            f"relative Rician error     = "
            f"{r['rel_rician']:.3e}"
        )

        print(
            f"Rician composition error  = "
            f"{r['max_rician_composition']:.3e}"
        )

        print(
            f"identical antenna-pair "
            f"phase blocks = "
            f"{r['identical_pairs']}/"
            f"{r['pair_comparisons']}"
        )

        print(
            f"phase range = "
            f"[{r['phase_min']:.6f}, "
            f"{r['phase_max']:.6f}]"
        )

        print(
            f"XPR={r['xpr_db']:.3f} dB "
            f"-> cross amplitude "
            f"{r['cross']:.12f} "
            f"(expected "
            f"{r['cross_expected']:.12f})"
        )

    # ===============================================================
    # Assertions
    # ===============================================================

    tol = 1e-12

    for r in results:

        assert r["max_nlos"] < tol, (
            "NLOS equation mismatch"
        )

        assert r["max_rician"] < tol, (
            "Rician equation mismatch"
        )

        assert (
            r["max_rician_composition"]
            < tol
        ), "Rician composition mismatch"

        assert (
            r["identical_pairs"] == 0
        ), "Different antenna pairs share identical Phi blocks"

        assert (
            r["phase_min"] >= -math.pi
            and
            r["phase_max"] <= math.pi
        ), "Phase outside [-pi,pi]"

        assert (
            r["xpr_amp_error"] < 1e-14
        ), "XPR amplitude conversion mismatch"

    print("\n" + "="*100)
    print("PASS")
    print("="*100)

    print(
        "Optimized Case-1 generator is numerically "
        "equivalent to the literal mathematical equation."
    )

    print(
        "Pair-specific polarization phases, XPR scaling, "
        "1/sqrt(M*L) normalization, and Rician composition "
        "all passed."
    )


if __name__ == "__main__":
    main()
