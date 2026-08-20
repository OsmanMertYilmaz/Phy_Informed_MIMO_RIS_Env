
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/content/Phy_Informed_MIMO_RIS_Env_case1")
sys.path.insert(0, str(REPO / "scripts"))

from ris_env.antenna import ArraySpec
from ris_env.channel_realizations import LinkConfig
from ris_env.case1 import generate_case1_native_link_chunk

from validate_case1_cascade_moments import (
    case1_single_hop_moments,
    case1_feff_moments,
)


@torch.inference_mode()
def run_one(nris_side, N_TOTAL, device, seed_offset=0):

    gnb = ArraySpec(1, 1)
    ris = ArraySpec(nris_side, nris_side)
    ue  = ArraySpec(1, 1)

    cfg_br = LinkConfig(
        tx_spec=gnb,
        rx_spec=ris,

        a_vector=np.array([-1.0, 0.0, 0.0]),
        d_vector=np.array([ 1.0, 0.0, 0.0]),

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

    cfg_ru = LinkConfig(
        tx_spec=ris,
        rx_spec=ue,

        a_vector=np.array([-1.0, 0.0, 0.0]),
        d_vector=np.array([ 1.0, 0.0, 0.0]),

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

    nT = 2 * gnb.M * gnb.N
    nRIS = 2 * ris.M * ris.N
    nR = 2 * ue.M * ue.N

    # ------------------------------------------------------------
    # Fixed W
    # ------------------------------------------------------------

    idx = torch.arange(
        nT,
        device=device,
        dtype=torch.float32,
    )

    W = torch.exp(
        1j * 2.0 * math.pi * idx / nT
    ).to(torch.complex64)

    W /= torch.linalg.vector_norm(W)

    # ------------------------------------------------------------
    # Fixed alternating RIS pattern
    # ------------------------------------------------------------

    z = torch.arange(
        nRIS,
        device=device,
    ) % 2

    phi = torch.where(
        z == 0,
        torch.tensor(math.pi/4, device=device),
        torch.tensor(3*math.pi/4, device=device),
    )

    gamma = torch.exp(
        1j * phi
    ).to(torch.complex64)

    # ------------------------------------------------------------
    # Analytic
    # ------------------------------------------------------------

    mu_h, var_h = case1_single_hop_moments(
        cfg_br,
        device=device,
        parity=False,
    )

    mu_g, var_g = case1_single_hop_moments(
        cfg_ru,
        device=device,
        parity=False,
    )

    mu_f, Sigma = case1_feff_moments(
        mu_h,
        var_h,
        mu_g,
        var_g,
        W,
        gamma,
    )

    meanY_a = (
        torch.trace(Sigma).real
        + torch.sum(torch.abs(mu_f)**2)
    )

    varY_wick = (
        torch.trace(Sigma @ Sigma).real
        +
        2.0 * torch.real(
            torch.vdot(
                mu_f,
                Sigma @ mu_f,
            )
        )
    )

    # ------------------------------------------------------------
    # MC
    # ------------------------------------------------------------

    gen_br = torch.Generator(device=device)
    gen_ru = torch.Generator(device=device)

    gen_br.manual_seed(112233 + seed_offset)
    gen_ru.manual_seed(998877 + seed_offset)

    # Reduce chunk for large RIS.
    if nRIS <= 32:
        MC_CHUNK = 2048
    elif nRIS <= 128:
        MC_CHUNK = 1024
    else:
        MC_CHUNK = 256

    PAIR_CHUNK = 8

    count = 0
    sum_y = 0.0
    sum_y2 = 0.0

    for start in range(0, N_TOTAL, MC_CHUNK):

        n = min(
            MC_CHUNK,
            N_TOTAL-start,
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

        q = torch.einsum(
            "nit,t->ni",
            HBR,
            W,
        )

        Feff = torch.einsum(
            "nri,i,ni->nr",
            HRU,
            gamma,
            q,
        )

        Y = torch.sum(
            torch.abs(Feff)**2,
            dim=1,
        ).double()

        sum_y += float(Y.sum().cpu())
        sum_y2 += float((Y*Y).sum().cpu())

        count += n

        del HBR, HRU, q, Feff, Y

    meanY_mc = sum_y / count

    varY_mc = (
        sum_y2/count
        - meanY_mc**2
    )

    mean_ape = (
        abs(meanY_mc-float(meanY_a.cpu()))
        / meanY_mc
    )

    wick = float(varY_wick.cpu())

    return {
        "nRIS": nRIS,
        "N": count,

        "meanA": float(meanY_a.cpu()),
        "meanMC": meanY_mc,
        "meanAPE": 100*mean_ape,

        "wick": wick,
        "varMC": varY_mc,
        "ratio": varY_mc/wick,
        "wickAPE": (
            100*abs(wick-varY_mc)/varY_mc
        ),
    }


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # dual-pol port count:
    #
    # side 2 ->  8 RIS ports
    # side 4 -> 32 RIS ports
    # side 8 -> 128 RIS ports
    # side16 -> 512 RIS ports

    sides = [2, 4, 8, 16]

    # İlk test için 32k yeterli.
    N_TOTAL = 32768

    results = []

    for side in sides:

        nris = 2 * side * side

        print("\n" + "="*80)
        print("Testing nRIS =", nris)
        print("="*80)

        r = run_one(
            side,
            N_TOTAL,
            device,
        )

        results.append(r)

        print(
            f"mean APE = {r['meanAPE']:.3f}% | "
            f"MC/Wick = {r['ratio']:.6f} | "
            f"Wick APE = {r['wickAPE']:.3f}%"
        )

    print("\n")
    print("="*92)
    print("CASE 1 WICK SCALING SUMMARY")
    print("="*92)

    print(
        f"{'nRIS':>8} "
        f"{'N_MC':>10} "
        f"{'MeanAPE%':>12} "
        f"{'MC/Wick':>12} "
        f"{'WickAPE%':>12}"
    )

    print("-"*92)

    for r in results:

        print(
            f"{r['nRIS']:8d} "
            f"{r['N']:10d} "
            f"{r['meanAPE']:12.3f} "
            f"{r['ratio']:12.6f} "
            f"{r['wickAPE']:12.3f}"
        )


if __name__ == "__main__":
    main()
