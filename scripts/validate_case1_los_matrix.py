
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


def make_cfg(
    tx_spec,
    rx_spec,
    *,
    is_los,
    K,
    mu_xpr,
):
    return LinkConfig(
        tx_spec=tx_spec,
        rx_spec=rx_spec,

        a_vector=np.array([-1.0, 0.0, 0.0]),
        d_vector=np.array([ 1.0, 0.0, 0.0]),

        fc=3.5e9,

        K=K if is_los else 0.0,
        is_los=is_los,

        M=4,
        L=20,

        mu_XPR=mu_xpr,
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


@torch.inference_mode()
def run_case(
    nris_side,
    br_los,
    ru_los,
    *,
    N_TOTAL,
    device,
    seed,
):

    gnb = ArraySpec(1, 1)
    ris = ArraySpec(nris_side, nris_side)
    ue  = ArraySpec(1, 1)

    cfg_br = make_cfg(
        gnb,
        ris,
        is_los=br_los,
        K=5.0,
        mu_xpr=8.0,
    )

    cfg_ru = make_cfg(
        ris,
        ue,
        is_los=ru_los,
        K=3.0,
        mu_xpr=7.0,
    )

    nT = 2
    nRIS = 2 * nris_side * nris_side
    nR = 2

    # --------------------------------------------------------
    # W
    # --------------------------------------------------------

    idx = torch.arange(
        nT,
        dtype=torch.float32,
        device=device,
    )

    W = torch.exp(
        1j * 2*math.pi*idx/nT
    ).to(torch.complex64)

    W /= torch.linalg.vector_norm(W)

    # --------------------------------------------------------
    # gamma
    # --------------------------------------------------------

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
        1j*phi
    ).to(torch.complex64)

    # --------------------------------------------------------
    # Analytic
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MC
    # --------------------------------------------------------

    gb = torch.Generator(device=device)
    gr = torch.Generator(device=device)

    gb.manual_seed(seed)
    gr.manual_seed(seed + 100000)

    if nRIS <= 32:
        mc_chunk = 2048
    elif nRIS <= 128:
        mc_chunk = 1024
    else:
        mc_chunk = 256

    sum_y = 0.0
    sum_y2 = 0.0
    count = 0

    for start in range(0, N_TOTAL, mc_chunk):

        n = min(
            mc_chunk,
            N_TOTAL-start,
        )

        HBR = generate_case1_native_link_chunk(
            cfg_br,
            n,
            generator=gb,
            pair_chunk=8,
            device=device,
            parity=False,
        )

        HRU = generate_case1_native_link_chunk(
            cfg_ru,
            n,
            generator=gr,
            pair_chunk=8,
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

    mean_mc = sum_y / count

    var_mc = (
        sum_y2/count
        - mean_mc**2
    )

    mean_a = float(meanY_a.cpu())
    wick = float(varY_wick.cpu())

    return {
        "nRIS": nRIS,
        "BR": "LOS" if br_los else "NLOS",
        "RU": "LOS" if ru_los else "NLOS",
        "meanAPE": (
            100*abs(mean_mc-mean_a)
            / max(abs(mean_mc),1e-20)
        ),
        "ratio": var_mc / wick,
        "wickAPE": (
            100*abs(var_mc-wick)
            / max(abs(var_mc),1e-20)
        ),
    }


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:",device)

    # İlk matrix testinde 8,32,128 yeterli.
    # 512'yi sonuçlara göre sonra ekleriz.
    sides = [2,4,8]

    modes = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    N_TOTAL = 32768

    results = []

    for side in sides:

        nRIS = 2*side*side

        for br_los,ru_los in modes:

            label = (
                f"{'LOS' if br_los else 'NLOS'} / "
                f"{'LOS' if ru_los else 'NLOS'}"
            )

            print(
                f"\nTesting nRIS={nRIS:3d} | {label}"
            )

            r = run_case(
                side,
                br_los,
                ru_los,
                N_TOTAL=N_TOTAL,
                device=device,
                seed=12345 + nRIS,
            )

            results.append(r)

            print(
                f"meanAPE={r['meanAPE']:.3f}% | "
                f"MC/Wick={r['ratio']:.5f} | "
                f"WickAPE={r['wickAPE']:.3f}%"
            )

    print("\n")
    print("="*86)
    print("CASE 1 LOS/NLOS MATRIX")
    print("="*86)

    print(
        f"{'nRIS':>6} "
        f"{'BR':>6} "
        f"{'RU':>6} "
        f"{'MeanAPE%':>11} "
        f"{'MC/Wick':>11} "
        f"{'WickAPE%':>11}"
    )

    print("-"*86)

    for r in results:

        print(
            f"{r['nRIS']:6d} "
            f"{r['BR']:>6} "
            f"{r['RU']:>6} "
            f"{r['meanAPE']:11.3f} "
            f"{r['ratio']:11.5f} "
            f"{r['wickAPE']:11.3f}"
        )


if __name__ == "__main__":
    main()
