
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ris_env.environment as env

from ris_env.antenna import ArraySpec
from ris_env.channel_realizations import LinkConfig
from ris_env.case1 import generate_case1_native_link_chunk

from ris_env.codebook import (
    generate_codebook_rank1,
    flatten_codebook_matlab_loop_order,
)

from ris_env.ris_response import (
    z_to_phi,
    generate_ris_response_from_phi,
)


# ======================================================================
# Helpers
# ======================================================================

def scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().reshape(-1)[0].item()
    return np.asarray(x).reshape(-1)[0].item()


def los_label(s):
    s = str(s).upper()
    return "NLOS" if "NLOS" in s else "LOS"


def total_ris_ports(row):
    if "nRIS" in row.index:
        return int(row["nRIS"])
    return 2 * int(row["nRIS_x"]) * int(row["nRIS_y"])


def find_environment_file():
    root = Path(
        "/content/drive/MyDrive/"
        "Phy_Informed_MIMO_RIS_Env/environments"
    )

    if not root.exists():
        raise FileNotFoundError(root)

    candidates = (
        list(root.rglob("*.parquet"))
        + list(root.rglob("*.csv"))
    )

    # Önce adı environment/4000 içerenleri dene.
    candidates = sorted(
        candidates,
        key=lambda p: (
            "4000" in p.name.lower(),
            "environment" in p.name.lower(),
            p.stat().st_size,
        ),
        reverse=True,
    )

    required = {
        "scenario_BR",
        "scenario_RU",
        "fc",
        "ris_x", "ris_y", "ris_z",
        "gnb_x", "gnb_y", "gnb_z",
        "ue_x", "ue_y", "ue_z",
        "nT1", "nT2",
        "nR1", "nR2",
        "nRIS_x", "nRIS_y",
    }

    for p in candidates:
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)

            if required.issubset(df.columns) and len(df) >= 100:
                return p, df

        except Exception:
            pass

    raise RuntimeError(
        "Uygun environment dataframe bulunamadı."
    )


def make_link_cfg(
    row,
    geom,
    lsp,
    *,
    hop,
):
    if hop == "BR":
        tx_spec = ArraySpec(
            int(row.nT1),
            int(row.nT2),
        )
        rx_spec = ArraySpec(
            int(row.nRIS_x),
            int(row.nRIS_y),
        )

        a_vector = (
            geom.ris2gnb.detach()
            .cpu().numpy()
            .reshape(3)
        )

        d_vector = (
            geom.gnb2ris.detach()
            .cpu().numpy()
            .reshape(3)
        )

    elif hop == "RU":
        tx_spec = ArraySpec(
            int(row.nRIS_x),
            int(row.nRIS_y),
        )
        rx_spec = ArraySpec(
            int(row.nR1),
            int(row.nR2),
        )

        a_vector = (
            geom.ue2ris.detach()
            .cpu().numpy()
            .reshape(3)
        )

        d_vector = (
            geom.ris2ue.detach()
            .cpu().numpy()
            .reshape(3)
        )

    else:
        raise ValueError(hop)

    return LinkConfig(
        tx_spec=tx_spec,
        rx_spec=rx_spec,

        a_vector=a_vector,
        d_vector=d_vector,

        fc=float(row.fc),

        K=float(scalar(lsp.K_linear)),
        is_los=bool(scalar(lsp.isLOS)),

        M=int(scalar(lsp.M)),
        L=int(scalar(lsp.L)),

        mu_XPR=float(scalar(lsp.mu_XPR)),
        sigma_XPR=float(scalar(lsp.sigma_XPR)),

        mu_ASA=float(scalar(lsp.mu_ASA)),
        sigma_ASA=float(scalar(lsp.sigma_ASA)),

        mu_ZSA=float(scalar(lsp.mu_ZSA)),
        sigma_ZSA=float(scalar(lsp.sigma_ZSA)),

        mu_ASD=float(scalar(lsp.mu_ASD)),
        sigma_ASD=float(scalar(lsp.sigma_ASD)),

        mu_ZSD=float(scalar(lsp.mu_ZSD)),
        sigma_ZSD=float(scalar(lsp.sigma_ZSD)),

        c_ASA=float(scalar(lsp.c_ASA)),
        c_ZSA=float(scalar(lsp.c_ZSA)),
        c_ASD=float(scalar(lsp.c_ASD)),
        c_ZSD=float(scalar(lsp.c_ZSD)),

        mu_offset_ZOD=float(
            scalar(lsp.mu_offset_ZOD)
        ),
    )


# ======================================================================
# Case-1 analytic Feff moments
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
    cd = mu_h.dtype
    rd = mu_h.real.dtype
    dev = mu_h.device

    W = W.to(
        device=dev,
        dtype=cd,
    )

    gamma = gamma.to(
        device=dev,
        dtype=cd,
    )

    # q_i = sum_t W_t h_it
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

    # mean Feff
    mu_f = torch.einsum(
        "i,ri,i->r",
        gamma,
        mu_g,
        m,
    )

    abs_gamma2 = (
        torch.abs(gamma)**2
    ).to(rd)

    # BR random contribution
    Sigma = torch.einsum(
        "i,ri,si->rs",
        (abs_gamma2 * v).to(cd),
        mu_g,
        mu_g.conj(),
    )

    # RU random contribution + product term
    second_q = (
        torch.abs(m)**2 + v
    ).real

    diag_extra = torch.einsum(
        "i,ri,i->r",
        abs_gamma2,
        var_g,
        second_q,
    )

    Sigma += torch.diag(
        diag_extra.to(cd)
    )

    Sigma = 0.5 * (
        Sigma + Sigma.conj().T
    )

    return mu_f, Sigma


# ======================================================================
# Real environment physics without Case-2 rho
# ======================================================================

@torch.inference_mode()
def prepare_real_case1_bank(
    row,
    *,
    device,
):
    dev = torch.device(device)

    parity = False

    ris_xyz = (
        float(row.ris_x),
        float(row.ris_y),
        float(row.ris_z),
    )

    gnb_xyz = (
        float(row.gnb_x),
        float(row.gnb_y),
        float(row.gnb_z),
    )

    ue_xyz = (
        float(row.ue_x),
        float(row.ue_y),
        float(row.ue_z),
    )

    # --------------------------------------------------------------
    # Exact production geometry
    # --------------------------------------------------------------

    geom = env.generate_geometry_batch(
        ris_xyz,
        gnb_xyz,
        ue_xyz,
        device=dev,
        parity=parity,
    )

    sid_br = torch.tensor(
        env.SCENARIO_TO_ID[str(row.scenario_BR)],
        dtype=torch.long,
        device=dev,
    )

    sid_ru = torch.tensor(
        env.SCENARIO_TO_ID[str(row.scenario_RU)],
        dtype=torch.long,
        device=dev,
    )

    # --------------------------------------------------------------
    # Exact production LSP
    # --------------------------------------------------------------

    lsp_br = env.generate_lsp_batch(
        sid_br,
        float(row.fc),
        geom.ris2gnb,
        geom.gnb2ris,
        device=dev,
        parity=parity,
    )

    lsp_ru = env.generate_lsp_batch(
        sid_ru,
        float(row.fc),
        geom.ue2ris,
        geom.ris2ue,
        device=dev,
        parity=parity,
    )

    # --------------------------------------------------------------
    # Exact production marginal channel moments
    # --------------------------------------------------------------

    br = env.generate_channel_moments_batch(
        tx_spec=ArraySpec(
            int(row.nT1),
            int(row.nT2),
        ),

        rx_spec=ArraySpec(
            int(row.nRIS_x),
            int(row.nRIS_y),
        ),

        a_vectors=geom.ris2gnb.reshape(1,3),
        d_vectors=geom.gnb2ris.reshape(1,3),

        carrier_frequency=float(row.fc),

        K=lsp_br.K_linear.reshape(1),

        mu_xpr=lsp_br.mu_XPR.reshape(1),
        sigma_xpr=lsp_br.sigma_XPR.reshape(1),

        c0=env.C0,
        device=dev,
        parity=parity,
    )

    ru = env.generate_channel_moments_batch(
        tx_spec=ArraySpec(
            int(row.nRIS_x),
            int(row.nRIS_y),
        ),

        rx_spec=ArraySpec(
            int(row.nR1),
            int(row.nR2),
        ),

        a_vectors=geom.ue2ris.reshape(1,3),
        d_vectors=geom.ris2ue.reshape(1,3),

        carrier_frequency=float(row.fc),

        K=lsp_ru.K_linear.reshape(1),

        mu_xpr=lsp_ru.mu_XPR.reshape(1),
        sigma_xpr=lsp_ru.sigma_XPR.reshape(1),

        c0=env.C0,
        device=dev,
        parity=parity,
    )

    cfg_br = make_link_cfg(
        row,
        geom,
        lsp_br,
        hop="BR",
    )

    cfg_ru = make_link_cfg(
        row,
        geom,
        lsp_ru,
        hop="RU",
    )

    # --------------------------------------------------------------
    # Real Type-I codebook W
    # --------------------------------------------------------------

    nT = (
        2
        * int(row.nT1)
        * int(row.nT2)
    )

    cb = generate_codebook_rank1(
        2,
        int(row.nT1),
        int(row.nT2),
        1,
        1,
        device=dev,
        parity=False,
    )

    Wpool, Widx = (
        flatten_codebook_matlab_loop_order(cb)
    )

    # Support either [K,nT] or [nT,K].
    if Wpool.ndim != 2:
        raise RuntimeError(
            f"Unexpected Wpool shape {Wpool.shape}"
        )

    bank_id = int(
        row.bankID
        if "bankID" in row.index
        else row.name
    )

    if Wpool.shape[1] == nT:
        k = bank_id % Wpool.shape[0]
        W = Wpool[k]
    elif Wpool.shape[0] == nT:
        k = bank_id % Wpool.shape[1]
        W = Wpool[:,k]
    else:
        raise RuntimeError(
            f"Cannot identify W dimension: "
            f"Wpool={tuple(Wpool.shape)}, nT={nT}"
        )

    W = W.to(
        device=dev,
        dtype=torch.complex64,
    )

    W = W / torch.linalg.vector_norm(W)

    # --------------------------------------------------------------
    # Real RIS amplitude/phase model
    #
    # z generated deterministically from bank ris_seed.
    # --------------------------------------------------------------

    nRIS = total_ris_ports(row)

    ris_seed = int(
        row.ris_seed
        if "ris_seed" in row.index
        else bank_id + 777
    )

    rng = np.random.default_rng(
        ris_seed
    )

    z_np = rng.integers(
        0,
        2,
        size=nRIS,
        dtype=np.int64,
    )

    z = torch.as_tensor(
        z_np,
        device=dev,
    )

    phi = z_to_phi(
        z,
        device=dev,
        parity=False,
    )

    beta, gamma = (
        generate_ris_response_from_phi(
            phi,
            device=dev,
            parity=False,
        )
    )

    # --------------------------------------------------------------
    # Case-1 single-hop moments:
    #
    # mean is unchanged relative to production Stage-1.
    # marginal variance is unchanged.
    #
    # Only pair cross-covariances are removed.
    # --------------------------------------------------------------

    mu_h = br["muH"][0].to(
        torch.complex64
    )

    mu_g = ru["muH"][0].to(
        torch.complex64
    )

    sigma2_h = float(
        scalar(br["sigma2H"][0])
    )

    sigma2_g = float(
        scalar(ru["sigma2H"][0])
    )

    var_h = torch.full(
        mu_h.shape,
        sigma2_h,
        dtype=torch.float32,
        device=dev,
    )

    var_g = torch.full(
        mu_g.shape,
        sigma2_g,
        dtype=torch.float32,
        device=dev,
    )

    return {
        "cfg_br": cfg_br,
        "cfg_ru": cfg_ru,
        "mu_h": mu_h,
        "var_h": var_h,
        "mu_g": mu_g,
        "var_g": var_g,
        "W": W,
        "gamma": gamma.to(torch.complex64),
        "Wpick": k,
        "lsp_br": lsp_br,
        "lsp_ru": lsp_ru,
    }


# ======================================================================
# One-bank test
# ======================================================================

@torch.inference_mode()
def run_bank(
    row,
    *,
    N_MC,
    device,
):
    x = prepare_real_case1_bank(
        row,
        device=device,
    )

    mu_f, Sigma = (
        case1_feff_moments(
            x["mu_h"],
            x["var_h"],
            x["mu_g"],
            x["var_g"],
            x["W"],
            x["gamma"],
        )
    )

    meanY_a = (
        torch.trace(Sigma).real
        + torch.sum(
            torch.abs(mu_f)**2
        )
    )

    varY_wick = (
        torch.trace(
            Sigma @ Sigma
        ).real
        +
        2.0 * torch.real(
            torch.vdot(
                mu_f,
                Sigma @ mu_f,
            )
        )
    )

    nR = mu_f.numel()
    nRIS = x["gamma"].numel()

    bank_id = int(
        row.bankID
        if "bankID" in row.index
        else row.name
    )

    gb = torch.Generator(
        device=device
    )
    gr = torch.Generator(
        device=device
    )

    base_seed = int(
        row.channel_seed
        if "channel_seed" in row.index
        else 100000 + bank_id
    )

    gb.manual_seed(
        base_seed + 12345
    )

    gr.manual_seed(
        base_seed + 98765
    )

    # Conservative chunks.
    if nRIS <= 128:
        mc_chunk = 1024
    elif nRIS <= 256:
        mc_chunk = 512
    else:
        mc_chunk = 256

    sum_f = torch.zeros(
        nR,
        dtype=torch.complex128,
        device=device,
    )

    sum_ff = torch.zeros(
        (nR,nR),
        dtype=torch.complex128,
        device=device,
    )

    sum_y = 0.0
    sum_y2 = 0.0
    count = 0

    for start in range(
        0,
        N_MC,
        mc_chunk,
    ):
        n = min(
            mc_chunk,
            N_MC-start,
        )

        HBR = (
            generate_case1_native_link_chunk(
                x["cfg_br"],
                n,
                generator=gb,
                pair_chunk=8,
                device=device,
                parity=False,
            )
        )

        HRU = (
            generate_case1_native_link_chunk(
                x["cfg_ru"],
                n,
                generator=gr,
                pair_chunk=8,
                device=device,
                parity=False,
            )
        )

        q = torch.einsum(
            "nit,t->ni",
            HBR,
            x["W"],
        )

        Feff = torch.einsum(
            "nri,i,ni->nr",
            HRU,
            x["gamma"],
            q,
        )

        Y = torch.sum(
            torch.abs(Feff)**2,
            dim=1,
        ).double()

        Fd = Feff.to(
            torch.complex128
        )

        sum_f += Fd.sum(dim=0)

        sum_ff += torch.einsum(
            "nr,ns->rs",
            Fd,
            Fd.conj(),
        )

        sum_y += float(
            Y.sum().cpu()
        )

        sum_y2 += float(
            (Y*Y).sum().cpu()
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

    # ----------------------------------------------------------
    # MC statistics
    # ----------------------------------------------------------

    mu_mc = sum_f / count

    Eff = sum_ff / count

    Sigma_mc = (
        Eff
        - mu_mc[:,None]
        * mu_mc.conj()[None,:]
    )

    Sigma_mc = 0.5 * (
        Sigma_mc
        + Sigma_mc.conj().T
    )

    mean_mc = sum_y / count

    var_mc = (
        sum_y2/count
        - mean_mc**2
    )

    mean_a = float(
        meanY_a.cpu()
    )

    wick = float(
        varY_wick.cpu()
    )

    # Mean vector error normalized by stochastic scale.
    sig_scale = torch.sqrt(
        torch.trace(
            Sigma.to(
                torch.complex128
            )
        ).real
    ).clamp_min(1e-20)

    mean_vec_err = float(
        (
            torch.linalg.vector_norm(
                mu_mc
                - mu_f.to(
                    torch.complex128
                )
            )
            / sig_scale
        ).cpu()
    )

    cov_err = float(
        (
            torch.linalg.matrix_norm(
                Sigma_mc
                - Sigma.to(
                    torch.complex128
                )
            )
            /
            torch.linalg.matrix_norm(
                Sigma.to(
                    torch.complex128
                )
            ).clamp_min(1e-20)
        ).cpu()
    )

    return {
        "bankID": bank_id,

        "scenario_BR": str(
            row.scenario_BR
        ),
        "scenario_RU": str(
            row.scenario_RU
        ),

        "BR_state": (
            "LOS"
            if x["cfg_br"].is_los
            else "NLOS"
        ),

        "RU_state": (
            "LOS"
            if x["cfg_ru"].is_los
            else "NLOS"
        ),

        "nT": int(
            x["W"].numel()
        ),

        "nR": int(nR),
        "nRIS": int(nRIS),

        "K_BR": float(
            x["cfg_br"].K
        ),

        "K_RU": float(
            x["cfg_ru"].K
        ),

        "N_MC": int(count),

        "meanVecErr_pct": (
            100*mean_vec_err
        ),

        "covFroErr_pct": (
            100*cov_err
        ),

        "meanAnalytic": mean_a,
        "meanMC": mean_mc,

        "meanAPE_pct": (
            100
            * abs(mean_mc-mean_a)
            / max(abs(mean_mc),1e-20)
        ),

        "wickVar": wick,
        "varMC": var_mc,

        "MC_over_Wick": (
            var_mc / wick
        ),

        "wickAPE_pct": (
            100
            * abs(var_mc-wick)
            / max(abs(var_mc),1e-20)
        ),
    }


# ======================================================================
# Bank selection
# ======================================================================

def select_smoke_banks(df):
    df = df.copy()

    df["_nRIS"] = df.apply(
        total_ris_ports,
        axis=1,
    )

    df["_BRstate"] = (
        df["scenario_BR"]
        .map(los_label)
    )

    df["_RUstate"] = (
        df["scenario_RU"]
        .map(los_label)
    )

    available = sorted(
        df["_nRIS"].unique()
    )

    if len(available) <= 3:
        sizes = available
    else:
        sizes = [
            available[0],
            available[len(available)//2],
            available[-1],
        ]

    selected = []

    # Prioritize easiest + hardest state.
    for nris in sizes:

        for brs, rus in [
            ("LOS","LOS"),
            ("NLOS","NLOS"),
        ]:
            q = df[
                (df["_nRIS"] == nris)
                & (df["_BRstate"] == brs)
                & (df["_RUstate"] == rus)
            ]

            if len(q):
                selected.append(
                    q.iloc[0]
                )

    # Fallback if some combinations missing.
    if len(selected) < 6:
        used = {
            int(
                x.bankID
                if "bankID" in x.index
                else x.name
            )
            for x in selected
        }

        for _,row in df.iterrows():
            bid = int(
                row.bankID
                if "bankID" in row.index
                else row.name
            )

            if bid not in used:
                selected.append(row)
                used.add(bid)

            if len(selected) >= 6:
                break

    return selected[:6]


# ======================================================================
# Main
# ======================================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    path, df = (
        find_environment_file()
    )

    print("Environment file:", path)
    print("Rows:", len(df))

    print(
        "Available nRIS:",
        sorted(
            {
                total_ris_ports(row)
                for _,row in df.iterrows()
            }
        ),
    )

    rows = select_smoke_banks(df)

    print(
        "\nSelected banks:",
        [
            int(
                r.bankID
                if "bankID" in r.index
                else r.name
            )
            for r in rows
        ],
    )

    # ----------------------------------------------------------
    # First real-bank smoke:
    # enough to validate integration.
    # ----------------------------------------------------------

    N_MC = 8192

    results = []

    for j,row in enumerate(
        rows,
        1,
    ):
        bid = int(
            row.bankID
            if "bankID" in row.index
            else row.name
        )

        print("\n" + "="*100)
        print(
            f"[{j}/{len(rows)}] "
            f"bank={bid} | "
            f"{row.scenario_BR} / "
            f"{row.scenario_RU} | "
            f"nRIS={total_ris_ports(row)}"
        )
        print("="*100)

        r = run_bank(
            row,
            N_MC=N_MC,
            device=device,
        )

        results.append(r)

        print(
            f"meanVecErr={r['meanVecErr_pct']:.3f}% | "
            f"covErr={r['covFroErr_pct']:.3f}%"
        )

        print(
            f"meanAPE={r['meanAPE_pct']:.3f}% | "
            f"MC/Wick={r['MC_over_Wick']:.5f} | "
            f"WickAPE={r['wickAPE_pct']:.3f}%"
        )

    out = pd.DataFrame(results)

    print("\n")
    print("="*120)
    print("CASE 1 REAL-BANK SMOKE SUMMARY")
    print("="*120)

    cols = [
        "bankID",
        "scenario_BR",
        "scenario_RU",
        "nT",
        "nR",
        "nRIS",
        "K_BR",
        "K_RU",
        "meanAPE_pct",
        "covFroErr_pct",
        "MC_over_Wick",
        "wickAPE_pct",
    ]

    print(
        out[cols].to_string(
            index=False
        )
    )

    output = Path(
        "/content/drive/MyDrive/"
        "Phy_Informed_MIMO_RIS_Env/"
        "case1_independent_phase/"
        "real_bank_smoke.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output,
        index=False,
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
