from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

import numpy as np
import pandas as pd
import torch

from ris_env.channel_realizations import (
    generate_native_link_chunk,
)

from ris_env.label_engine import (
    row_to_link_configs,
)


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@dataclass
class StatefulMC:
    """
    Persistent Monte-Carlo state.

    Keeps:
        - BR/RU RNG states
        - sum(Y)
        - sum(Y^2)
        - number of accumulated channel realizations
    """

    row: pd.Series

    W: torch.Tensor
    gamma: torch.Tensor

    br: Any
    ru: Any

    gen_br: torch.Generator
    gen_ru: torch.Generator

    sum1: torch.Tensor
    sum2: torch.Tensor

    n_samples: int

    seed_br: int
    seed_ru: int

    mc_chunk: int
    w_chunk: int
    z_chunk: int

    device: torch.device
    parity: bool


@torch.inference_mode()
def create_stateful_mc(
    row: pd.Series,
    W: torch.Tensor,
    gamma: torch.Tensor,
    *,
    mc_chunk: int = 256,
    w_chunk: int = 4,
    z_chunk: int = 64,
    device=None,
    parity: bool = False,
    seed_br: Optional[int] = None,
    seed_ru: Optional[int] = None,
) -> StatefulMC:
    """
    Create an EMPTY MC accumulator.

    Important:
        No channel sample is generated here.
    """

    dev = _device(device)

    br, ru = row_to_link_configs(row)

    cd = (
        torch.complex128
        if parity
        else torch.complex64
    )

    W = torch.as_tensor(
        W,
        dtype=cd,
        device=dev,
    )

    gamma = torch.as_tensor(
        gamma,
        dtype=cd,
        device=dev,
    )

    if W.ndim != 2:
        raise ValueError(
            "W must be [K,nT]."
        )

    if gamma.ndim not in (2, 3):
        raise ValueError(
            "gamma must be shared [C,nRIS] or paired [K,C,nRIS]."
        )

    Knum, nT = W.shape
    if gamma.ndim == 2:
        C, nRIS = gamma.shape
    else:
        Kgamma, C, nRIS = gamma.shape
        if Kgamma != Knum:
            raise ValueError(
                f"paired gamma K={Kgamma}, expected W K={Knum}."
            )

    expected_nT = (
        2
        * int(row.nT1)
        * int(row.nT2)
    )

    if nT != expected_nT:
        raise ValueError(
            f"W nT={nT}, "
            f"expected {expected_nT}"
        )

    if nRIS != int(row.nRIS):
        raise ValueError(
            f"gamma nRIS={nRIS}, "
            f"expected {int(row.nRIS)}"
        )


    # ----------------------------------------------------
    # EXACTLY same seed convention as legacy MC engine
    # ----------------------------------------------------

    seed0 = int(row.ch_seed)

    if seed_br is None:
        seed_br = (
            seed0
            + 70_000_121
        )

    if seed_ru is None:
        seed_ru = (
            seed0
            + 80_000_147
        )


    gen_br = torch.Generator(
        device=dev
    )

    gen_ru = torch.Generator(
        device=dev
    )

    gen_br.manual_seed(
        int(seed_br)
    )

    gen_ru.manual_seed(
        int(seed_ru)
    )


    # ----------------------------------------------------
    # Running sufficient statistics
    # ----------------------------------------------------

    # Same convention as current production engine:
    # accumulate in float64.

    sum1 = torch.zeros(
        (Knum, C),
        dtype=torch.float64,
        device=dev,
    )

    sum2 = torch.zeros(
        (Knum, C),
        dtype=torch.float64,
        device=dev,
    )


    return StatefulMC(
        row=row,

        W=W,
        gamma=gamma,

        br=br,
        ru=ru,

        gen_br=gen_br,
        gen_ru=gen_ru,

        sum1=sum1,
        sum2=sum2,

        n_samples=0,

        seed_br=int(seed_br),
        seed_ru=int(seed_ru),

        mc_chunk=int(mc_chunk),
        w_chunk=int(w_chunk),
        z_chunk=int(z_chunk),

        device=dev,
        parity=bool(parity),
    )


@torch.inference_mode()
def advance_stateful_mc(
    state: StatefulMC,
    additional_samples: int,
) -> Dict[str, Any]:
    """
    Continue the SAME MC stream.

    Example:

        advance(..., 64000)
        advance(..., 64000)

    gives a single continuous 128k stream.

    The RNGs are NOT reseeded.
    """

    additional_samples = int(
        additional_samples
    )

    if additional_samples <= 0:
        raise ValueError(
            "additional_samples "
            "must be positive."
        )


    dev = state.device

    W = state.W
    gamma = state.gamma

    Knum = int(
        W.shape[0]
    )

    C = int(gamma.shape[0] if gamma.ndim == 2 else gamma.shape[1])


    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            dev
        )


    _sync(dev)

    t0 = time.perf_counter()

    added = 0


    while added < additional_samples:

        n = min(
            state.mc_chunk,
            additional_samples - added,
        )


        # ------------------------------------------------
        # IMPORTANT:
        # same generator objects continue from
        # their previous internal RNG states.
        # ------------------------------------------------

        HBR = generate_native_link_chunk(
            state.br,
            n,
            generator=state.gen_br,
            device=dev,
            parity=state.parity,
        )

        HRU = generate_native_link_chunk(
            state.ru,
            n,
            generator=state.gen_ru,
            device=dev,
            parity=state.parity,
        )


        N = int(
            HBR.shape[0]
        )

        R = int(
            HRU.shape[1]
        )

        I = int(
            HBR.shape[1]
        )


        # ------------------------------------------------
        # Same W/Z contraction as production MC engine
        # ------------------------------------------------

        for k0 in range(
            0,
            Knum,
            state.w_chunk,
        ):

            k1 = min(
                k0 + state.w_chunk,
                Knum,
            )

            Wc = W[k0:k1]


            # [N,I,T] @ [T,Kw]
            # -> [N,Kw,I]

            u = torch.matmul(
                HBR,
                Wc.T,
            ).permute(
                0,
                2,
                1,
            ).contiguous()


            # [N,Kw,R,I]

            basis = (
                HRU[:, None, :, :]
                * u[:, :, None, :]
            )


            basis_flat = (
                basis.reshape(N * (k1-k0) * R, I)
                if gamma.ndim == 2
                else None
            )


            for c0 in range(
                0,
                C,
                state.z_chunk,
            ):

                c1 = min(
                    c0 + state.z_chunk,
                    C,
                )

                if gamma.ndim == 2:
                    gc = gamma[c0:c1]
                    eff = (basis_flat @ gc.T).reshape(
                        N, k1-k0, R, c1-c0
                    )
                else:
                    gc = gamma[k0:k1, c0:c1, :]
                    eff = torch.einsum("nkri,kci->nkrc", basis, gc)


                Y = torch.sum(
                    torch.abs(eff)**2,
                    dim=2,
                )


                Y64 = Y.to(
                    torch.float64
                )


                state.sum1[
                    k0:k1,
                    c0:c1,
                ] += torch.sum(
                    Y64,
                    dim=0,
                )


                state.sum2[
                    k0:k1,
                    c0:c1,
                ] += torch.sum(
                    Y64 * Y64,
                    dim=0,
                )


                del eff
                del Y
                del Y64


            del u
            del basis
            if basis_flat is not None:
                del basis_flat


        del HBR
        del HRU


        added += N

        state.n_samples += N


    _sync(dev)

    elapsed = (
        time.perf_counter()
        - t0
    )


    peak_mb = np.nan

    if dev.type == "cuda":

        peak_mb = float(
            torch.cuda.max_memory_allocated(
                dev
            )
            / 1024**2
        )


    return {
        "added_samples":
            int(added),

        "n_samples":
            int(state.n_samples),

        "seconds":
            float(elapsed),

        "peak_memory_MB":
            peak_mb,
    }


@torch.inference_mode()
def snapshot_stateful_mc(
    state: StatefulMC,
) -> Dict[str, Any]:
    """
    Compute current MC moments without
    changing RNG state.

    Population variance:

        var = E[Y^2] - E[Y]^2
    """

    if state.n_samples <= 0:

        raise RuntimeError(
            "State contains "
            "zero samples."
        )


    n = float(
        state.n_samples
    )


    mean = (
        state.sum1
        / n
    )


    var = torch.clamp(
        state.sum2 / n
        - mean * mean,

        min=0.0,
    )


    Knum, C = mean.shape


    return {
        "meanEmp":
            mean,

        "varEmp":
            var,

        "n_samples":
            int(state.n_samples),

        "candidate_count":
            int(Knum * C),

        "sample_evaluations":
            int(
                state.n_samples
                * Knum
                * C
            ),

        "seed_br":
            int(state.seed_br),

        "seed_ru":
            int(state.seed_ru),
    }
