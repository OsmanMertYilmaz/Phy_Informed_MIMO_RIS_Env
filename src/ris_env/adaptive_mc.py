from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import time

import numpy as np
import torch

from ris_env.channel_realizations import (
    generate_native_link_chunk,
)

from ris_env.stateful_mc import (
    StatefulMC,
)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@dataclass
class SelectiveMCRefiner:
    """
    Continue an existing full-W StatefulMC stream,
    but accumulate new realizations only for selected W indices.

    Important:
    - Initial sums are copied from the baseline StatefulMC.
    - RNG states continue exactly from the baseline stream.
    - New W indices cannot be added later after they have missed samples.
    """

    base_n: int

    W: torch.Tensor
    gamma: torch.Tensor

    w_indices: torch.Tensor

    br: Any
    ru: Any

    gen_br: torch.Generator
    gen_ru: torch.Generator

    sum1: torch.Tensor
    sum2: torch.Tensor

    n_samples_per_w: torch.Tensor

    active: torch.Tensor

    seed_br: int
    seed_ru: int

    mc_chunk: int
    w_chunk: int
    z_chunk: int

    device: torch.device
    parity: bool


@torch.inference_mode()
def create_selective_refiner(
    baseline: StatefulMC,
    w_indices: Sequence[int],
) -> SelectiveMCRefiner:
    """
    Fork a selective refinement state from a completed baseline.

    Example:
        baseline contains all W at N=64k.
        w_indices=[15,17]

    The refiner starts W15 and W17 from the baseline 64k
    sufficient statistics, and its RNG starts exactly where
    baseline's RNG stopped.
    """

    if baseline.n_samples <= 0:
        raise ValueError(
            "Baseline must contain MC samples."
        )

    idx = torch.as_tensor(
        list(w_indices),
        dtype=torch.long,
        device=baseline.device,
    )

    if idx.ndim != 1 or idx.numel() == 0:
        raise ValueError(
            "w_indices must contain at least one W."
        )

    if torch.unique(idx).numel() != idx.numel():
        raise ValueError(
            "w_indices contains duplicates."
        )

    if torch.any(idx < 0) or torch.any(
        idx >= baseline.W.shape[0]
    ):
        raise IndexError(
            "w_indices outside valid W range."
        )


    # ----------------------------------------------------
    # Clone RNG state AT THE END OF THE BASELINE STREAM.
    # Do not reseed.
    # ----------------------------------------------------

    gen_br = torch.Generator(
        device=baseline.device
    )

    gen_ru = torch.Generator(
        device=baseline.device
    )

    gen_br.set_state(
        baseline.gen_br.get_state()
    )

    gen_ru.set_state(
        baseline.gen_ru.get_state()
    )


    # ----------------------------------------------------
    # Start selected Ws from their existing baseline sums.
    # ----------------------------------------------------

    sum1 = (
        baseline.sum1[idx]
        .clone()
    )

    sum2 = (
        baseline.sum2[idx]
        .clone()
    )

    n_selected = int(
        idx.numel()
    )

    n_samples_per_w = torch.full(
        (n_selected,),
        int(baseline.n_samples),
        dtype=torch.int64,
        device=baseline.device,
    )

    active = torch.ones(
        (n_selected,),
        dtype=torch.bool,
        device=baseline.device,
    )


    return SelectiveMCRefiner(
        base_n=int(baseline.n_samples),

        W=baseline.W[idx],
        gamma=(
            baseline.gamma
            if baseline.gamma.ndim == 2
            else baseline.gamma[idx]
        ),

        w_indices=idx,

        br=baseline.br,
        ru=baseline.ru,

        gen_br=gen_br,
        gen_ru=gen_ru,

        sum1=sum1,
        sum2=sum2,

        n_samples_per_w=n_samples_per_w,
        active=active,

        seed_br=int(baseline.seed_br),
        seed_ru=int(baseline.seed_ru),

        mc_chunk=int(baseline.mc_chunk),
        w_chunk=int(baseline.w_chunk),
        z_chunk=int(baseline.z_chunk),

        device=baseline.device,
        parity=bool(baseline.parity),
    )


@torch.inference_mode()
def advance_selective_refiner(
    state: SelectiveMCRefiner,
    additional_samples: int,
    *,
    active_local_indices=None,
) -> Dict[str, Any]:
    """
    Add NEW common channel realizations only to active selected Ws.

    active_local_indices refers to positions inside state.w_indices.

    Example:
        state.w_indices = [15,17]

        active_local_indices = [0,1]
            -> refine W15 and W17

        later active_local_indices = [1]
            -> only W17 continues

    A previously inactive W must NOT be reactivated after missing samples.
    """

    additional_samples = int(
        additional_samples
    )

    if additional_samples <= 0:
        raise ValueError(
            "additional_samples must be positive."
        )


    if active_local_indices is None:
        local = torch.nonzero(
            state.active,
            as_tuple=False,
        ).reshape(-1)

    else:
        local = torch.as_tensor(
            active_local_indices,
            dtype=torch.long,
            device=state.device,
        ).reshape(-1)


    if local.numel() == 0:
        raise ValueError(
            "No active W to refine."
        )

    if torch.any(local < 0) or torch.any(
        local >= state.W.shape[0]
    ):
        raise IndexError(
            "active_local_indices out of range."
        )

    if torch.unique(local).numel() != local.numel():
        raise ValueError(
            "active_local_indices contains duplicates."
        )

    # A W that was permanently stopped cannot be reactivated.
    if not torch.all(state.active[local]):
        raise RuntimeError(
            "Cannot reactivate a W after it has stopped refinement."
        )


    dev = state.device
    gamma = state.gamma

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


        N = int(HBR.shape[0])
        R = int(HRU.shape[1])
        I = int(HBR.shape[1])


        # ------------------------------------------------
        # Only selected ACTIVE W rows are contracted.
        # ------------------------------------------------

        for j0 in range(
            0,
            int(local.numel()),
            state.w_chunk,
        ):

            j1 = min(
                j0 + state.w_chunk,
                int(local.numel()),
            )

            loc = local[
                j0:j1
            ]

            Wc = state.W[
                loc
            ]


            u = torch.matmul(
                HBR,
                Wc.T,
            ).permute(
                0,
                2,
                1,
            ).contiguous()


            basis = (
                HRU[:, None, :, :]
                * u[:, :, None, :]
            )


            basis_flat = (
                basis.reshape(N * int(loc.numel()) * R, I)
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
                        N, int(loc.numel()), R, c1-c0
                    )
                else:
                    gc = gamma[loc, c0:c1, :]
                    eff = torch.einsum("njri,jci->njrc", basis, gc)


                Y = torch.sum(
                    torch.abs(eff)**2,
                    dim=2,
                )


                Y64 = Y.to(
                    torch.float64
                )


                state.sum1[
                    loc,
                    c0:c1,
                ] += torch.sum(
                    Y64,
                    dim=0,
                )


                state.sum2[
                    loc,
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


    # Every active W processed exactly the same newly generated
    # realization block.
    state.n_samples_per_w[
        local
    ] += int(added)


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

        "active_local_indices":
            local.detach().cpu().tolist(),

        "active_w_indices":
            state.w_indices[
                local
            ].detach().cpu().tolist(),

        "n_samples_per_w":
            state.n_samples_per_w
            .detach()
            .cpu()
            .tolist(),

        "seconds":
            float(elapsed),

        "peak_memory_MB":
            peak_mb,
    }


@torch.inference_mode()
def deactivate_local_w(
    state: SelectiveMCRefiner,
    local_indices,
) -> None:
    """
    Permanently stop refinement for selected local W positions.
    """

    idx = torch.as_tensor(
        local_indices,
        dtype=torch.long,
        device=state.device,
    ).reshape(-1)

    if idx.numel() == 0:
        return

    state.active[idx] = False


@torch.inference_mode()
def snapshot_selective_refiner(
    state: SelectiveMCRefiner,
) -> Dict[str, Any]:
    """
    Return current moments for selected W rows.
    """

    n = (
        state.n_samples_per_w
        .to(torch.float64)
        [:, None]
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


    return {
        "w_indices":
            state.w_indices,

        "meanEmp":
            mean,

        "varEmp":
            var,

        "n_samples_per_w":
            state.n_samples_per_w,

        "active":
            state.active,
    }
