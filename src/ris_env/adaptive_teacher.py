from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from ris_env.stateful_mc import (
    create_stateful_mc,
    advance_stateful_mc,
    snapshot_stateful_mc,
)

from ris_env.adaptive_mc import (
    create_selective_refiner,
    advance_selective_refiner,
    snapshot_selective_refiner,
    deactivate_local_w,
)

from ris_env.label_engine import (
    analytic_mu_snr_multi_w_z,
    flatten_label_result,
)

from ris_env.gamma_gamma_log import (
    torch_log_lookup,
    symmetric_gg_logq05_torch,
)

from ris_env.teacher_pipeline import (
    load_packaged_gg_lookup,
)
from ris_env.candidate_design import evaluate_paired_analytic_stats


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.inference_mode()
def run_teacher_bank_adaptive(
    prepared,
    *,
    lookup=None,

    # ----------------------------------------------------
    # Frozen adaptive policy
    # ----------------------------------------------------
    half_n: int = 32_000,
    base_n: int = 64_000,

    early_p99_threshold: float = 0.05,
    stability_p90_threshold: float = 0.10,

    max_n: int = 512_000,

    # ----------------------------------------------------
    # GPU chunking
    #
    # IMPORTANT:
    # 4000 divides 32k / 64k / 128k exactly.
    # This preserves the validated stream/chunk boundaries.
    # ----------------------------------------------------
    mc_chunk: int = 4000,
    w_chunk: int = 8,
    z_chunk: int = 128,

    device=None,
    parity: bool = False,
) -> Dict[str, Any]:
    """
    Adaptive Monte-Carlo Teacher for one prepared bank.

    Policy
    ------

    All W:
        0 -> 32k -> 64k

    Initial refinement trigger:
        P99_z |logQ64 - logQ32| > 0.05

    Selected W only:
        64k -> 128k -> 256k -> ...

    Continue while:
        P90_z |logQ_new - logQ_previous| > 0.10

    Stop when stable, or at max_n.

    Final supervised target remains:
        logQ05GG(meanEmp_MC, varEmp_MC)

    Analytic muSNR is retained only as the analytic
    physics feature / diagnostic.
    """

    half_n = int(half_n)
    base_n = int(base_n)
    max_n = int(max_n)

    if half_n <= 0:
        raise ValueError("half_n must be positive.")

    if base_n != 2 * half_n:
        raise ValueError(
            "Expected base_n == 2*half_n. "
            f"Got half_n={half_n}, base_n={base_n}."
        )

    if max_n < base_n:
        raise ValueError(
            "max_n must be >= base_n."
        )

    if (
        half_n % int(mc_chunk) != 0
        or base_n % int(mc_chunk) != 0
    ):
        raise ValueError(
            "For validated deterministic chunk boundaries, "
            "mc_chunk must divide half_n and base_n exactly."
        )


    dev = torch.device(
        device
        if device is not None
        else prepared.W.device
    )

    if lookup is None:
        lookup = load_packaged_gg_lookup()

    tlookup = torch_log_lookup(
        lookup,
        device=dev,
        dtype=torch.float64,
    )


    # ====================================================
    # Timing / memory
    # ====================================================

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    _sync(dev)
    t_total = time.perf_counter()


    # ====================================================
    # 1. Create common full-W MC stream
    # ====================================================

    state = create_stateful_mc(
        prepared.row_physical,
        prepared.W,
        prepared.gamma,
        mc_chunk=mc_chunk,
        w_chunk=w_chunk,
        z_chunk=z_chunk,
        device=dev,
        parity=parity,
    )


    # ====================================================
    # 2. 32k snapshot
    # ====================================================

    info32 = advance_stateful_mc(
        state,
        additional_samples=half_n,
    )

    snap32 = snapshot_stateful_mc(state)

    gg32 = symmetric_gg_logq05_torch(
        snap32["meanEmp"],
        snap32["varEmp"],
        tlookup,
    )

    log32 = gg32["logQ05GG"].clone()


    # ====================================================
    # 3. +32k => base 64k snapshot
    # ====================================================

    info64 = advance_stateful_mc(
        state,
        additional_samples=half_n,
    )

    snap64 = snapshot_stateful_mc(state)

    gg64 = symmetric_gg_logq05_torch(
        snap64["meanEmp"],
        snap64["varEmp"],
        tlookup,
    )

    log64 = gg64["logQ05GG"].clone()


    # ====================================================
    # 4. Frozen initial trigger
    #
    # [K,C] -> P99 over C (Z candidates)
    # ====================================================

    d_early = torch.abs(
        log64 - log32
    )

    early_p99 = torch.quantile(
        d_early,
        q=0.99,
        dim=1,
    )

    selected_mask = (
        early_p99
        > float(early_p99_threshold)
    )

    selected_w = torch.nonzero(
        selected_mask,
        as_tuple=False,
    ).reshape(-1)


    K, C = snap64["meanEmp"].shape


    # ====================================================
    # 5. Start final result from 64k baseline
    # ====================================================

    final_mean = (
        snap64["meanEmp"]
        .clone()
    )

    final_var = (
        snap64["varEmp"]
        .clone()
    )

    n_per_w = torch.full(
        (K,),
        base_n,
        dtype=torch.int64,
        device=dev,
    )

    # Unselected W is considered converged at base level.
    converged_w = (
        ~selected_mask
    ).clone()

    refined_w = selected_mask.clone()

    # Last observed P90 target movement for each W.
    final_p90_delta = torch.zeros(
        (K,),
        dtype=torch.float64,
        device=dev,
    )

    history = []


    # ====================================================
    # 6. Selective refinement
    # ====================================================

    selective_seconds = 0.0

    if selected_w.numel() > 0:

        refiner = create_selective_refiner(
            state,
            selected_w.detach().cpu().tolist(),
        )

        # Selected rows are not converged yet.
        converged_w[
            selected_w
        ] = False


        while True:

            active_local = torch.nonzero(
                refiner.active,
                as_tuple=False,
            ).reshape(-1)

            if active_local.numel() == 0:
                break


            active_n = (
                refiner.n_samples_per_w[
                    active_local
                ]
            )

            # All surviving active W must be at same level.
            unique_n = torch.unique(
                active_n
            )

            if unique_n.numel() != 1:
                raise RuntimeError(
                    "Active selective W have inconsistent "
                    "MC sample counts."
                )

            current_n = int(
                unique_n.item()
            )


            # ------------------------------------------------
            # Max budget reached before convergence
            # ------------------------------------------------

            if current_n >= max_n:

                global_w = refiner.w_indices[
                    active_local
                ]

                converged_w[
                    global_w
                ] = False

                deactivate_local_w(
                    refiner,
                    active_local,
                )

                history.append({
                    "from_n": current_n,
                    "to_n": current_n,
                    "active_w":
                        global_w.detach().cpu().tolist(),
                    "stable_w": [],
                    "unstable_w":
                        global_w.detach().cpu().tolist(),
                    "max_budget_reached": True,
                })

                break


            # ------------------------------------------------
            # Snapshot BEFORE next doubling
            # ------------------------------------------------

            prev = snapshot_selective_refiner(
                refiner
            )

            gg_prev = symmetric_gg_logq05_torch(
                prev["meanEmp"],
                prev["varEmp"],
                tlookup,
            )

            prev_log = (
                gg_prev["logQ05GG"]
                .clone()
            )


            # ------------------------------------------------
            # Double MC budget
            #
            # 64 -> 128
            # 128 -> 256
            # ...
            # ------------------------------------------------

            add_n = min(
                current_n,
                max_n - current_n,
            )

            t_sel = time.perf_counter()

            info = advance_selective_refiner(
                refiner,
                additional_samples=add_n,
                active_local_indices=
                    active_local.detach().cpu().tolist(),
            )

            selective_seconds += (
                time.perf_counter()
                - t_sel
            )


            # ------------------------------------------------
            # Snapshot AFTER refinement
            # ------------------------------------------------

            curr = snapshot_selective_refiner(
                refiner
            )

            gg_curr = symmetric_gg_logq05_torch(
                curr["meanEmp"],
                curr["varEmp"],
                tlookup,
            )

            curr_log = (
                gg_curr["logQ05GG"]
            )


            # ------------------------------------------------
            # Actual target stability:
            #
            # P90 over Z
            # ------------------------------------------------

            delta = torch.abs(
                curr_log[
                    active_local
                ]
                -
                prev_log[
                    active_local
                ]
            )

            p90 = torch.quantile(
                delta,
                q=0.90,
                dim=1,
            )

            stable_now = (
                p90
                <= float(
                    stability_p90_threshold
                )
            )


            global_active_w = (
                refiner.w_indices[
                    active_local
                ]
            )

            final_p90_delta[
                global_active_w
            ] = p90


            stable_local = (
                active_local[
                    stable_now
                ]
            )

            unstable_local = (
                active_local[
                    ~stable_now
                ]
            )

            stable_global = (
                refiner.w_indices[
                    stable_local
                ]
            )

            unstable_global = (
                refiner.w_indices[
                    unstable_local
                ]
            )


            # Stable W permanently leaves refinement.
            if stable_local.numel() > 0:

                converged_w[
                    stable_global
                ] = True

                deactivate_local_w(
                    refiner,
                    stable_local,
                )


            new_n = (
                current_n
                + add_n
            )


            # If an unstable W just hit max_n,
            # it remains a valid label but is flagged
            # mcConverged=False.
            if (
                new_n >= max_n
                and unstable_local.numel() > 0
            ):

                converged_w[
                    unstable_global
                ] = False

                deactivate_local_w(
                    refiner,
                    unstable_local,
                )


            history.append({
                "from_n": int(current_n),
                "to_n": int(new_n),

                "active_w":
                    global_active_w
                    .detach()
                    .cpu()
                    .tolist(),

                "stable_w":
                    stable_global
                    .detach()
                    .cpu()
                    .tolist(),

                "unstable_w":
                    unstable_global
                    .detach()
                    .cpu()
                    .tolist(),

                "p90_delta":
                    p90
                    .detach()
                    .cpu()
                    .tolist(),

                "max_budget_reached":
                    bool(
                        new_n >= max_n
                        and unstable_local.numel() > 0
                    ),
            })


        # ====================================================
        # 7. Merge selected final MC moments back into
        #    full [K,C] matrices
        # ====================================================

        final_selected = (
            snapshot_selective_refiner(
                refiner
            )
        )

        selected_global = (
            final_selected["w_indices"]
        )

        final_mean[
            selected_global
        ] = final_selected[
            "meanEmp"
        ]

        final_var[
            selected_global
        ] = final_selected[
            "varEmp"
        ]

        n_per_w[
            selected_global
        ] = final_selected[
            "n_samples_per_w"
        ]


    # ====================================================
    # 8. Final GG target from FINAL empirical moments
    # ====================================================

    _sync(dev)
    t_gg = time.perf_counter()

    gg_final = symmetric_gg_logq05_torch(
        final_mean,
        final_var,
        tlookup,
    )

    _sync(dev)

    gg_seconds = (
        time.perf_counter()
        - t_gg
    )


    # ====================================================
    # 9. Analytic physics mean -- NOT teacher mean
    # ====================================================

    _sync(dev)
    t_mu = time.perf_counter()

    if prepared.gamma.ndim == 2:
        analytic = {
            "muSNR": analytic_mu_snr_multi_w_z(
                prepared.static_env,
                prepared.W,
                prepared.gamma,
                z_chunk=z_chunk,
            )
        }
    else:
        analytic = evaluate_paired_analytic_stats(
            prepared.static_env,
            prepared.W,
            prepared.gamma,
            z_chunk=z_chunk,
        )
    mu_snr = analytic["muSNR"]

    _sync(dev)

    analytic_seconds = (
        time.perf_counter()
        - t_mu
    )


    # ====================================================
    # 10. Standard result contract
    # ====================================================

    result = {
        "muSNR":
            mu_snr,

        "meanEmp":
            final_mean,

        "varEmp":
            final_var,

        "logQ05GG":
            gg_final["logQ05GG"],

        "q05GG":
            gg_final["q05GG"],

        "shapeA":
            gg_final["shapeA"],

        "cv2":
            gg_final["cv2"],

        "lookupClamped":
            gg_final["lookupClamped"],

        # Adaptive diagnostics
        "N_MC_used":
            n_per_w[:, None]
            .expand(K, C),

        "mcConverged":
            converged_w[:, None]
            .expand(K, C),

        "mcRefined":
            refined_w[:, None]
            .expand(K, C),

        "mcEarlyP99":
            early_p99[:, None]
            .expand(K, C),

        "mcFinalP90Delta":
            final_p90_delta[:, None]
            .expand(K, C),
    }

    if prepared.candidate_design == "variance_ratio_v1":
        var64 = final_var.to(torch.float64)
        wick64 = analytic["sigma2Wick"].to(torch.float64)
        finite_positive = (
            torch.isfinite(var64) & torch.isfinite(wick64)
            & (var64 > 0.0) & (wick64 > 0.0)
        )
        tiny = torch.finfo(torch.float64).tiny
        log_var = torch.log(torch.clamp(var64, min=tiny))
        log_wick = torch.log(torch.clamp(wick64, min=tiny))
        target = log_var - log_wick
        result.update({
            "sigma2Wick": wick64,
            "wickCV2": analytic["wickCV2"].to(torch.float64),
            "Neff": analytic["Neff"].to(torch.float64),
            "varRatio": torch.exp(target),
            "targetLogVarRatio": target,
            "targetIdentityError": torch.abs(target - (log_var - log_wick)),
            "analyticValid": finite_positive,
            "isReliable": finite_positive & result["mcConverged"],
        })


    # ====================================================
    # 11. Existing 32x512 flattening contract
    # ====================================================

    labels = flatten_label_result(
        result,
        prepared.WIdx,
        prepared.z,
    )


    # Candidate metadata
    zc = labels[
        "zCandidate"
    ].to_numpy(
        dtype=np.int64
    )

    if prepared.candidate_metadata is None:
        labels["candidateType"] = prepared.z_candidate_type[zc]
        labels["anchorIndex"] = prepared.z_anchor_index[zc]
    else:
        meta = prepared.candidate_metadata
        labels["candidateType"] = meta["candidate_type"].reshape(-1)
        labels["anchorIndex"] = -1
        labels["canonicalSplit"] = meta["canonical_split"].reshape(-1)
        labels["optimizationObjective"] = meta[
            "optimization_objective"
        ].reshape(-1)
        labels["optimizationSeedRank"] = meta[
            "optimization_seed_rank"
        ].reshape(-1)
        labels["optimizationSweepCount"] = meta[
            "optimization_sweep_count"
        ].reshape(-1)
        labels["isOptimized"] = meta["is_optimized"].reshape(-1)
        labels["isDuplicate"] = meta["is_duplicate"].reshape(-1)
        labels["optimizationInitialObjective"] = meta[
            "optimization_initial_objective"
        ].reshape(-1)
        labels["optimizationFinalObjective"] = meta[
            "optimization_final_objective"
        ].reshape(-1)
        labels["optimizationAcceptedFlips"] = meta[
            "optimization_accepted_flips"
        ].reshape(-1)


    # Adaptive fields follow same K-major/C-minor
    # flatten order as flatten_label_result.
    labels["N_MC_used"] = (
        result["N_MC_used"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels["mcConverged"] = (
        result["mcConverged"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(bool)
    )

    labels["mcRefined"] = (
        result["mcRefined"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(bool)
    )

    labels["mcEarlyP99"] = (
        result["mcEarlyP99"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels["mcFinalP90Delta"] = (
        result["mcFinalP90Delta"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    for name in (
        "sigma2Wick", "wickCV2", "Neff", "varRatio",
        "targetLogVarRatio", "targetIdentityError",
        "analyticValid", "isReliable",
    ):
        if name in result:
            values = result[name].detach().cpu().numpy().reshape(-1)
            if name in ("analyticValid", "isReliable"):
                values = values.astype(bool)
            labels[name] = values


    # ====================================================
    # 12. Existing bank identity metadata
    # ====================================================

    raw = prepared.row_raw

    prefix = {
        "bankID":
            int(raw.bankID),

        "splitID":
            str(
                raw.splitID
                if "splitID" in raw.index
                else raw.split
            ),

        "scenario_BR":
            str(raw.scenario_BR),

        "scenario_RU":
            str(raw.scenario_RU),

        "fc":
            float(raw.fc),

        "nT1":
            int(raw.nT1),

        "nT2":
            int(raw.nT2),

        "nT":
            int(raw.nT),

        "nR1":
            int(raw.nR1),

        "nR2":
            int(raw.nR2),

        "nR":
            int(raw.nR),

        "nRIS1":
            int(raw.nRIS1),

        "nRIS2":
            int(raw.nRIS2),

        "nRIS":
            int(raw.nRIS),

        # Fixed initial production budget.
        "N_MC_base":
            int(base_n),
    }

    for key, value in reversed(
        list(prefix.items())
    ):
        labels.insert(
            0,
            key,
            value,
        )


    # ====================================================
    # 13. Timing / diagnostics
    # ====================================================

    _sync(dev)

    total_seconds = (
        time.perf_counter()
        - t_total
    )

    peak_mb = np.nan

    if dev.type == "cuda":
        peak_mb = float(
            torch.cuda.max_memory_allocated(
                dev
            )
            / 1024**2
        )


    # Effective sample evaluations.
    # Each W has C Z candidates.
    effective_samples = int(
        torch.sum(
            n_per_w
        ).item()
        * C
    )


    result.update({
        "labels":
            labels,

        "selected_w":
            selected_w
            .detach()
            .cpu()
            .tolist(),

        "n_mc_per_w":
            n_per_w
            .detach()
            .cpu()
            .tolist(),

        "converged_per_w":
            converged_w
            .detach()
            .cpu()
            .tolist(),

        "early_p99_per_w":
            early_p99
            .detach()
            .cpu()
            .tolist(),

        "final_p90_delta_per_w":
            final_p90_delta
            .detach()
            .cpu()
            .tolist(),

        "refinement_history":
            history,

        "base_32k_seconds":
            float(
                info32["seconds"]
            ),

        "base_second_32k_seconds":
            float(
                info64["seconds"]
            ),

        "selective_seconds":
            float(
                selective_seconds
            ),

        "analytic_mu_seconds":
            float(
                analytic_seconds
            ),

        "gg_final_seconds":
            float(
                gg_seconds
            ),

        "total_seconds":
            float(
                total_seconds
            ),

        "candidate_count":
            int(K * C),

        "effective_sample_evaluations":
            effective_samples,

        "peak_memory_MB":
            peak_mb,

        "base_n":
            int(base_n),

        "max_n":
            int(max_n),

        "early_p99_threshold":
            float(
                early_p99_threshold
            ),

        "stability_p90_threshold":
            float(
                stability_p90_threshold
            ),
    })


    return result
