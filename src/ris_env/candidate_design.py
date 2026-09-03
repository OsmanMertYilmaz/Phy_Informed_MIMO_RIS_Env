"""Frozen 480-canonical + 32 W-specific RIS candidate design.

The optimization path uses exact coordinate updates of the analytical
quadratic forms.  It therefore evaluates one full bit sweep without rebuilding
the large RIS covariance contractions after every trial flip.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from ris_env.ris_response import generate_ris_response_from_z
from ris_env.snr_statistics import (
    WState,
    evaluate_gamma_batch,
    prepare_w_state,
)


CANONICAL_COUNTS = {
    "anchor": (4, 3),
    "block": (64, 45),
    "structured": (64, 45),
    "random": (348, 243),
}
OPTIMIZATION_OBJECTIVES = (
    "muSNR_max",
    "wickCV2_min",
    "wickCV2_max",
    "Neff_min",
)


def _add_unique(
    store: List[np.ndarray],
    labels: List[str],
    seen: set[bytes],
    z: np.ndarray,
    label: str,
) -> bool:
    z = np.asarray(z, dtype=np.uint8).reshape(-1)
    key = z.tobytes()
    if key in seen:
        return False
    seen.add(key)
    store.append(z.copy())
    labels.append(label)
    return True


def _anchor_patterns(n_ris: int) -> List[np.ndarray]:
    idx = np.arange(n_ris, dtype=np.int64)
    return [
        np.zeros(n_ris, dtype=np.uint8),
        np.ones(n_ris, dtype=np.uint8),
        (idx % 2).astype(np.uint8),
        (1 - idx % 2).astype(np.uint8),
    ]


def _spatial_masks(nris_x: int, nris_y: int):
    rr, cc = np.meshgrid(
        np.arange(int(nris_x)), np.arange(int(nris_y)), indexing="ij"
    )
    return rr.reshape(-1, order="F"), cc.reshape(-1, order="F")


def _block_candidates(nris_x: int, nris_y: int) -> Iterable[np.ndarray]:
    """Deterministic stream of contiguous 2-D block/stripe masks."""
    m, n = int(nris_x), int(nris_y)
    rr, cc = _spatial_masks(m, n)
    for hm in range(1, m + 1):
        for wn in range(1, n + 1):
            for r0 in range(m):
                for c0 in range(n):
                    spatial = (
                        ((rr - r0) % m < hm)
                        & ((cc - c0) % n < wn)
                    ).astype(np.uint8)
                    yield np.concatenate([spatial, spatial])
                    yield np.concatenate([spatial, 1 - spatial])
                    yield np.concatenate([spatial, np.zeros_like(spatial)])
                    yield np.concatenate([np.zeros_like(spatial), spatial])


def _structured_candidates(nris_x: int, nris_y: int) -> Iterable[np.ndarray]:
    rr, cc = _spatial_masks(nris_x, nris_y)
    for period in range(2, 18):
        for a, b in (
            (1, 0), (0, 1), (1, 1), (1, 2), (2, 1),
            (1, 3), (3, 1), (2, 3), (3, 2),
        ):
            for offset in range(period):
                residue = (a * rr + b * cc + offset) % period
                for width in range(1, period):
                    spatial = (residue < width).astype(np.uint8)
                    yield np.concatenate([spatial, spatial])
                    yield np.concatenate([spatial, 1 - spatial])
                    yield np.concatenate([spatial, np.zeros_like(spatial)])
                    yield np.concatenate([np.zeros_like(spatial), spatial])


def build_canonical_z_pool(
    nris_x: int,
    nris_y: int,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Build the frozen 480 canonical patterns and their train/holdout split."""
    n_ris = 2 * int(nris_x) * int(nris_y)
    rng = np.random.default_rng(int(seed))
    store: List[np.ndarray] = []
    labels: List[str] = []
    seen: set[bytes] = set()

    for z in _anchor_patterns(n_ris):
        _add_unique(store, labels, seen, z, "anchor")

    for label, stream in (
        ("block", _block_candidates(nris_x, nris_y)),
        ("structured", _structured_candidates(nris_x, nris_y)),
    ):
        wanted = CANONICAL_COUNTS[label][0]
        made = 0
        for z in stream:
            if _add_unique(store, labels, seen, z, label):
                made += 1
                if made == wanted:
                    break
        if made != wanted:
            raise RuntimeError(f"Could only create {made}/{wanted} {label} patterns.")

    while labels.count("random") < CANONICAL_COUNTS["random"][0]:
        _add_unique(
            store,
            labels,
            seen,
            rng.integers(0, 2, size=n_ris, dtype=np.uint8),
            "random",
        )

    z = np.stack(store, axis=0)
    candidate_type = np.asarray(labels, dtype=object)
    canonical_split = np.empty(len(z), dtype=object)
    for label, (_, n_train) in CANONICAL_COUNTS.items():
        idx = np.flatnonzero(candidate_type == label)
        canonical_split[idx[:n_train]] = "train"
        canonical_split[idx[n_train:]] = "holdout"

    if z.shape != (480, n_ris):
        raise RuntimeError(f"Canonical Z shape={z.shape}, expected={(480, n_ris)}.")
    if len({row.tobytes() for row in z}) != 480:
        raise RuntimeError("Canonical Z pool contains duplicates.")
    return {
        "z": z,
        "candidate_type": candidate_type,
        "canonical_split": canonical_split,
        "is_canonical_train": canonical_split == "train",
    }


def max_min_hamming_order(z: np.ndarray) -> np.ndarray:
    """Stable deterministic max-min Hamming ordering."""
    x = np.asarray(z, dtype=np.uint8)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("z must be a non-empty [C,nRIS] binary matrix.")
    selected = [0]
    available = np.ones(len(x), dtype=bool)
    available[0] = False
    min_dist = np.sum(x != x[0], axis=1).astype(np.int64)
    while len(selected) < len(x):
        scores = np.where(available, min_dist, -1)
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        available[nxt] = False
        min_dist = np.minimum(min_dist, np.sum(x != x[nxt], axis=1))
    return np.asarray(selected, dtype=np.int64)


def _metrics_from_quadratics(
    q: torch.Tensor,
    mu: torch.Tensor,
    n_r: int,
) -> Dict[str, torch.Tensor]:
    tiny = torch.finfo(q.real.dtype).tiny
    second_eff = q[:, :n_r].real
    abs_mu2 = torch.abs(mu) ** 2
    sigma2_eff = torch.clamp(second_eff - abs_mu2, min=0.0)
    mu_snr = torch.clamp(torch.sum(sigma2_eff + abs_mu2, dim=1), min=tiny)
    second_cov = q[:, n_r:].reshape(-1, n_r, n_r)
    cmat = second_cov - mu[:, :, None] * mu.conj()[:, None, :]
    cmat = 0.5 * (cmat + cmat.transpose(-1, -2).conj())
    wick1 = torch.sum(torch.abs(cmat) ** 2, dim=(1, 2))
    wick2 = torch.einsum("sr,srt,st->s", mu.conj(), cmat, mu).real
    sigma2_wick = torch.clamp(wick1 + 2.0 * wick2, min=tiny)
    wick_cv2 = sigma2_wick / torch.clamp(mu_snr * mu_snr, min=tiny)
    trace = torch.diagonal(cmat, dim1=-2, dim2=-1).sum(dim=-1).real
    trace_c2 = torch.clamp(torch.sum(torch.abs(cmat) ** 2, dim=(1, 2)), min=tiny)
    neff = torch.clamp(trace * trace / trace_c2, min=tiny)
    return {
        "muSNR_max": mu_snr,
        "wickCV2_min": wick_cv2,
        "wickCV2_max": wick_cv2,
        "Neff_min": neff,
        "sigma2Wick": sigma2_wick,
        "wickCV2": wick_cv2,
        "Neff": neff,
    }


@torch.inference_mode()
def optimize_z_batch_one_sweep(
    state: WState,
    initial_z: np.ndarray | torch.Tensor,
    objectives: Sequence[str],
    *,
    parity: bool = False,
) -> Dict[str, Any]:
    """Optimize independent trajectories with exact incremental bit flips."""
    dev = state.UBR.device
    z = torch.as_tensor(initial_z, dtype=torch.long, device=dev).clone()
    if z.ndim != 2 or z.shape[0] != len(objectives):
        raise ValueError("initial_z/objectives shape mismatch.")
    if any(name not in OPTIMIZATION_OBJECTIVES for name in objectives):
        raise ValueError(f"Unknown optimization objective in {list(objectives)}")

    gamma = generate_ris_response_from_z(
        z, device=dev, parity=parity
    )["gamma"]
    n_r = int(state.env.muRU.shape[0])
    k_cov = state.cov_kernel.reshape(
        n_r * n_r, gamma.shape[1], gamma.shape[1]
    )
    kernels = torch.cat([state.eff_moment_kernel, k_cov], dim=0)
    linear = state.ubarBR[:, None] * state.env.muRU.T

    left = torch.einsum("aij,sj->sai", kernels, gamma.conj())
    right = torch.einsum("si,aij->saj", gamma, kernels)
    q = torch.einsum("si,sai->sa", gamma, left)
    mu = gamma @ linear
    metrics = _metrics_from_quadratics(q, mu, n_r)
    current = torch.stack([metrics[name][s] for s, name in enumerate(objectives)])
    initial_objective = current.clone()
    accepted_flips = torch.zeros(len(objectives), dtype=torch.int64, device=dev)
    levels = generate_ris_response_from_z(
        torch.tensor([0, 1], device=dev), device=dev, parity=parity
    )["gamma"]

    for p in range(z.shape[1]):
        new_gamma = torch.where(z[:, p] == 0, levels[1], levels[0])
        delta = new_gamma - gamma[:, p]
        q_trial = (
            q
            + delta[:, None] * left[:, :, p]
            + delta.conj()[:, None] * right[:, :, p]
            + (delta.abs() ** 2)[:, None] * kernels[:, p, p][None, :]
        )
        mu_trial = mu + delta[:, None] * linear[p][None, :]
        trial_metrics = _metrics_from_quadratics(q_trial, mu_trial, n_r)
        trial = torch.stack([
            trial_metrics[name][s] for s, name in enumerate(objectives)
        ])
        tol = torch.maximum(
            torch.full_like(current, 1e-12),
            torch.abs(current) * 1e-10,
        )
        maximize = torch.tensor(
            [name.endswith("_max") for name in objectives],
            dtype=torch.bool,
            device=dev,
        )
        accept = torch.where(
            maximize, trial > current + tol, trial < current - tol
        )
        if not bool(torch.any(accept)):
            continue

        d = delta[accept]
        gamma[accept, p] = new_gamma[accept]
        z[accept, p] ^= 1
        q[accept] = q_trial[accept]
        mu[accept] = mu_trial[accept]
        current[accept] = trial[accept]
        accepted_flips[accept] += 1
        left[accept] += (
            d.conj()[:, None, None] * kernels[:, :, p][None, :, :]
        )
        right[accept] += d[:, None, None] * kernels[:, p, :][None, :, :]

    return {
        "z": z,
        "gamma": gamma,
        "initial_objective": initial_objective,
        "final_objective": current,
        "accepted_flips": accepted_flips,
    }


@torch.inference_mode()
def build_w_specific_z_pool(
    static_env,
    w: torch.Tensor,
    *,
    nris_x: int,
    nris_y: int,
    seed: int,
    sweeps: int = 1,
    parity: bool = False,
) -> Dict[str, Any]:
    """Build [K,512,nRIS] Z/gamma with shared canonical positions."""
    if int(sweeps) != 1:
        raise ValueError("Frozen production design requires exactly one sweep.")
    canonical = build_canonical_z_pool(nris_x, nris_y, seed=seed)
    random_train = np.flatnonzero(
        (canonical["candidate_type"] == "random")
        & canonical["is_canonical_train"]
    )
    order_local = max_min_hamming_order(canonical["z"][random_train])
    diverse_indices = random_train[order_local]

    w = torch.as_tensor(w)
    k_count = int(w.shape[0])
    z_all = []
    gamma_all = []
    metadata = {
        "candidate_type": [], "canonical_split": [],
        "optimization_objective": [], "optimization_seed_rank": [],
        "optimization_sweep_count": [], "is_optimized": [],
        "is_duplicate": [], "optimization_initial_objective": [],
        "optimization_final_objective": [], "optimization_accepted_flips": [],
    }

    primary_seed_z = canonical["z"][diverse_indices[:8]]
    objective_rows = [name for name in OPTIMIZATION_OBJECTIVES for _ in range(8)]
    initial_rows = np.concatenate([primary_seed_z for _ in OPTIMIZATION_OBJECTIVES], axis=0)
    primary_ranks = np.tile(np.arange(1, 9, dtype=np.int64), len(OPTIMIZATION_OBJECTIVES))

    for k in range(k_count):
        state = prepare_w_state(static_env, w[k])
        optimized = optimize_z_batch_one_sweep(
            state, initial_rows, objective_rows, parity=parity
        )
        opt_z = optimized["z"].detach().cpu().numpy().astype(np.uint8)
        init_obj = optimized["initial_objective"].detach().cpu().numpy()
        final_obj = optimized["final_objective"].detach().cpu().numpy()
        flips = optimized["accepted_flips"].detach().cpu().numpy()
        seed_ranks = primary_ranks.copy()
        duplicate = np.zeros(32, dtype=bool)

        seen_opt: set[bytes] = set()
        next_rank = 9
        for slot in range(32):
            while opt_z[slot].tobytes() in seen_opt and next_rank <= len(diverse_indices):
                retry = optimize_z_batch_one_sweep(
                    state,
                    canonical["z"][diverse_indices[next_rank - 1]][None, :],
                    [objective_rows[slot]],
                    parity=parity,
                )
                opt_z[slot] = retry["z"][0].detach().cpu().numpy().astype(np.uint8)
                init_obj[slot] = float(retry["initial_objective"][0].item())
                final_obj[slot] = float(retry["final_objective"][0].item())
                flips[slot] = int(retry["accepted_flips"][0].item())
                seed_ranks[slot] = next_rank
                next_rank += 1
            duplicate[slot] = opt_z[slot].tobytes() in seen_opt
            seen_opt.add(opt_z[slot].tobytes())

        z_bank = np.concatenate([canonical["z"], opt_z], axis=0)
        z_tensor = torch.as_tensor(z_bank, dtype=torch.long, device=w.device)
        gamma_bank = generate_ris_response_from_z(
            z_tensor, device=w.device, parity=parity
        )["gamma"]
        z_all.append(z_tensor)
        gamma_all.append(gamma_bank)

        metadata["candidate_type"].append(np.concatenate([
            canonical["candidate_type"], np.full(32, "optimized", dtype=object)
        ]))
        metadata["canonical_split"].append(np.concatenate([
            canonical["canonical_split"], np.full(32, "optimized", dtype=object)
        ]))
        metadata["optimization_objective"].append(np.concatenate([
            np.full(480, "", dtype=object), np.asarray(objective_rows, dtype=object)
        ]))
        metadata["optimization_seed_rank"].append(np.concatenate([
            np.full(480, -1, dtype=np.int64), seed_ranks
        ]))
        metadata["optimization_sweep_count"].append(np.concatenate([
            np.zeros(480, dtype=np.int64), np.ones(32, dtype=np.int64)
        ]))
        metadata["is_optimized"].append(np.concatenate([
            np.zeros(480, dtype=bool), np.ones(32, dtype=bool)
        ]))
        metadata["is_duplicate"].append(np.concatenate([
            np.zeros(480, dtype=bool), duplicate
        ]))
        metadata["optimization_initial_objective"].append(np.concatenate([
            np.full(480, np.nan), init_obj
        ]))
        metadata["optimization_final_objective"].append(np.concatenate([
            np.full(480, np.nan), final_obj
        ]))
        metadata["optimization_accepted_flips"].append(np.concatenate([
            np.zeros(480, dtype=np.int64), flips
        ]))

    return {
        "z": torch.stack(z_all, dim=0),
        "gamma": torch.stack(gamma_all, dim=0),
        "diverse_seed_indices": diverse_indices,
        **{key: np.stack(value, axis=0) for key, value in metadata.items()},
    }


@torch.inference_mode()
def evaluate_paired_analytic_stats(
    static_env,
    w: torch.Tensor,
    gamma: torch.Tensor,
    *,
    z_chunk: int = 128,
) -> Dict[str, torch.Tensor]:
    """Evaluate analytical physics for matched [W_k, gamma_kc] candidates."""
    if gamma.ndim != 3 or gamma.shape[0] != w.shape[0]:
        raise ValueError("gamma must be [K,C,nRIS] and match W.")
    mu_rows, wick_rows, neff_rows = [], [], []
    for k in range(int(w.shape[0])):
        state = prepare_w_state(static_env, w[k])
        mu_parts, wick_parts, neff_parts = [], [], []
        for c0 in range(0, int(gamma.shape[1]), int(z_chunk)):
            out = evaluate_gamma_batch(
                state, gamma[k, c0:c0 + int(z_chunk)]
            )
            mu_parts.append(out["muSNR"])
            wick_parts.append(out["sigma2Wick"])
            cmat = out["Cmat"]
            tiny = torch.finfo(cmat.real.dtype).tiny
            trace = torch.diagonal(cmat, dim1=-2, dim2=-1).sum(dim=-1).real
            trace_c2 = torch.clamp(
                torch.sum(torch.abs(cmat) ** 2, dim=(1, 2)), min=tiny
            )
            neff_parts.append(torch.clamp(trace * trace / trace_c2, min=tiny))
        mu_rows.append(torch.cat(mu_parts))
        wick_rows.append(torch.cat(wick_parts))
        neff_rows.append(torch.cat(neff_parts))
    mu = torch.stack(mu_rows)
    sigma2_wick = torch.stack(wick_rows)
    neff = torch.stack(neff_rows)
    tiny = torch.finfo(mu.dtype).tiny
    return {
        "muSNR": mu,
        "sigma2Wick": sigma2_wick,
        "wickCV2": sigma2_wick / torch.clamp(mu * mu, min=tiny),
        "Neff": neff,
    }
