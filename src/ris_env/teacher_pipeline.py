"""
Production teacher pipeline for the q05-GG neural-network dataset.

One bank:
    raw environment row
      -> deterministic geometry/LSP/rho/static environment
      -> 32 unique Type-I rank-1 W candidates
      -> 512 RIS candidates
      -> 64k native-CUDA Monte-Carlo shared across candidates
      -> empirical varEmp
      -> analytic muSNR
      -> symmetric Gamma-Gamma q05

The 512 RIS pool is fixed to the dataset specification:
    4 anchors
    256 global Bernoulli(0.5)
    96 density-stratified
    64 structured 2-D
    92 local perturbations around four high-analytic-mean seeds
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ris_env.environment import build_deterministic_bank
from ris_env.gamma_gamma import GGQ05Lookup
from ris_env.ris_response import generate_ris_response_from_z
from ris_env.label_engine import (
    row_to_bank_input,
    build_w_candidate_pool,
    analytic_mu_snr_multi_w_z,
    run_symmetric_gg_label_engine,
    flatten_label_result,
)


Z_TYPE_COUNTS = {
    "anchor": 4,
    "global_random": 256,
    "density_stratified": 96,
    "structured": 64,
    "local_perturbation": 92,
}


def _device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_packaged_gg_lookup() -> GGQ05Lookup:
    """Load the frozen project q05 lookup shipped inside the Python package."""
    asset = resources.files("ris_env").joinpath("assets/gg_q05_lookup.npz")
    with resources.as_file(asset) as p:
        x = np.load(p)
        return GGQ05Lookup(
            log_cv2=np.asarray(x["log_cv2"], dtype=np.float64),
            qnorm=np.asarray(x["qnorm"], dtype=np.float64),
        )


def _scalar(x):
    if torch.is_tensor(x):
        return x.reshape(()).detach().cpu().item()
    return np.asarray(x).reshape(()).item()


def enrich_environment_row_with_lsp(
    row: pd.Series,
    prepared: Dict[str, Any],
) -> pd.Series:
    """
    Convert the new raw environments_4000 row into the legacy physical-row
    interface expected by the already validated stochastic label engine.

    No empirical quantity is introduced here. All added fields come from the
    deterministic geometry/LSP model.
    """
    out = row.copy()

    # Compatibility aliases expected by label_engine / validation.
    out["ch_seed"] = int(
        row["channel_seed"] if "channel_seed" in row.index else row["ch_seed"]
    )

    if "nRIS1" not in out.index:
        out["nRIS1"] = int(row["nRIS_x"])
    if "nRIS2" not in out.index:
        out["nRIS2"] = int(row["nRIS_y"])

    for prefix, lsp in (("BR", prepared["lsp_br"]), ("RU", prepared["lsp_ru"])):
        out[f"K_{prefix}"] = float(_scalar(lsp.K_linear))
        out[f"M_{prefix}"] = int(_scalar(lsp.M))
        out[f"L_{prefix}"] = int(_scalar(lsp.L))
        out[f"isLOS_{prefix}"] = int(bool(_scalar(lsp.isLOS)))

        for name in (
            "c_ASA", "c_ZSA", "c_ASD", "c_ZSD",
            "mu_XPR", "sigma_XPR",
            "mu_ASA", "sigma_ASA",
            "mu_ZSA", "sigma_ZSA",
            "mu_ASD", "sigma_ASD",
            "mu_ZSD", "sigma_ZSD",
        ):
            out[f"{name}_{prefix}"] = float(_scalar(getattr(lsp, name)))

        out[f"ZODoffset_{prefix}"] = float(_scalar(lsp.mu_offset_ZOD))

    return out


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
        (1 - (idx % 2)).astype(np.uint8),
    ]


def _structured_candidates(nris_x: int, nris_y: int) -> Iterable[np.ndarray]:
    """
    Infinite deterministic stream of 2-D dual-pol structured patterns.

    Array flattening follows the project convention: spatial locations of the
    first polarization block, followed by the second polarization block.
    """
    M, N = int(nris_x), int(nris_y)
    rr, cc = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    rr = rr.reshape(-1, order="F")
    cc = cc.reshape(-1, order="F")

    # Large deterministic family of stripe/block/lattice masks.
    for period in range(2, 18):
        for a, b in (
            (1,0),(0,1),(1,1),(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)
        ):
            for offset in range(period):
                residue = (a * rr + b * cc + offset) % period
                for width in range(1, period):
                    spatial = (residue < width).astype(np.uint8)

                    # Same polarization mask.
                    yield np.concatenate([spatial, spatial])

                    # Opposite mask on second polarization.
                    yield np.concatenate([spatial, 1 - spatial])

                    # Polarization-selective variants.
                    yield np.concatenate([spatial, np.zeros_like(spatial)])
                    yield np.concatenate([np.zeros_like(spatial), spatial])


def build_base_z_pool(
    nris_x: int,
    nris_y: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the first 420 patterns:
        4 + 256 + 96 + 64
    """
    n_ris = 2 * int(nris_x) * int(nris_y)
    rng = np.random.default_rng(int(seed))

    store: List[np.ndarray] = []
    labels: List[str] = []
    seen: set[bytes] = set()

    # 4 anchors
    for z in _anchor_patterns(n_ris):
        _add_unique(store, labels, seen, z, "anchor")

    # 256 global Bernoulli(0.5)
    while labels.count("global_random") < 256:
        z = rng.integers(0, 2, size=n_ris, dtype=np.uint8)
        _add_unique(store, labels, seen, z, "global_random")

    # 96 density-stratified: 8 densities x 12 patterns.
    for ratio in (0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9):
        made = 0
        ones = int(round(ratio * n_ris))
        ones = max(1, min(n_ris - 1, ones))
        while made < 12:
            z = np.zeros(n_ris, dtype=np.uint8)
            pos = rng.choice(n_ris, size=ones, replace=False)
            z[pos] = 1
            if _add_unique(store, labels, seen, z, "density_stratified"):
                made += 1

    # 64 genuine 2-D structured patterns.
    made = 0
    for z in _structured_candidates(nris_x, nris_y):
        if _add_unique(store, labels, seen, z, "structured"):
            made += 1
            if made == 64:
                break
    if made != 64:
        raise RuntimeError(f"Could only create {made}/64 unique structured patterns.")

    z = np.stack(store, axis=0)
    t = np.asarray(labels, dtype=object)

    expected = 4 + 256 + 96 + 64
    if z.shape != (expected, n_ris):
        raise RuntimeError(f"Base Z pool shape {z.shape} != {(expected,n_ris)}")

    return z, t


def add_local_perturbations(
    base_z: np.ndarray,
    base_types: np.ndarray,
    *,
    seed_indices: Sequence[int],
    seed: int,
    hamming_sizes: Sequence[int]=(1,2,4,8),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add exactly 23 unique local neighbors around each of four seeds.

    Returns:
        z_final       [512,nRIS]
        type_final    [512]
        anchor_index  [512], -1 except local rows where it stores base seed idx
    """
    z0 = np.asarray(base_z, dtype=np.uint8)
    n_base, n_ris = z0.shape
    if n_base != 420:
        raise ValueError(f"Expected 420 base patterns, got {n_base}")
    if len(seed_indices) != 4 or len(set(map(int, seed_indices))) != 4:
        raise ValueError("seed_indices must contain four distinct base indices.")

    rng = np.random.default_rng(int(seed))
    store = [x.copy() for x in z0]
    labels = [str(x) for x in base_types.tolist()]
    anchors = [-1] * n_base
    seen = {x.tobytes() for x in store}

    for seed_idx in map(int, seed_indices):
        seed_z = z0[seed_idx]
        made = 0
        attempts = 0
        while made < 23:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError("Could not create unique local perturbations.")
            h = int(hamming_sizes[made % len(hamming_sizes)])
            h = min(h, n_ris)
            pos = rng.choice(n_ris, size=h, replace=False)
            z = seed_z.copy()
            z[pos] ^= 1
            key = z.tobytes()
            if key in seen:
                continue
            seen.add(key)
            store.append(z)
            labels.append("local_perturbation")
            anchors.append(seed_idx)
            made += 1

    zf = np.stack(store, axis=0)
    tf = np.asarray(labels, dtype=object)
    af = np.asarray(anchors, dtype=np.int64)

    if zf.shape[0] != 512:
        raise RuntimeError(f"Final Z pool has {zf.shape[0]} rows, expected 512.")
    if len({x.tobytes() for x in zf}) != 512:
        raise RuntimeError("Final Z pool contains duplicates.")

    return zf, tf, af


@torch.inference_mode()
def build_final_z_pool(
    static_env,
    W: torch.Tensor,
    *,
    nris_x: int,
    nris_y: int,
    seed: int,
    device=None,
    parity: bool=False,
    z_chunk: int=64,
) -> Dict[str, Any]:
    """
    Build the final 512-pattern pool.

    The four local-search seeds are the four base patterns with highest
    max_W analytic muSNR. This is cheap, deterministic, and deployable.
    """
    dev = _device(device)

    base_z, base_types = build_base_z_pool(
        nris_x, nris_y, seed=int(seed)
    )
    base_t = torch.as_tensor(base_z, dtype=torch.long, device=dev)
    base_gamma = generate_ris_response_from_z(
        base_t, device=dev, parity=parity
    )["gamma"]

    score_matrix = analytic_mu_snr_multi_w_z(
        static_env, W, base_gamma, z_chunk=z_chunk
    )
    score = torch.max(score_matrix, dim=0).values

    # Stable descending ranking.
    seed_indices = torch.argsort(score, descending=True, stable=True)[:4]
    seed_indices_np = seed_indices.detach().cpu().numpy().astype(np.int64)

    final_z, final_types, anchor_idx = add_local_perturbations(
        base_z,
        base_types,
        seed_indices=seed_indices_np.tolist(),
        seed=int(seed) + 97_531,
    )

    zt = torch.as_tensor(final_z, dtype=torch.long, device=dev)
    gamma = generate_ris_response_from_z(
        zt, device=dev, parity=parity
    )["gamma"]

    counts = {
        k: int(np.sum(final_types == k))
        for k in Z_TYPE_COUNTS
    }
    if counts != Z_TYPE_COUNTS:
        raise RuntimeError(f"Z type count mismatch: {counts} != {Z_TYPE_COUNTS}")

    return {
        "z": zt,
        "gamma": gamma,
        "candidate_type": final_types,
        "anchor_index": anchor_idx,
        "seed_indices": seed_indices_np,
        "base_analytic_score": score.detach().cpu().numpy(),
        "type_counts": counts,
    }


@dataclass
class TeacherBankPrepared:
    row_raw: pd.Series
    row_physical: pd.Series
    static_env: Any
    W: torch.Tensor
    WIdx: torch.Tensor
    z: torch.Tensor
    gamma: torch.Tensor
    z_candidate_type: np.ndarray
    z_anchor_index: np.ndarray
    z_seed_indices: np.ndarray
    deterministic_timings: Dict[str, float]


@torch.inference_mode()
def prepare_teacher_bank(
    row: pd.Series,
    *,
    k_w: int=32,
    device=None,
    parity: bool=False,
    z_chunk: int=64,
) -> TeacherBankPrepared:
    """
    Prepare one raw environment row for teacher labeling.
    """
    dev = _device(device)

    bank_input = row_to_bank_input(row)

    # Reuse the MATLAB-parity-validated Stage-8 deterministic builder.
    # It also constructs one dummy W state; we discard that state immediately.
    prepared = build_deterministic_bank(
        bank_input,
        device=dev,
        parity=parity,
    )

    row_physical = enrich_environment_row_with_lsp(row, prepared)
    static_env = prepared["static_env"]
    timings = dict(prepared["timings"])

    # Release the large dummy full-W state before the Monte-Carlo engine.
    prepared.pop("w_state", None)
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    W, WIdx = build_w_candidate_pool(
        row_physical,
        k_w=k_w,
        device=dev,
        parity=parity,
    )

    zpack = build_final_z_pool(
        static_env,
        W,
        nris_x=int(row["nRIS1"] if "nRIS1" in row.index else row["nRIS_x"]),
        nris_y=int(row["nRIS2"] if "nRIS2" in row.index else row["nRIS_y"]),
        seed=int(row["ris_seed"]),
        device=dev,
        parity=parity,
        z_chunk=z_chunk,
    )

    return TeacherBankPrepared(
        row_raw=row.copy(),
        row_physical=row_physical,
        static_env=static_env,
        W=W,
        WIdx=WIdx,
        z=zpack["z"],
        gamma=zpack["gamma"],
        z_candidate_type=zpack["candidate_type"],
        z_anchor_index=zpack["anchor_index"],
        z_seed_indices=zpack["seed_indices"],
        deterministic_timings=timings,
    )


@torch.inference_mode()
def run_teacher_bank(
    prepared: TeacherBankPrepared,
    *,
    lookup: Optional[GGQ05Lookup]=None,
    n_mc: int=64_000,
    mc_chunk: int=256,
    w_chunk: int=4,
    z_chunk: int=64,
    device=None,
    parity: bool=False,
) -> Dict[str, Any]:
    """
    Generate all 32x512 teacher labels for one prepared bank.
    """
    if lookup is None:
        lookup = load_packaged_gg_lookup()

    result = run_symmetric_gg_label_engine(
        prepared.row_physical,
        prepared.static_env,
        prepared.W,
        prepared.gamma,
        lookup,
        n_mc=n_mc,
        mc_chunk=mc_chunk,
        w_chunk=w_chunk,
        z_chunk=z_chunk,
        device=device,
        parity=parity,
    )

    labels = flatten_label_result(
        result,
        prepared.WIdx,
        prepared.z,
    )

    # zCandidate is shared across W; map candidate metadata by index.
    zc = labels["zCandidate"].to_numpy(dtype=np.int64)
    labels["candidateType"] = prepared.z_candidate_type[zc]
    labels["anchorIndex"] = prepared.z_anchor_index[zc]

    raw = prepared.row_raw
    prefix = {
        "bankID": int(raw.bankID),
        "splitID": str(raw.splitID if "splitID" in raw.index else raw.split),
        "scenario_BR": str(raw.scenario_BR),
        "scenario_RU": str(raw.scenario_RU),
        "fc": float(raw.fc),
        "nT1": int(raw.nT1),
        "nT2": int(raw.nT2),
        "nT": int(raw.nT),
        "nR1": int(raw.nR1),
        "nR2": int(raw.nR2),
        "nR": int(raw.nR),
        "nRIS1": int(raw.nRIS1),
        "nRIS2": int(raw.nRIS2),
        "nRIS": int(raw.nRIS),
        "N_MC": int(n_mc),
    }
    for key, value in reversed(list(prefix.items())):
        labels.insert(0, key, value)

    # Diagnostics useful for smoke tests and later shard manifests.
    mu = result["muSNR"].detach().cpu().numpy()
    mean_emp = result["meanEmp"].detach().cpu().numpy()
    mean_rel = np.abs(mean_emp - mu) / np.maximum(
        np.abs(mu), np.finfo(np.float64).tiny
    )

    result["labels"] = labels
    result["mean_mdape_pct"] = float(100.0 * np.median(mean_rel))
    result["mean_p90ape_pct"] = float(100.0 * np.quantile(mean_rel, 0.90))
    result["z_type_counts"] = {
        k: int(np.sum(prepared.z_candidate_type == k))
        for k in Z_TYPE_COUNTS
    }
    result["w_unique"] = int(
        np.unique(prepared.WIdx.detach().cpu().numpy(), axis=0).shape[0]
    )
    result["z_unique"] = int(
        len({x.tobytes() for x in prepared.z.detach().cpu().numpy().astype(np.uint8)})
    )
    result["deterministic_timings"] = prepared.deterministic_timings
    return result
