"""
Controlled 4,000-bank environment design for the q05-GG NN dataset.

This module generates only cheap bank/environment metadata. It deliberately
does NOT run rho, channel Monte-Carlo, W selection, RIS candidates, or labels.

Design:
- 4 scenario families x 1,000 banks = 4,000 banks.
- Same scenario family on BR/RU; LOS/NLOS independently varied.
- Exact 2,800 / 600 / 600 train/validation/test_interpolation split.
- Controlled geometry interpolation using 10x10 BR/RU distance cells.
- Every cell has 7 train and 3 non-train points; validation/test distances
  lie strictly inside the train distance envelope of the same cell.
- nT >= 4.
- nRIS in {64,128,256,512}, exactly balanced.
- RIS fixed at [0,0,0] to preserve the current project convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Any
import math

import numpy as np
import pandas as pd
import yaml


LINK_STATES = ("LOS_LOS", "LOS_NLOS", "NLOS_LOS", "NLOS_NLOS")

STATE_TO_SUFFIX = {
    "LOS_LOS": ("LOS", "LOS"),
    "LOS_NLOS": ("LOS", "NLOS"),
    "NLOS_LOS": ("NLOS", "LOS"),
    "NLOS_NLOS": ("NLOS", "NLOS"),
}


@dataclass(frozen=True)
class FamilyGeometrySpec:
    br_min_m: float
    br_max_m: float
    ru_min_m: float
    ru_max_m: float
    gnb_height_min_m: float
    gnb_height_max_m: float
    ue_height_min_m: float
    ue_height_max_m: float


def load_dataset_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping.")
    return cfg


def _family_specs(cfg: Mapping[str, Any]) -> Dict[str, FamilyGeometrySpec]:
    out = {}
    for family, s in cfg["geometry"]["families"].items():
        out[str(family)] = FamilyGeometrySpec(
            br_min_m=float(s["dist_BR_m"][0]),
            br_max_m=float(s["dist_BR_m"][1]),
            ru_min_m=float(s["dist_RU_m"][0]),
            ru_max_m=float(s["dist_RU_m"][1]),
            gnb_height_min_m=float(s["gnb_height_m"][0]),
            gnb_height_max_m=float(s["gnb_height_m"][1]),
            ue_height_min_m=float(s["ue_height_m"][0]),
            ue_height_max_m=float(s["ue_height_m"][1]),
        )
    return out


def _balanced_scalars(n: int, labels: Sequence[Any], rng: np.random.Generator):
    labels = list(labels)
    q, r = divmod(int(n), len(labels))
    vals = []
    for i, x in enumerate(labels):
        vals.extend([x] * (q + (1 if i < r else 0)))
    vals = np.asarray(vals)
    rng.shuffle(vals)
    return vals


def _balanced_shapes(n: int, shapes: Sequence[Sequence[int]], rng: np.random.Generator):
    shapes = [tuple(map(int, s)) for s in shapes]
    q, r = divmod(int(n), len(shapes))
    vals = []
    for i, s in enumerate(shapes):
        vals.extend([s] * (q + (1 if i < r else 0)))
    rng.shuffle(vals)
    return vals


def _balanced_four_state_for_split(
    split_name: str,
    n: int,
    family_index: int,
    rng: np.random.Generator,
):
    if n == 700:
        counts = [175, 175, 175, 175]
    elif n == 150:
        if split_name == "validation":
            extra = {(family_index + 0) % 4, (family_index + 1) % 4}
        elif split_name == "test_interpolation":
            extra = {(family_index + 2) % 4, (family_index + 3) % 4}
        else:
            raise ValueError(split_name)
        counts = [37 + int(i in extra) for i in range(4)]
    else:
        return _balanced_scalars(n, LINK_STATES, rng)

    vals = []
    for state, count in zip(LINK_STATES, counts):
        vals.extend([state] * count)
    rng.shuffle(vals)
    return np.asarray(vals, dtype=object)


def _split_pattern_for_cell(cell_id: int):
    if cell_id % 2 == 0:
        return ["train"] * 7 + ["validation"] * 2 + ["test_interpolation"]
    return ["train"] * 7 + ["validation"] + ["test_interpolation"] * 2


def _distance_offsets_for_cell(cell_id: int):
    train_br = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95])
    train_ru = np.array([0.95, 0.50, 0.20, 0.80, 0.35, 0.65, 0.05])
    train_ru = np.roll(train_ru, cell_id % len(train_ru))

    val_pairs = [(0.275, 0.725), (0.725, 0.275)]
    test_pairs = [(0.425, 0.575), (0.575, 0.425)]

    if cell_id % 2 == 0:
        return {
            "train": list(zip(train_br.tolist(), train_ru.tolist())),
            "validation": val_pairs,
            "test_interpolation": test_pairs[:1],
        }
    return {
        "train": list(zip(train_br.tolist(), train_ru.tolist())),
        "validation": val_pairs[:1],
        "test_interpolation": test_pairs,
    }


def _sample_height(lo, hi, u, distance_m):
    h = float(lo + u * (hi - lo))
    if h >= distance_m:
        h = max(0.0, 0.8 * float(distance_m))
    return h


def _xyz_from_distance_height(distance_m, z_m, azimuth_rad):
    rho = math.sqrt(max(float(distance_m) ** 2 - float(z_m) ** 2, 0.0))
    return (
        rho * math.cos(azimuth_rad),
        rho * math.sin(azimuth_rad),
        float(z_m),
    )


def _canonical_nris_shape(nris: int):
    mapping = {
        64: (8, 4),
        128: (8, 8),
        256: (16, 8),
        512: (16, 16),
    }
    if int(nris) not in mapping:
        raise ValueError(f"Unsupported nRIS={nris}")
    return mapping[int(nris)]


def _make_family_rows(family, family_index, cfg, spec, seed):
    rng = np.random.default_rng(int(seed) + 10_000 * (family_index + 1))
    raw_rows = []

    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase_br = rng.uniform(0.0, 2.0 * math.pi)
    phase_ru = rng.uniform(0.0, 2.0 * math.pi)

    row_in_family = 0
    for br_bin in range(10):
        for ru_bin in range(10):
            cell_id = 10 * br_bin + ru_bin
            split_pattern = _split_pattern_for_cell(cell_id)
            offsets = _distance_offsets_for_cell(cell_id)
            counters = {k: 0 for k in offsets}

            for split in split_pattern:
                j = counters[split]
                off_br, off_ru = offsets[split][j]
                counters[split] += 1

                u_br = (br_bin + off_br) / 10.0
                u_ru = (ru_bin + off_ru) / 10.0

                d_br = spec.br_min_m + u_br * (spec.br_max_m - spec.br_min_m)
                d_ru = spec.ru_min_m + u_ru * (spec.ru_max_m - spec.ru_min_m)

                h_u_br = ((row_in_family * 0.6180339887498949) + 0.17 * family_index) % 1.0
                h_u_ru = ((row_in_family * 0.4142135623730950) + 0.29 * family_index) % 1.0
                gnb_z = _sample_height(spec.gnb_height_min_m, spec.gnb_height_max_m, h_u_br, d_br)
                ue_z = _sample_height(spec.ue_height_min_m, spec.ue_height_max_m, h_u_ru, d_ru)

                split_phase = {
                    "train": 0.0,
                    "validation": 0.37,
                    "test_interpolation": 0.73,
                }[split]
                az_br = (phase_br + row_in_family * golden + split_phase) % (2.0 * math.pi)
                az_ru = (phase_ru - row_in_family * golden * 0.754877666 + 1.7 * split_phase) % (2.0 * math.pi)

                gnb = _xyz_from_distance_height(d_br, gnb_z, az_br)
                ue = _xyz_from_distance_height(d_ru, ue_z, az_ru)

                raw_rows.append({
                    "family": family,
                    "split": split,
                    "splitID": split,
                    "geometry_cell_id": cell_id,
                    "br_distance_bin": br_bin,
                    "ru_distance_bin": ru_bin,
                    "u_BR": u_br,
                    "u_RU": u_ru,
                    "ris_x": 0.0,
                    "ris_y": 0.0,
                    "ris_z": 0.0,
                    "gnb_x": gnb[0],
                    "gnb_y": gnb[1],
                    "gnb_z": gnb[2],
                    "ue_x": ue[0],
                    "ue_y": ue[1],
                    "ue_z": ue[2],
                    "dist_BR": d_br,
                    "dist_RU": d_ru,
                    "eta": d_br / (d_br + d_ru),
                })
                row_in_family += 1

    df = pd.DataFrame(raw_rows)
    if len(df) != 1000:
        raise RuntimeError("Family design must have exactly 1000 rows.")

    # Link states: exact 250/state/family, while train has exact 175/state.
    states = np.empty(len(df), dtype=object)
    for split in ("train", "validation", "test_interpolation"):
        idx = np.flatnonzero(df["split"].to_numpy() == split)
        vals = _balanced_four_state_for_split(split, len(idx), family_index, rng)
        states[idx] = vals
    df["link_state"] = states

    suffixes = [STATE_TO_SUFFIX[s] for s in df["link_state"]]
    df["scenario_BR"] = [f"{family}-{x[0]}" for x in suffixes]
    df["scenario_RU"] = [f"{family}-{x[1]}" for x in suffixes]

    # Exact 250 of each nRIS value in every family.
    nris_vals = [int(x) for x in cfg["arrays"]["nRIS_values"]]
    nris = _balanced_scalars(len(df), nris_vals, rng).astype(int)
    df["nRIS"] = nris
    shapes = np.asarray([_canonical_nris_shape(x) for x in nris], dtype=int)
    df["nRIS_x"] = shapes[:, 0]
    df["nRIS_y"] = shapes[:, 1]
    df["nRIS1"] = df["nRIS_x"]
    df["nRIS2"] = df["nRIS_y"]

    tx_shapes = _balanced_shapes(len(df), cfg["arrays"]["tx_spatial_shapes"], rng)
    rx_shapes = _balanced_shapes(len(df), cfg["arrays"]["rx_spatial_shapes"], rng)
    tx = np.asarray(tx_shapes, dtype=int)
    rx = np.asarray(rx_shapes, dtype=int)

    df["nT1"] = tx[:, 0]
    df["nT2"] = tx[:, 1]
    df["nT"] = 2 * df["nT1"] * df["nT2"]
    df["nR1"] = rx[:, 0]
    df["nR2"] = rx[:, 1]
    df["nR"] = 2 * df["nR1"] * df["nR2"]

    freqs = [float(x) for x in cfg["carrier_frequencies_hz"]]
    df["fc"] = _balanced_scalars(len(df), freqs, rng).astype(float)

    return df


def generate_environment_dataframe(cfg: Mapping[str, Any]) -> pd.DataFrame:
    seed = int(cfg["seed"])
    specs = _family_specs(cfg)
    frames = []

    for family_index, (family, requested) in enumerate(
        cfg["environments"]["scenario_families"].items()
    ):
        if int(requested) != 1000:
            raise ValueError("Current controlled design requires exactly 1000 banks/family.")
        frames.append(
            _make_family_rows(
                str(family), family_index, cfg, specs[str(family)], seed
            )
        )

    df = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(seed + 999_983)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    df.insert(0, "bankID", np.arange(1, len(df) + 1, dtype=np.int64))
    df.insert(1, "index", np.arange(len(df), dtype=np.int64))

    base_seed = int(cfg["rng"]["channel_seed_base"])
    df["channel_seed"] = base_seed + df["bankID"].astype(np.int64)

    ris_rng = np.random.default_rng(int(cfg["rng"]["ris_seed"]))
    df["ris_seed"] = ris_rng.integers(
        1, np.iinfo(np.uint32).max, size=len(df), dtype=np.uint32
    ).astype(np.uint64)

    cols = [
        "bankID", "index", "split", "splitID",
        "family", "link_state",
        "fc", "scenario_BR", "scenario_RU",
        "ris_x", "ris_y", "ris_z",
        "gnb_x", "gnb_y", "gnb_z",
        "ue_x", "ue_y", "ue_z",
        "dist_BR", "dist_RU", "eta",
        "geometry_cell_id", "br_distance_bin", "ru_distance_bin",
        "u_BR", "u_RU",
        "nT1", "nT2", "nT",
        "nR1", "nR2", "nR",
        "nRIS_x", "nRIS_y", "nRIS1", "nRIS2", "nRIS",
        "channel_seed", "ris_seed",
    ]
    df = df[cols]
    validate_environment_dataframe(df, cfg, strict=True)
    return df


def _scenario_family_from_name(s: pd.Series):
    return s.astype(str).str.replace(r"-(LOS|NLOS)$", "", regex=True)


def validate_environment_dataframe(df, cfg, strict=True):
    errors = []

    expected_rows = int(cfg["environments"]["total_banks"])
    if len(df) != expected_rows:
        errors.append(f"rows={len(df)} expected={expected_rows}")

    expected_split = {str(k): int(v) for k, v in cfg["environments"]["split"].items()}
    got_split = df["split"].value_counts().to_dict()
    if got_split != expected_split:
        errors.append(f"split counts {got_split} != {expected_split}")

    expected_family = {
        str(k): int(v) for k, v in cfg["environments"]["scenario_families"].items()
    }
    got_family = df["family"].value_counts().to_dict()
    if got_family != expected_family:
        errors.append(f"family counts {got_family} != {expected_family}")

    if not (_scenario_family_from_name(df["scenario_BR"]) == df["family"]).all():
        errors.append("scenario_BR family mismatch")
    if not (_scenario_family_from_name(df["scenario_RU"]) == df["family"]).all():
        errors.append("scenario_RU family mismatch")

    for family, g in df.groupby("family"):
        vc = g["link_state"].value_counts().to_dict()
        for state in LINK_STATES:
            if vc.get(state, 0) != 250:
                errors.append(f"{family}: {state} count={vc.get(state,0)} expected=250")

    expected_nris = {int(x): 1000 for x in cfg["arrays"]["nRIS_values"]}
    got_nris = df["nRIS"].value_counts().sort_index().to_dict()
    if got_nris != expected_nris:
        errors.append(f"nRIS counts {got_nris} != {expected_nris}")

    if int(df["nT"].min()) < int(cfg["arrays"]["nT_min"]):
        errors.append("nT_min violated")

    br = np.sqrt(
        (df["gnb_x"] - df["ris_x"]) ** 2
        + (df["gnb_y"] - df["ris_y"]) ** 2
        + (df["gnb_z"] - df["ris_z"]) ** 2
    )
    ru = np.sqrt(
        (df["ue_x"] - df["ris_x"]) ** 2
        + (df["ue_y"] - df["ris_y"]) ** 2
        + (df["ue_z"] - df["ris_z"]) ** 2
    )
    if not np.allclose(br, df["dist_BR"], rtol=0, atol=2e-8):
        errors.append("stored dist_BR mismatch")
    if not np.allclose(ru, df["dist_RU"], rtol=0, atol=2e-8):
        errors.append("stored dist_RU mismatch")

    specs = _family_specs(cfg)
    for family, g in df.groupby("family"):
        s = specs[family]
        if not g["dist_BR"].between(s.br_min_m, s.br_max_m).all():
            errors.append(f"{family}: BR distance out of range")
        if not g["dist_RU"].between(s.ru_min_m, s.ru_max_m).all():
            errors.append(f"{family}: RU distance out of range")

    coord_cols = [
        "ris_x", "ris_y", "ris_z",
        "gnb_x", "gnb_y", "gnb_z",
        "ue_x", "ue_y", "ue_z",
    ]
    keys = df[coord_cols].round(9).astype(str).agg("|".join, axis=1)
    tmp = pd.DataFrame({"key": keys, "split": df["split"]})
    exact_overlap = int((tmp.groupby("key")["split"].nunique() > 1).sum())
    if exact_overlap:
        errors.append(f"exact geometry overlap across splits={exact_overlap}")

    train = df[df["split"] == "train"]
    env = (
        train.groupby(["family", "geometry_cell_id"])
        .agg(
            br_min=("dist_BR", "min"),
            br_max=("dist_BR", "max"),
            ru_min=("dist_RU", "min"),
            ru_max=("dist_RU", "max"),
        )
    )
    interp_fail = 0
    missing_cell = 0
    for row in df[df["split"] != "train"].itertuples(index=False):
        key = (row.family, row.geometry_cell_id)
        if key not in env.index:
            missing_cell += 1
            continue
        e = env.loc[key]
        ok = (
            e.br_min < row.dist_BR < e.br_max
            and e.ru_min < row.dist_RU < e.ru_max
        )
        interp_fail += int(not ok)

    if missing_cell:
        errors.append(f"missing train geometry cells={missing_cell}")
    if interp_fail:
        errors.append(f"controlled interpolation failures={interp_fail}")

    report = {
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in got_split.items()},
        "family_counts": {str(k): int(v) for k, v in got_family.items()},
        "nRIS_counts": {str(int(k)): int(v) for k, v in got_nris.items()},
        "nT_min": int(df["nT"].min()),
        "exact_geometry_overlap_across_splits": exact_overlap,
        "interpolation_failures": int(interp_fail),
        "missing_train_geometry_cells": int(missing_cell),
        "errors": errors,
        "pass": not errors,
    }

    if strict and errors:
        raise ValueError("Environment dataset validation failed:\n- " + "\n- ".join(errors))
    return report


def distance_summary(df):
    return (
        df.groupby("family")
        .agg(
            banks=("bankID", "size"),
            BR_min=("dist_BR", "min"),
            BR_median=("dist_BR", "median"),
            BR_max=("dist_BR", "max"),
            RU_min=("dist_RU", "min"),
            RU_median=("dist_RU", "median"),
            RU_max=("dist_RU", "max"),
            eta_min=("eta", "min"),
            eta_median=("eta", "median"),
            eta_max=("eta", "max"),
        )
        .reset_index()
    )


def eta_bin_summary(df):
    bins = [0.0, 0.35, 0.50, 0.65, 1.0000001]
    labels = ["RIS_near_gNB", "middle_gNB_side", "middle_UE_side", "RIS_near_UE"]
    x = df.copy()
    x["eta_region"] = pd.cut(x["eta"], bins=bins, labels=labels, right=False)
    return (
        x.groupby(["family", "eta_region"], observed=False)
        .size()
        .rename("banks")
        .reset_index()
    )
