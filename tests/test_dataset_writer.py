import json
from pathlib import Path

import pandas as pd
import pytest

from ris_env.dataset_writer import (
    plan_shards,
    stable_json_hash,
    load_or_create_manifest,
    summarize_manifest,
)


def _env():
    return pd.DataFrame({
        "bankID": [4,1,2,3,6,5],
        "splitID": [
            "validation","train","train",
            "validation","test_interpolation","test_interpolation"
        ],
    })


def test_plan_shards_is_split_scoped_and_deterministic():
    p = plan_shards(_env(), banks_per_shard=2)
    got = [(x["split"], x["bank_ids"], x["filename"]) for x in p]
    assert got == [
        ("train", [1,2], "shard_0000.parquet"),
        ("validation", [3,4], "shard_0000.parquet"),
        ("test_interpolation", [5,6], "shard_0000.parquet"),
    ]
    assert all(x["expected_rows"] == 2*32*512 for x in p)


def test_manifest_refuses_signature_change(tmp_path):
    spec1 = {"signature": stable_json_hash({"a": 1})}
    spec2 = {"signature": stable_json_hash({"a": 2})}
    m = load_or_create_manifest(tmp_path, generation_spec=spec1, total_banks=10)
    assert summarize_manifest(m)["remaining_banks"] == 10

    with pytest.raises(RuntimeError):
        load_or_create_manifest(tmp_path, generation_spec=spec2, total_banks=10)
