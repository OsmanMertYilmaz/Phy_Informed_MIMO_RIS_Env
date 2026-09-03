from pathlib import Path
from ris_env.environment_dataset import (
    load_dataset_config, generate_environment_dataframe,
    validate_environment_dataframe
)

def test_controlled_4000_environment_design():
    root = Path(__file__).resolve().parents[1]
    cfg = load_dataset_config(root / "configs/nn_dataset_4000.yaml")
    df = generate_environment_dataframe(cfg)

    assert len(df) == 4000
    assert df["split"].value_counts().to_dict() == {
        "train": 2800,
        "validation": 600,
        "test_interpolation": 600,
    }
    assert df["family"].value_counts().to_dict() == {
        "Indoor-Office": 1000,
        "UMi": 1000,
        "UMa": 1000,
        "RMa": 1000,
    }
    assert df["nRIS"].value_counts().sort_index().to_dict() == {
        64: 1000, 128: 1000, 256: 1000, 512: 1000,
    }
    assert int(df["nT"].min()) >= 4

    report = validate_environment_dataframe(df, cfg, strict=False)
    assert report["pass"], report["errors"]
    assert report["exact_geometry_overlap_across_splits"] == 0
    assert report["interpolation_failures"] == 0


def test_controlled_27200_environment_design():
    root = Path(__file__).resolve().parents[1]
    cfg = load_dataset_config(root / "configs/variance_ratio_dataset_27200.yaml")
    df = generate_environment_dataframe(cfg)

    assert len(df) == 27200
    assert df["split"].value_counts().to_dict() == {
        "train": 19040,
        "validation": 4080,
        "final_test": 4080,
    }
    assert df["family"].value_counts().to_dict() == {
        "Indoor-Office": 6800,
        "UMi": 6800,
        "UMa": 6800,
        "RMa": 6800,
    }
    assert df["nRIS"].value_counts().sort_index().to_dict() == {
        64: 6800, 128: 6800, 256: 6800, 512: 6800,
    }

    report = validate_environment_dataframe(df, cfg, strict=False)
    assert report["pass"], report["errors"]
    assert report["exact_geometry_overlap_across_splits"] == 0
    assert report["interpolation_failures"] == 0
