"""Unit tests for manifest generation, validation, and split strategies."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from pathoshift.data.manifest import (
    build_manifest,
    compute_sha256,
    extract_group_ids,
    load_manifest,
    validate_manifest,
)
from pathoshift.data.splits import (
    create_patient_aware_splits,
    create_stratified_splits,
)


def test_extract_group_ids():
    p, s = extract_group_ids("patient_042_slide_01_tile_0005.png")
    assert p == "042"
    assert s == "01"

    p2, s2 = extract_group_ids("pt99-sl03-crop.jpg")
    assert p2 == "99"
    assert s2 == "03"


def test_patient_aware_splits():
    records = [
        {"image_id": f"tile_{i}", "patient_id": f"pt_{i // 3}", "label": "class_A"}
        for i in range(30)
    ]
    split_records = create_patient_aware_splits(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Verify no patient appears in both train and val/test
    patients_by_split = {}
    for r in split_records:
        patients_by_split.setdefault(r["split"], set()).add(r["patient_id"])

    assert len(patients_by_split["train"].intersection(patients_by_split["val"])) == 0
    assert len(patients_by_split["train"].intersection(patients_by_split["test"])) == 0
    assert len(patients_by_split["val"].intersection(patients_by_split["test"])) == 0


def test_build_and_validate_manifest(tmp_path: Path):
    data_dir = tmp_path / "raw_data"
    class_a = data_dir / "tumor"
    class_b = data_dir / "normal"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    # Create dummy images
    for i in range(5):
        img_a = Image.fromarray(np.full((32, 32, 3), [180, 80, 140], dtype=np.uint8))
        img_a.save(class_a / f"patient_0{i}_tile_01.png")

        img_b = Image.fromarray(np.full((32, 32, 3), [190, 90, 150], dtype=np.uint8))
        img_b.save(class_b / f"patient_1{i}_tile_01.png")

    manifest_file = tmp_path / "manifest.jsonl"
    summary = build_manifest(
        input_dir=data_dir,
        output_manifest_path=manifest_file,
        split_strategy="patient",
        seed=42,
    )

    assert summary["total_images"] == 10
    assert "tumor" in summary["classes"]
    assert "normal" in summary["classes"]

    validation = validate_manifest(manifest_file, verify_checksum=True)
    assert validation["is_valid"] is True
    assert validation["missing_count"] == 0
    assert validation["corrupted_count"] == 0
