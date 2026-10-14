"""Dataset reading, validation, and reproducible JSONL manifest generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pathoshift.data.splits import (
    create_patient_aware_splits,
    create_stratified_splits,
)
from pathoshift.preprocessing.quality import evaluate_tile_quality

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_group_ids(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract patient and slide identifiers if encoded in file name."""
    stem = Path(filename).stem
    patient_id = None
    slide_id = None

    # Common patterns: patient_001, slide_02, pt123, s01, etc.
    p_match = re.search(r"(?:patient|pt)[-_]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)
    if p_match:
        patient_id = p_match.group(1)

    s_match = re.search(r"(?:slide|sl)[-_]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)
    if s_match:
        slide_id = s_match.group(1)

    return patient_id, slide_id


def inspect_image_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read image metadata and compute preliminary quality metrics."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode

        # Read RGB with OpenCV for fast quality metric calculation
        bgr = cv2.imread(str(file_path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        quality = evaluate_tile_quality(rgb)

        file_size = file_path.stat().st_size
        sha256 = compute_sha256(file_path)

        patient_id, slide_id = extract_group_ids(file_path.name)

        return {
            "width": width,
            "height": height,
            "channels": 3 if mode == "RGB" else (len(mode) if mode else 3),
            "file_size": file_size,
            "sha256": sha256,
            "patient_id": patient_id,
            "slide_id": slide_id,
            "tissue_ratio": round(quality["tissue_ratio"], 4),
            "focus_score": round(quality["focus_score"], 2),
            "is_valid_tissue": quality["is_valid_tissue"],
            "is_accepted": quality["is_accepted"],
        }
    except Exception:
        return None


def scan_directory(input_dir: Path | str) -> List[Dict[str, Any]]:
    """Recursively discover and validate images under an input directory."""
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    image_files: List[Path] = [
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    records: List[Dict[str, Any]] = []

    for img_path in tqdm(image_files, desc="Scanning & validating images"):
        meta = inspect_image_file(img_path)
        if meta is None:
            continue

        # Infer class label: immediate parent directory name if not input root
        rel_parent = img_path.parent.relative_to(input_path)
        label = rel_parent.parts[0] if len(rel_parent.parts) > 0 else "unknown"

        record = {
            "image_id": img_path.stem,
            "file_name": img_path.name,
            "relative_path": str(img_path.relative_to(input_path)),
            "absolute_path": str(img_path),
            "label": label,
            **meta,
        }
        records.append(record)

    return records


def build_manifest(
    input_dir: Path | str,
    output_manifest_path: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_strategy: str = "stratified",
    seed: int = 42,
) -> Dict[str, Any]:
    """Scan input dataset, assign splits, and write reproducible JSONL manifest."""
    records = scan_directory(input_dir)
    if not records:
        raise ValueError(f"No valid histopathology images found under {input_dir}")

    # Check if patient or slide metadata exists for group split
    has_patient_ids = any(r.get("patient_id") is not None for r in records)
    has_slide_ids = any(r.get("slide_id") is not None for r in records)

    if split_strategy == "patient" and (has_patient_ids or has_slide_ids):
        records = create_patient_aware_splits(
            records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        records = create_stratified_splits(
            records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Compute statistics summary
    labels = [r["label"] for r in records]
    splits = [r["split"] for r in records]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    unique_splits, split_counts = np.unique(splits, return_counts=True)

    summary = {
        "manifest_path": str(out_path),
        "total_images": len(records),
        "classes": {str(k): int(v) for k, v in zip(unique_labels, label_counts)},
        "splits": {str(k): int(v) for k, v in zip(unique_splits, split_counts)},
        "accepted_tiles": sum(1 for r in records if r.get("is_accepted", True)),
    }

    return summary


def load_manifest(manifest_path: Path | str) -> List[Dict[str, Any]]:
    """Load JSONL manifest into a list of dictionaries."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_manifest(manifest_path: Path | str, verify_checksum: bool = True) -> Dict[str, Any]:
    """Validate all images listed in a manifest file exist and optionally verify SHA-256."""
    records = load_manifest(manifest_path)
    missing = []
    corrupted = []

    manifest_p = Path(manifest_path)
    for r in records:
        p = Path(r.get("absolute_path", ""))
        if not p.exists():
            rel = r.get("relative_path")
            if rel:
                for cand in [manifest_p.parent / rel, Path.cwd() / "data" / rel, Path.cwd() / rel]:
                    if cand.exists():
                        p = cand
                        break

        if not p.exists() or p.stat().st_size == 0:
            missing.append(r["image_id"])
            continue

        if verify_checksum and "sha256" in r:
            actual_hash = compute_sha256(p)
            if actual_hash != r["sha256"]:
                corrupted.append(r["image_id"])

    return {
        "total_records": len(records),
        "missing_count": len(missing),
        "corrupted_count": len(corrupted),
        "missing_ids": missing[:10],
        "corrupted_ids": corrupted[:10],
        "is_valid": len(missing) == 0 and len(corrupted) == 0,
    }
