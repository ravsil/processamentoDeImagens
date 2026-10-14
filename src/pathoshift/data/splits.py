"""Split strategies ensuring no data leakage between train, validation, and test partitions."""

from __future__ import annotations

import random
from typing import Dict, List, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def create_patient_aware_splits(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """Assign splits at the patient or slide level to prevent tile-level data leakage.

    All tiles belonging to a given patient/slide are assigned to the same partition.
    """
    rng = random.Random(seed)

    # Group records by patient_id or slide_id
    group_map: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        group_id = rec.get("patient_id") or rec.get("slide_id") or f"sample_{idx}"
        group_map.setdefault(group_id, []).append(idx)

    unique_groups = list(group_map.keys())
    rng.shuffle(unique_groups)

    n_total = len(unique_groups)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    updated_records = [dict(r) for r in records]
    for gid, indices in group_map.items():
        if gid in train_groups:
            split_name = "train"
        elif gid in val_groups:
            split_name = "val"
        else:
            split_name = "test"

        for idx in indices:
            updated_records[idx]["split"] = split_name

    return updated_records


def create_stratified_splits(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """Assign splits stratified by class label if patient/slide IDs are absent."""
    labels = [r.get("label", "unknown") for r in records]
    indices = np.arange(len(records))

    # If single class or degenerate
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or np.min(counts) < 2:
        # Fallback to random shuffle
        rng = np.random.RandomState(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = set(shuffled[:n_train])
        val_idx = set(shuffled[n_train : n_train + n_val])
    else:
        # First split train vs (val + test)
        eval_ratio = val_ratio + test_ratio
        train_idx_arr, eval_idx_arr, train_y, eval_y = train_test_split(
            indices,
            labels,
            test_size=eval_ratio,
            stratify=labels,
            random_state=seed,
        )

        # Then split val vs test
        val_share_of_eval = val_ratio / eval_ratio
        try:
            val_idx_arr, test_idx_arr, _, _ = train_test_split(
                eval_idx_arr,
                eval_y,
                test_size=1.0 - val_share_of_eval,
                stratify=eval_y,
                random_state=seed,
            )
        except ValueError:
            # Fallback if evaluation set has singletons
            n_eval = len(eval_idx_arr)
            n_v = int(n_eval * val_share_of_eval)
            val_idx_arr = eval_idx_arr[:n_v]
            test_idx_arr = eval_idx_arr[n_v:]

        train_idx = set(train_idx_arr)
        val_idx = set(val_idx_arr)

    updated_records = [dict(r) for r in records]
    for idx, r in enumerate(updated_records):
        if idx in train_idx:
            r["split"] = "train"
        elif idx in val_idx:
            r["split"] = "val"
        else:
            r["split"] = "test"

    return updated_records
