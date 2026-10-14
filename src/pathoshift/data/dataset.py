"""PyTorch Dataset for histopathology tiles loaded from JSONL manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from pathoshift.data.manifest import load_manifest


class PathologyDataset(Dataset):
    """Histopathology image tile dataset backed by a validated JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        filter_rejected: bool = False,
        class_to_idx: Optional[Dict[str, int]] = None,
        return_meta: bool = False,
    ):
        """
        Args:
            manifest_path: Path to manifest.jsonl.
            split: 'train', 'val', 'test', or 'all'.
            transform: Callable transform (e.g. torchvision.transforms).
            filter_rejected: If True, exclude tiles marked is_accepted=False.
            class_to_idx: Optional predefined mapping of class names to integer IDs.
            return_meta: If True, __getitem__ returns (tensor, label, meta_dict).
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta

        all_records = load_manifest(self.manifest_path)

        # Filter by split
        if split != "all":
            records = [r for r in all_records if r.get("split") == split]
        else:
            records = all_records

        # Optional quality rejection
        if filter_rejected:
            records = [r for r in records if r.get("is_accepted", True)]

        if not records:
            raise ValueError(
                f"No records found for split '{split}' in manifest {manifest_path}"
            )

        self.records = records

        # Derive or validate class mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(list(set(r.get("label", "unknown") for r in all_records)))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, int] | Tuple[Any, int, Dict[str, Any]]:
        record = self.records[idx]
        image_path = Path(record.get("absolute_path", ""))
        if not image_path.exists():
            rel_path = record.get("relative_path", "")
            if rel_path:
                candidates = [
                    self.manifest_path.parent / rel_path,
                    Path.cwd() / "data" / rel_path,
                    Path.cwd() / rel_path,
                ]
                for cand in candidates:
                    if cand.exists():
                        image_path = cand
                        break

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_name = record.get("label", "unknown")
        label_idx = self.class_to_idx.get(label_name, -1)

        if self.return_meta:
            return image, label_idx, record

        return image, label_idx

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[i] for i in range(len(self.class_to_idx))]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
