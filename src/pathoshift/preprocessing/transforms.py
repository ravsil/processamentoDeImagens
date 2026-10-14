"""Data transformation and augmentation pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
from PIL import Image
import numpy as np

from pathoshift.preprocessing.stain_norm import MacenkoNormalizer, ReinhardNormalizer


class StainNormTransform:
    """Callable transform applying stain normalization to PIL or NumPy images."""

    def __init__(self, method: str = "macenko"):
        self.method = method.lower()
        if self.method == "macenko":
            self.normalizer = MacenkoNormalizer()
        elif self.method == "reinhard":
            self.normalizer = ReinhardNormalizer()
        else:
            raise ValueError(f"Unknown stain normalization method: {method}")

    def __call__(self, img: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
            norm_arr = self.normalizer.transform(arr)
            return Image.fromarray(norm_arr)
        elif isinstance(img, np.ndarray):
            norm_arr = self.normalizer.transform(img)
            return Image.fromarray(norm_arr)
        return img


def get_image_transforms(
    target_size: Tuple[int, int] = (224, 224),
    split: str = "train",
    use_stain_norm: bool = False,
    stain_method: str = "macenko",
    augmentation: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build torchvision transform pipeline for training or evaluation.

    Args:
        target_size: (H, W) target image resolution.
        split: 'train', 'val', or 'test'.
        use_stain_norm: Whether to apply stain normalization.
        stain_method: 'macenko' or 'reinhard'.
        augmentation: Dict with augmentation options.

    Returns:
        torchvision.transforms.Compose instance.
    """
    try:
        from torchvision import transforms
    except ImportError:
        # Fallback if torchvision is not yet imported
        return None

    transform_list = []

    # Optional stain normalization step
    if use_stain_norm:
        transform_list.append(StainNormTransform(method=stain_method))

    # Resize to target dimensions
    transform_list.append(transforms.Resize(target_size))

    # Augmentations for training split only
    if split == "train" and augmentation:
        if augmentation.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if augmentation.get("vertical_flip", True):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        if augmentation.get("random_rotate90", True):
            transform_list.append(transforms.RandomRotation(degrees=(0, 270)))
        if augmentation.get("color_jitter", 0.0):
            jitter = augmentation.get("color_jitter", 0.1)
            transform_list.append(
                transforms.ColorJitter(
                    brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 2
                )
            )

    # Standard ImageNet tensor normalization
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transforms.Compose(transform_list)
