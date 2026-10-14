"""Configuration loading, validation, and reproducible seeding."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard library, NumPy, and PyTorch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file with error handling."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return data


def save_config(config_data: Dict[str, Any], output_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
