"""System and environment inspection utility for reproducibility tracking."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict


def get_git_info() -> Dict[str, Any]:
    """Retrieve Git commit hash and dirty status if in a git repository."""
    info: Dict[str, Any] = {"commit": "unknown", "is_dirty": False, "branch": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit
        
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["is_dirty"] = bool(status)

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        pass
    return info


def get_system_summary() -> Dict[str, Any]:
    """Collect platform, python, dependencies, and accelerator hardware details."""
    summary: Dict[str, Any] = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "git": get_git_info(),
        "packages": {},
        "cuda": {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "cuda_version": None,
        },
    }

    # Inspect installed packages
    pkg_map = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "scikit-image": "skimage",
        "pillow": "PIL",
        "opencv": "cv2",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "torchvision": "torchvision",
        "typer": "typer",
    }
    for display_name, mod_name in pkg_map.items():
        try:
            mod = __import__(mod_name)
            summary["packages"][display_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            summary["packages"][display_name] = "not installed"

    # PyTorch and CUDA check
    try:
        import torch
        summary["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["cuda"]["device_count"] = torch.cuda.device_count()
            summary["cuda"]["device_name"] = torch.cuda.get_device_name(0)
            summary["cuda"]["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    return summary
