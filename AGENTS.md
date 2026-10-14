# AGENTS.md — PathoShift Challenge Agent Summary

This document serves as the high-level operational guide for AI agents working within the **PathoShift** repository. Detailed workflows and procedures are partitioned into modular skills under [`.agents/skills/`](file:///home/rafael/procimgs/.agents/skills/).

---

## 1. Project Overview

- **Challenge**: PathoShift Challenge — Robust Histopathology Image Processing & Classification (UFRRJ).
- **Core Problem**: Build an end-to-end, reproducible pipeline for classifying microscopic H&E tiles under domain shifts (stain variations, focus defects, scanner differences).
- **Current Milestone**: **Entrega 1 — Pathology Image Pipeline**.
- **Package**: `pathoshift` (installable via `pip install -e .`).
- **Primary CLI**: `pathoshift [prepare | eda | train | predict | evaluate | describe]`.

---

## 2. Environment & Compute

- **Python Version**: 3.12.3 in local virtual environment (`.venv`).
- **Deep Learning Framework**: PyTorch 2.6.0+cu124 & Torchvision 0.21.0.
- **Hardware Accelerator**: NVIDIA GeForce RTX 3050 (6GB VRAM, CUDA 12.4).
- **Activation**: `source .venv/bin/activate`

---

## 3. Modular Skills Index

Detailed procedures are modularized into workspace skills:

| Skill | Path | Description |
| :--- | :--- | :--- |
| **Pipeline Execution** | [`.agents/skills/pipeline-execution/SKILL.md`](file:///home/rafael/procimgs/.agents/skills/pipeline-execution/SKILL.md) | Operating CLI commands (`prepare`, `eda`, `train`, `predict`, `evaluate`, `describe`) and demo data generation. |
| **Stain & Preprocessing** | [`.agents/skills/stain-preprocessing/SKILL.md`](file:///home/rafael/procimgs/.agents/skills/stain-preprocessing/SKILL.md) | Tissue mask segmentation, focus scoring ($\sigma^2_{\nabla^2}$), and Macenko/Reinhard stain normalization. |
| **Model Training** | [`.agents/skills/model-training/SKILL.md`](file:///home/rafael/procimgs/.agents/skills/model-training/SKILL.md) | Training ResNet-18, SimpleCNN, and classical feature extractors (GLCM, LBP, color moments) with RF/SVM. |
| **Evaluation & Reporting** | [`.agents/skills/evaluation-reporting/SKILL.md`](file:///home/rafael/procimgs/.agents/skills/evaluation-reporting/SKILL.md) | Balanced accuracy, macro metrics, confusion matrices, error gallery inspection, and SIBGRAPI LaTeX report. |

---

## 4. Key Architectural Patterns

- **CLI-First**: All operations are exposed via Typer subcommands (`src/pathoshift/cli.py`).
- **Config-Driven**: Experiments and hyperparameters reside in YAML configs (`configs/default.yaml`).
- **Data Integrity**: Every image in the manifest is validated and indexed with its SHA-256 hash.
- **Leakage Prevention**: Partitions are grouped by `patient_id` or `slide_id` to prevent tile-level leakage.
- **Self-Contained Reporting**: Reports are authored in LaTeX using the SIBGRAPI IEEE template (`report/`).

---

## 5. Golden Rules for Agents

1. **No Data Leakage**: Never evaluate or choose hyperparameters on the test split. Always use patient-aware grouping when patient/slide IDs exist.
2. **Deterministic Seeding**: Always pass and set the random seed (`set_seed(config['seed'])`) across Python `random`, `numpy`, and `torch`.
3. **Reproducibility Tracking**: Document package versions, environment details, and execution wall-clock time in output logs (`training_history.json`).
4. **Code Quality**: Always run `pytest tests/` after modifying core data, model, or evaluation logic.
5. **Ethical Disclaimer**: Never produce or suggest clinical diagnostic claims; this system is strictly for experimental computer vision research.
