# AGENTS.md — PathoShift Challenge Agent Summary

This document serves as the high-level operational guide for AI agents working within the **PathoShift** repository. Detailed workflows and procedures are partitioned into modular skills under [`skills/`](skills/).

---

## 1. Project Overview

- **Challenge**: PathoShift-CRC Mini Challenge — Robust Histopathology Image Processing & Classification (UFRRJ).
- **Core Problem**: Build an end-to-end, reproducible CPU-executable pipeline for classifying 224×224 H&E colorectal tissue tiles (9 classes from `NCT-CRC-HE-100K-NONORM`: `ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`) under acquisition shifts (`brightness_contrast`, `jpeg`, `gaussian_blur`, `gaussian_noise`).
- **Current Milestone**: **Entrega Final — PathoShift-CRC Official Protocol & Robustness Evaluation** (reusing Entrega 1 baseline).
- **Package**: `pathoshift` (installable via `pip install -e .`).
- **Primary Final CLI**: `python -m pathoshift.challenge [eda | features | experiment | predict | sensitivity | all]`.
- **Legacy Entrega 1 CLI**: `pathoshift [prepare | eda | train | predict | evaluate | describe]`.

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
| **Pipeline Execution** | [`skills/pipeline-execution/SKILL.md`](skills/pipeline-execution/SKILL.md) | Operating CLI commands (`prepare`, `eda`, `train`, `predict`, `evaluate`, `describe`) and demo data generation. |
| **Stain & Preprocessing** | [`skills/stain-preprocessing/SKILL.md`](skills/stain-preprocessing/SKILL.md) | Tissue mask segmentation, focus scoring ($\sigma^2_{\nabla^2}$), and Macenko/Reinhard stain normalization. |
| **Model Training** | [`skills/model-training/SKILL.md`](skills/model-training/SKILL.md) | Training ResNet-18, SimpleCNN, and classical feature extractors (GLCM, LBP, color moments) with RF/SVM. |
| **Evaluation & Reporting** | [`skills/evaluation-reporting/SKILL.md`](skills/evaluation-reporting/SKILL.md) | Balanced accuracy, macro metrics, confusion matrices, error gallery inspection, and SIBGRAPI LaTeX report. |

---

## 4. Key Architectural Patterns

- **CLI-First**: All operations are exposed via Typer subcommands (`src/pathoshift/cli.py`).
- **Config-Driven**: Experiments and hyperparameters reside in YAML configs (`configs/default.yaml`).
- **Data Integrity**: Every image in the manifest is validated and indexed with its SHA-256 hash.
- **Leakage Prevention**: Partitions are grouped by `patient_id` or `slide_id` to prevent tile-level leakage.
- **Self-Contained Reporting**: Reports are authored in LaTeX using the SIBGRAPI IEEE template (`report/`).

---

## 5. Golden Rules for Agents

1. **No Data Leakage**: Train exclusively on `train.csv` (9,000 tiles). Select configuration and hyperparameters ($C$) exclusively on `validation.csv` (clean + `validation_proxy` generated with independent seed `2000·10⁵ + i`) and freeze `outputs/results/selection.json` before evaluating `robustness.csv`. Never retrain or tune on `robustness.csv`.
2. **Deterministic Seeding**: Use global seed `42`, per-row deterministic augmentation/proxy seeds (`1000·10⁵ + i` and `2000·10⁵ + i`), and manifest-provided seeds for `robustness.csv`.
3. **Linux Multiprocessing Safety**: Always call `cv2.setNumThreads(1)` before invoking OpenCV functions in the parent process prior to `multiprocessing.Pool` creation to avoid Linux `fork()` pthreads deadlocks.
4. **Code Quality**: Always run `pytest tests/` (18 unit tests) and validate `predictions_validation.csv` / `predictions_robustness.csv` with `node evaluate_pathoshift_crc.mjs`.
5. **Ethical Disclaimer**: Never produce or suggest clinical diagnostic claims; this system is strictly for experimental computer vision research.

---

## 6. Design Decisions & Resolved Questions

### `brightness_contrast` Perturbation Formula Strategy (`robustness.csv`)
- **Context**: `robustness.csv` specifies `perturbation=brightness_contrast` with `parameters="brightness=0.15;contrast=1.20"`, without stating the algebraic equation. Because `ADI` and `BACK` tiles are near-white (~240 intensity), additive brightness clips pixels at 255 and erases fine membrane structures.
- **Candidate Formulas (`src/pathoshift/preprocessing/perturbations.py`)**:
  - **A (Official default — OpenCV/Albumentations $\alpha/\beta$)**: $y = \operatorname{clip}(\operatorname{round}(1.2\,x + 0.15\cdot 255),\, 0, 255)$
  - **B (Contrast centered at mid-gray 128)**: $y = \operatorname{clip}(\operatorname{round}((x - 128)\cdot 1.2 + 128 + 0.15\cdot 255),\, 0, 255)$
  - **C (Contrast centered at per-image mean $\mu$)**: $y = \operatorname{clip}(\operatorname{round}((x - \mu)\cdot 1.2 + \mu + 0.15\cdot 255),\, 0, 255)$
  - **D (Multiplicative brightness + PIL/torchvision contrast)**: $x_1 = \operatorname{clip}(\operatorname{round}(1.15\,x),\, 0, 255)$, $y = \operatorname{clip}(\operatorname{round}((x_1 - \mu_1)\cdot 1.2 + \mu_1),\, 0, 255)$
- **Decided Strategy**:
  1. **Keep Formula A** for the delivered `predictions_robustness.csv` and for training augmentation (`random_acquisition_shift`).
  2. **Sensitivity Analysis Without Retraining (`python -m pathoshift.challenge sensitivity`)**: Evaluate the already-trained frozen baseline (`raw`, $C=10$) and proposal (`gauss1+aug`, $C=10$, trained only with formula A) on the 563 `brightness_contrast` validation tiles under formulas A, B, C, and D. This obeys the no-tuning rule on `robustness.csv` while turning the formula assumption into a measured limitation in the report (Section VIII & Appendix Table III).
  3. **Measured Outcome**: Formula A is the worst-case clipping regime (95.8% of `ADI` and 99.7% of `BACK` pixels clip at 255). Even without retraining, the proposal's F1-macro on the `brightness_contrast` subset rises from **0.830** (A) to **0.896** (B), **0.890** (C), and **0.843** (D) (overall robustness F1-macro **0.925** in A vs. **0.929–0.941** in B–D, compared to **0.602–0.649** for the baseline).
