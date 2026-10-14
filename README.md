# PathoShift: Robust Histopathology Image Processing & Classification Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, reproducible, audit-ready computational pathology pipeline built for the **PathoShift Challenge (Entrega 1: Pathology Image Pipeline)**.

---

## 📋 Features & Entrega 1 Requirements

- **1. Data Ingestion & Manifest**: Automatic recursive dataset discovery, image format validation, SHA-256 cryptographic integrity hashing, and patient/slide-aware non-leaking train/val/test splits (`manifest.jsonl`).
- **2. Exploratory Data Analysis (EDA)**: Automated generation of class distribution charts, RGB/HSV photometric density curves, and quantitative focus scoring ($\sigma^2_{\nabla^2}$) in `outputs/eda/`.
- **3. Justified Preprocessing**:
  - HSV saturation-based glass background rejection.
  - Laplacian variance focus filtering.
  - Optical Density SVD-based **Macenko normalisation** and perceptual $L\alpha\beta$ **Reinhard color transfer**.
- **4. Baseline Classification**:
  - Deep Learning: **ResNet-18** (ImageNet pretrained & from scratch) with class-weighted cross-entropy and early stopping.
  - Classical Machine Learning: **Random Forest** and **SVM** on GLCM texture, LBP, and color moments.
- **5. Evaluation Suite**:
  - Per-class precision, recall, F1, and support.
  - Macro and weighted averages, balanced accuracy.
  - Raw and row-normalized confusion matrices (`confusion_matrix.png`).
  - Automated inspection gallery of top confident classification errors (`error_gallery.png`).
- **6. Reproducibility & Tracking**:
  - `pathoshift describe` command reporting OS, CPU, CUDA GPU accelerators, and exact package versions.
  - Wall-clock execution time logging and deterministic seeding (`outputs/training_history.json`).
- **7. Report Template**: SIBGRAPI / IEEEtran LaTeX paper template in `report/` ready to compile (`report.pdf`).

---

## 📁 Repository Structure

```
procimgs/
├── configs/
│   └── default.yaml             # Experiment hyperparameters & pipeline options
├── data/                        # Datasets (raw & sample)
│   └── sample/                  # Synthetic H&E tiles for quick testing
├── outputs/                     # Generated artifacts & results
│   ├── checkpoints/             # Best & last PyTorch weights (.pt)
│   ├── eda/                     # EDA charts and numerical summary JSON
│   ├── plots/                   # Confusion matrix & error inspection gallery
│   ├── manifest.jsonl           # Validated dataset manifest
│   ├── predictions.jsonl        # Model inference outputs
│   └── metrics.json             # Per-class & macro evaluation metrics
├── report/                      # SIBGRAPI IEEEtran LaTeX report
│   ├── IEEEtran.cls
│   ├── report.tex               # Paper report source
│   ├── references.bib           # BibTeX citations
│   └── report.pdf               # Compiled 2-page conference report
├── src/
│   └── pathoshift/
│       ├── cli.py               # Typer CLI (prepare, predict, evaluate, describe, train, eda)
│       ├── config.py            # YAML configuration loader & deterministic seeding
│       ├── data/
│       │   ├── manifest.py      # Manifest scanner, hashing & integrity checker
│       │   ├── dataset.py       # PyTorch Dataset for pathology tiles
│       │   ├── splits.py        # Patient/slide-aware and stratified splits
│       │   └── synthetic.py     # Synthetic H&E tile generator
│       ├── preprocessing/
│       │   ├── quality.py       # Background rejection & Laplacian focus measure
│       │   ├── stain_norm.py    # Macenko & Reinhard stain normalisation
│       │   └── transforms.py    # Torchvision augmentation & transform pipelines
│       ├── models/
│       │   ├── cnn.py           # ResNet-18, ResNet-50, SimpleCNN
│       │   └── classical.py     # GLCM, LBP, color features + RF/SVM
│       ├── training/
│       │   ├── trainer.py       # Training loop with early stopping & timing
│       │   └── metrics.py       # Precision, recall, macro-F1, confusion matrix
│       ├── visualization/
│       │   ├── eda.py           # Automated EDA plotting
│       │   └── report_plots.py  # Confusion matrix & error gallery visualizer
│       └── utils/
│           └── system_info.py   # Hardware, OS, CUDA, and environment tracker
├── tests/                       # Complete pytest unit and integration test suite
├── pyproject.toml
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e .
```

### 2. Environment Verification

Inspect platform, hardware accelerator (CUDA), and dependency versions:

```bash
pathoshift describe
```

### 3. Generate Sample Dataset (for local testing)

```bash
pathoshift generate-sample-data --output data/sample --patients 6 --tiles-per-patient 8
```

### 4. Official CLI Workflow

#### Step 1: Prepare Dataset & Generate Manifest
```bash
pathoshift prepare --input data/sample --output outputs --strategy patient
```

#### Step 2: Run Exploratory Data Analysis (EDA)
```bash
pathoshift eda --manifest outputs/manifest.jsonl --output outputs/eda
```

#### Step 3: Train Baseline Model
```bash
pathoshift train --manifest outputs/manifest.jsonl --config configs/default.yaml --output outputs
```

#### Step 4: Batch Inference & Predictions
```bash
pathoshift predict --input outputs/manifest.jsonl --output outputs/predictions.jsonl --model outputs/checkpoints/best_model.pt
```

#### Step 5: Evaluate Predictions
```bash
pathoshift evaluate --input outputs/manifest.jsonl --predictions outputs/predictions.jsonl --output outputs/metrics.json --plots outputs/plots
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

All 10 unit and integration tests pass verifying data splits, stain normalization, focus calculation, metric computations, and CLI pipelines.

---

## 📄 Compiling the Report

The SIBGRAPI IEEE conference report is located in `report/`:

```bash
cd report
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

This generates `report/report.pdf`.

---

## ⚠️ Ethical & Clinical Disclaimer

This software is developed strictly for academic evaluation and research within the scope of the Digital Image Processing course. It does not constitute a medical device, nor should its predictions be used for clinical decision-making or diagnosis.
