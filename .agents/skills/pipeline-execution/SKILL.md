---
name: pathology-pipeline-execution
description: >-
  Use this skill to run, operate, and troubleshoot the PathoShift histopathology pipeline CLI commands (prepare, eda, train, predict, evaluate, describe) and synthetic demo data generation.
---

# Pathology Pipeline Execution Runbook

This skill outlines the command-line workflows for operating the PathoShift pipeline.

## Environment Setup

Always ensure the local virtual environment is activated and dependencies are loaded:

```bash
source .venv/bin/activate
# Verify hardware and dependencies
pathoshift describe
```

## Step-by-Step Command Workflows

### 1. Generating Mock Data (for testing when raw data is missing)

```bash
pathoshift generate-sample-data --output data/sample --patients 6 --tiles-per-patient 8
```
- Creates synthetic H&E tiles with nuclei clusters and stroma.
- Formats patient IDs into filenames (`patient_XXX_slide_YY_tile_ZZZ.png`) for leakage-free split verification.

### 2. Dataset Ingestion & Manifest Generation (`prepare`)

```bash
pathoshift prepare --input <path_to_images> --output outputs --strategy patient --seed 42
```
- Recursively inspects images under the input directory.
- Verifies readability and calculates **SHA-256** checksums.
- Extracts `patient_id` or `slide_id` if present in filenames/paths.
- Splits into `train` (70%), `val` (15%), and `test` (15%) partitions without patient-level leakage.
- Output: `outputs/manifest.jsonl` and `outputs/prepare_summary.json`.

### 3. Automated Exploratory Data Analysis (`eda`)

```bash
pathoshift eda --manifest outputs/manifest.jsonl --output outputs/eda
```
- Produces:
  - `eda_class_distribution.png`: Bar charts of class distribution across splits.
  - `eda_color_channels.png`: Photometric density curves for RGB and HSV channels.
  - `eda_quality_metrics.png`: Tissue ratio and Laplacian variance focus distributions.
  - `eda_summary.json`: Aggregate statistics (imbalance ratio, dimensions, mean focus).

### 4. Training Baseline Models (`train`)

```bash
pathoshift train --manifest outputs/manifest.jsonl --config configs/default.yaml --output outputs
```
- Trains model specified in YAML (`resnet18`, `simple_cnn`, etc.).
- Saves best model based on validation Macro-F1 to `outputs/checkpoints/best_model.pt`.
- Saves last checkpoint to `outputs/checkpoints/last_model.pt`.
- Logs training metrics and wall-clock execution time to `outputs/training_history.json`.

### 5. Batch Inference (`predict`)

```bash
pathoshift predict --input outputs/manifest.jsonl --output outputs/predictions.jsonl --model outputs/checkpoints/best_model.pt
```
- Runs inference over all tiles listed in the manifest.
- Outputs `outputs/predictions.jsonl` formatted with `image_id`, `prediction`, `confidence`, and class `probabilities`.

### 6. Quantitative Evaluation & Error Gallery (`evaluate`)

```bash
pathoshift evaluate --input outputs/manifest.jsonl --predictions outputs/predictions.jsonl --output outputs/metrics.json --plots outputs/plots
```
- Calculates per-class and macro precision, recall, F1, and balanced accuracy.
- Generates `outputs/plots/confusion_matrix.png`.
- Generates `outputs/plots/error_gallery.png` highlighting top high-confidence classification errors.

## Verification & Troubleshooting

- Run automated tests: `pytest tests/`
- Ensure DataLoader `num_workers=0` on small datasets or WSL2 environments to avoid IPC worker deadlocks.
- Ensure CUDA is available via `torch.cuda.is_available()` (system has NVIDIA GeForce RTX 3050).
