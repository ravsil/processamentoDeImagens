---
name: pathology-evaluation-reporting
description: >-
  Use this skill to compute per-class and macro metrics, plot normalized confusion matrices, extract error galleries, and build the SIBGRAPI / IEEEtran LaTeX report.
---

# Pathology Evaluation & Academic Reporting

This skill guides quantitative metric computation, qualitative failure inspection, and LaTeX document compilation for course deliverables.

## 1. Metrics Computation (`src/pathoshift/training/metrics.py`)

Raw accuracy alone is insufficient in digital pathology due to class imbalance. The evaluation suite computes:

- **Per-Class Metrics**: Precision, Recall, F1-Score, and Support for each tissue category.
- **Macro-Averaged Metrics**: Macro Precision, Macro Recall, Macro-F1 (equal weight per class).
- **Weighted-Averaged Metrics**: Weighted by class frequency.
- **Balanced Accuracy**: Arithmetic mean of recalls achieved across all classes:
  $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^K \text{Recall}_k$$
- **Confusion Matrix**:
  - Raw counts ($C_{ij}$)
  - Row-normalized percentages ($\frac{C_{ij}}{\sum_j C_{ij}}$) to easily spot minority class confusion.
- **Confidence Calibration Statistics**:
  - Mean predicted probability on correct predictions vs. incorrect predictions.

## 2. Qualitative Error Inspection Gallery

The CLI generates an inspection grid of the model's most confident mistakes:
- Ranks errors where $\hat{y} \ne y$ descending by predicted probability $P(\hat{y})$.
- Plots the high-confidence false predictions in `outputs/plots/error_gallery.png`.
- Essential for identifying scanner artifacts, cutting folds, or borderline histological cases in the academic report.

Run the evaluation command:
```bash
pathoshift evaluate \
  --input outputs/manifest.jsonl \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json \
  --plots outputs/plots
```

---

## 3. LaTeX Report Compilation (`report/`)

The report uses the official SIBGRAPI IEEEtran conference template:
- Main file: `report/report.tex`
- Citations: `report/references.bib`
- Class file: `report/IEEEtran.cls`

### Compilation Command:
```bash
cd report
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

### Pre-Submission Checklist:
1. Update `\author{...}` block with the names and emails of all 3 students in the group.
2. Ensure figures generated in `outputs/eda/` and `outputs/plots/` are properly referenced:
   - `eda_class_distribution.png`
   - `eda_color_channels.png`
   - `confusion_matrix.png`
   - `error_gallery.png`
3. Check that the final PDF (`report/report.pdf`) does not exceed the allowed page limit.
4. Verify that the ethical and non-clinical disclaimer is preserved.
