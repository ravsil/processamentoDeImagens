---
name: pathology-stain-preprocessing
description: >-
  Use this skill to configure, apply, and benchmark histopathology image preprocessing algorithms: background glass rejection, Laplacian focus evaluation, and Macenko or Reinhard stain normalization.
---

# Histopathology Preprocessing & Stain Normalization

This skill guides the implementation, tuning, and evaluation of image quality and stain correction techniques in digital pathology.

## 1. Background Glass Rejection (`src/pathoshift/preprocessing/quality.py`)

Histopathology whole-slide crops often contain large regions of empty glass slide.
- **Method**: Saturation and luminance filtering.
  - Tissue stained with Hematoxylin (nuclei) and Eosin (cytoplasm/stroma) exhibits higher color saturation than empty white glass.
  - Convert image to HSV space:
    $$\text{is\_tissue} = (\text{Saturation} \ge \text{threshold}) \land (\text{Grayscale} \le \text{max\_luminance})$$
  - Default parameters: `saturation_threshold = 15`, `max_luminance = 245`, `min_tissue_ratio = 0.40`.
- **Usage**:
  ```python
  from pathoshift.preprocessing.quality import compute_tissue_mask
  mask, ratio = compute_tissue_mask(image_rgb, saturation_threshold=15)
  if ratio < 0.40:
      # Flag or reject empty glass tile
  ```

## 2. Focus & Blur Assessment

Histopathology images may suffer from scanner defocus or thickness variation.
- **Metric**: Variance of the Laplacian ($\sigma^2_{\nabla^2}$).
  $$\text{Focus Score} = \text{Var}(\nabla^2 I_{\text{gray}})$$
- High edge density (sharp chromatin granules, nuclear membranes) produces high variance (> 100). Low contrast or out-of-focus blurs yield low variance (< 50).
- **Usage**:
  ```python
  from pathoshift.preprocessing.quality import compute_focus_measure
  score = compute_focus_measure(image_rgb)
  ```

## 3. Stain Normalization Algorithms (`src/pathoshift/preprocessing/stain_norm.py`)

Inter-laboratory and inter-scanner variations alter H&E appearance. Two standard methods are available:

### Method A: Macenko Stain Normalization
- Converts RGB to Optical Density (OD) space:
  $$\text{OD} = -\log_{10}\left(\frac{I + 1}{255}\right)$$
- Identifies tissue pixels ($\text{OD} > \beta$, where $\beta = 0.15$).
- Performs Singular Value Decomposition (SVD) on covariance matrix of OD vectors.
- Projects OD vectors onto the 2D plane and robustly extracts Hematoxylin and Eosin stain vectors from extremal percentiles (1st and 99th).
- Solves for stain concentrations: $C = \text{OD} \cdot S^+$
- Maps concentrations to reference target slide matrices and reconstructs normalized RGB.
- **Usage**:
  ```python
  from pathoshift.preprocessing.stain_norm import MacenkoNormalizer
  normalizer = MacenkoNormalizer()
  normalized_rgb = normalizer.transform(image_rgb)
  ```

### Method B: Reinhard Color Transfer
- Maps image to perceptually uniform $L\alpha\beta$ color space.
- Normalizes source channels using reference means and standard deviations:
  $$c_{\text{norm}} = (c_{\text{src}} - \mu_{\text{src}}) \cdot \frac{\sigma_{\text{tgt}}}{\sigma_{\text{src}}} + \mu_{\text{tgt}}$$
- Faster than Macenko, suitable as a lightweight alternative.

## 4. Configuration Toggle & Experimental Ablation

To enable stain normalization during training, edit `configs/default.yaml`:
```yaml
preprocessing:
  stain_normalization:
    enabled: true        # set to true to benchmark stain normalization impact
    method: "macenko"    # "macenko" or "reinhard"
```
Always report baseline performance **with vs. without** stain normalization to satisfy the course requirement of an **experimentally justified preprocessing choice**.
