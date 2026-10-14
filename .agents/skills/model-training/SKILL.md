---
name: pathology-model-training
description: >-
  Use this skill to configure, train, fine-tune, and optimize classification baselines for histopathology tiles, including CNNs (ResNet-18, SimpleCNN) and classical ML (GLCM, LBP, Random Forest, SVM).
---

# Histopathology Model Training & Optimization

This skill provides guidelines and procedures for training both deep learning and classical machine learning baselines for pathology tile classification.

## 1. Deep Learning Baseline (`src/pathoshift/models/cnn.py`)

### Architecture
- **ResNet-18**: Standard baseline in digital pathology literature. Pretrained on ImageNet with the final fully connected layer adapted to `num_classes`:
  ```python
  from pathoshift.models.cnn import get_model
  model = get_model("resnet18", num_classes=2, pretrained=True)
  ```
- **SimpleCNN**: 4-stage convolutional baseline with BatchNorm, ReLU, and AdaptiveAvgPool2d for fast sanity checks or low-compute environments.

### Training Strategies (`src/pathoshift/training/trainer.py`)
1. **Class Imbalance Mitigation**:
   - Computes reciprocal class frequencies:
     $$w_c = \frac{N}{K \cdot N_c}$$
   - Passed directly to `nn.CrossEntropyLoss(weight=class_weights)`.
2. **Optimizer & Learning Rate Schedule**:
   - Optimizer: `AdamW` with $\text{lr} = 3 \times 10^{-4}$ and $\text{weight\_decay} = 10^{-4}$.
   - Scheduler: `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)` tracking validation Macro-F1.
3. **Early Stopping**:
   - Monitors validation Macro-F1 with configurable patience (default: 4 epochs).
   - Checkpoints the best model weights to `outputs/checkpoints/best_model.pt`.
4. **Data Augmentation**:
   - Random horizontal & vertical flips (cells have no natural anatomical orientation in microscopic tiles).
   - Random 90-degree rotations.
   - Subtle ColorJitter ($\pm 10\%$).

### DataLoader Best Practices
- When training on small datasets or inside WSL2, configure `num_workers=0` (or dynamic `min(cfg_workers, len(train_ds) // batch_size)`) to prevent multiprocessing queue hangs.
- Pin memory only when `workers > 0` and CUDA is active.

---

## 2. Classical Machine Learning Baseline (`src/pathoshift/models/classical.py`)

To satisfy the baseline comparison requirement, a classical computer vision pipeline is provided:

### Feature Extraction
1. **Color Representations**:
   - 16-bin histograms per channel in RGB and HSV color spaces.
   - Channel statistical moments: mean, standard deviation, and skewness.
2. **Texture Representations**:
   - **GLCM** (Gray-Level Co-occurrence Matrix) properties at distances (1, 3) and angles ($0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}$):
     - Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
   - **LBP** (Local Binary Patterns): uniform rotation-invariant LBP histogram with $P=24, R=3$.

### Classifiers
- **Random Forest**: 150 estimators, balanced class weights, multithreaded (`n_jobs=-1`).
- **SVM**: RBF kernel, probability estimation enabled, feature scaling via `StandardScaler`.
- **Usage**:
  ```python
  from pathoshift.models.classical import ClassicalBaselineModel
  clf = ClassicalBaselineModel(classifier_type="rf", n_estimators=150)
  clf.fit(X_train, y_train)
  preds = clf.predict(X_test)
  clf.save("outputs/checkpoints/classical_model.joblib")
  ```

---

## 3. Configuration Management (`configs/default.yaml`)

Experiment parameters are controlled via YAML:
```yaml
seed: 42
device: "auto"        # "cuda", "cpu", or "auto"
num_workers: 0

model:
  type: "resnet18"    # "resnet18", "simple_cnn", "random_forest", "svm"
  pretrained: true
  learning_rate: 0.0003
  weight_decay: 0.0001
  batch_size: 32
  epochs: 15
  early_stopping_patience: 4
```

To run an experiment:
```bash
pathoshift train --manifest outputs/manifest.jsonl --config configs/default.yaml --output outputs
```
