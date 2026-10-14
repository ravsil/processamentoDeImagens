"""Training orchestration for deep learning and classical baseline models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathoshift.config import set_seed
from pathoshift.data.dataset import PathologyDataset
from pathoshift.models.classical import ClassicalBaselineModel
from pathoshift.models.cnn import get_model
from pathoshift.preprocessing.transforms import get_image_transforms
from pathoshift.training.metrics import compute_comprehensive_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute a single neural network training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], List[int], List[int], np.ndarray]:
    """Evaluate neural network on validation or test dataset."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        all_probs.extend(probs)
        total += targets.size(0)

    eval_loss = running_loss / total if total > 0 else 0.0
    y_prob_arr = np.array(all_probs)
    metrics = compute_comprehensive_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        y_prob=y_prob_arr,
    )

    return eval_loss, metrics, all_targets, all_preds, y_prob_arr


def train_deep_model(
    config: Dict[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Train a CNN baseline model and log checkpoints and execution metrics."""
    start_time = time.time()
    set_seed(config.get("seed", 42))

    # Device configuration
    dev_cfg = config.get("device", "auto")
    if dev_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_cfg)

    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    target_size = tuple(config.get("data", {}).get("target_size", [224, 224]))
    prep_cfg = config.get("preprocessing", {})
    stain_cfg = prep_cfg.get("stain_normalization", {})
    use_stain = stain_cfg.get("enabled", False)
    stain_method = stain_cfg.get("method", "macenko")

    train_tf = get_image_transforms(
        target_size=target_size,
        split="train",
        use_stain_norm=use_stain,
        stain_method=stain_method,
        augmentation=prep_cfg.get("augmentation", {}),
    )
    eval_tf = get_image_transforms(
        target_size=target_size,
        split="val",
        use_stain_norm=use_stain,
        stain_method=stain_method,
    )

    # Datasets and Loaders
    train_ds = PathologyDataset(manifest_path, split="train", transform=train_tf)
    val_ds = PathologyDataset(
        manifest_path, split="val", transform=eval_tf, class_to_idx=train_ds.class_to_idx
    )

    batch_size = config.get("model", {}).get("batch_size", 32)
    cfg_workers = config.get("num_workers", 0)
    # Avoid multi-worker queue starvation when number of samples is small
    workers = min(cfg_workers, max(0, len(train_ds) // batch_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(workers > 0 and torch.cuda.is_available()),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(workers > 0 and torch.cuda.is_available()),
    )

    # Model instantiation
    m_cfg = config.get("model", {})
    arch = m_cfg.get("type", "resnet18")
    pretrained = m_cfg.get("pretrained", True)
    num_classes = train_ds.num_classes

    model = get_model(model_name=arch, num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    # Calculate class weights for loss if requested
    labels = [r["label"] for r in train_ds.records]
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = torch.tensor(
        [float(len(labels)) / (len(unique_labels) * count) for count in counts],
        dtype=torch.float,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr = float(m_cfg.get("learning_rate", 0.0003))
    wd = float(m_cfg.get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    epochs = m_cfg.get("epochs", 10)
    patience = m_cfg.get("early_stopping_patience", 4)

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    print(f"Training {arch} on {device} ({num_classes} classes, {len(train_ds)} train samples, {len(val_ds)} val samples)...", flush=True)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_model(
            model, val_loader, criterion, device, class_names=train_ds.classes
        )

        val_f1 = val_metrics["macro"]["f1"]
        scheduler.step(val_f1)
        ep_duration = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "duration_seconds": round(ep_duration, 2),
        })

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Macro-F1: {val_f1:.4f} ({ep_duration:.2f}s)",
            flush=True,
        )

        # Checkpoint best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                ckpt_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save last model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "config": config,
        },
        ckpt_dir / "last_model.pt",
    )

    total_time = time.time() - start_time
    timing = {
        "start_timestamp": start_time,
        "total_seconds": round(total_time, 2),
        "epochs_trained": len(history),
        "best_val_macro_f1": best_val_f1,
        "device": str(device),
    }

    # Save training logs
    with open(out_path / "training_history.json", "w") as f:
        json.dump({"history": history, "timing": timing}, f, indent=2)

    return {"best_macro_f1": best_val_f1, "timing": timing, "history": history}
