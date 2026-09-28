"""
ml/training/torch_common.py
───────────────────────────
Shared PyTorch training loop for the video-keypoint comparison matrix.

Every arch trains under identical discipline: same splits (from
ml.training.datasets.video_keypoints), CrossEntropyLoss, AdamW, early
stopping on val accuracy, ReduceLROnPlateau, best-checkpoint save.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


def _epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch. Returns (avg_loss, accuracy, all_true, all_pred)."""
    import torch

    training = optimizer is not None
    model.train(training)
    if not training:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.float().to(device), yb.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum())
            total += len(xb)
            all_true.extend(yb.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), all_true, all_pred


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    min_delta: float = 0.002,
    checkpoint_path: str | Path,
) -> dict:
    """Full training cycle with early stopping on val accuracy. Returns history."""
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
    }
    best_acc, bad_epochs = -1.0, 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = _epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, _, _ = _epoch(model, val_loader, criterion, device)

        scheduler.step(va_acc)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = va_acc > best_acc + min_delta
        if improved:
            best_acc = va_acc
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:>3}/{epochs}] loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
            f"(best={best_acc:.4f}){' *' if improved else ''}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"[Training] Early stop at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    history["best_val_acc"] = best_acc
    history["stopped_epoch"] = epoch
    return history


def evaluate_loader(model, loader, device) -> dict:
    """Accuracy + per-class report inputs on one loader."""
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred = _epoch(model, loader, criterion, device)
    return {"loss": loss, "accuracy": acc, "y_true": y_true, "y_pred": y_pred}


def save_reports(y_true, y_pred, labels: list[str], reports_dir: str | Path) -> dict:
    """Write classification_report.txt + confusion_matrix.png. Returns metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report_txt = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    return {"accuracy": accuracy, "f1_macro": f1_macro}


def save_history_plot(history: dict, plots_dir: str | Path) -> None:
    """Write accuracy/loss curves."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_acc"], label="train acc")
    ax1.plot(epochs, history["val_acc"], label="val acc")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(epochs, history["train_loss"], label="train loss")
    ax2.plot(epochs, history["val_loss"], label="val loss")
    ax2.set_title("Loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "training_history.png", dpi=120)
    plt.close(fig)
