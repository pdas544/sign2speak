"""
ml/training/train_torch.py
──────────────────────────
Unified PyTorch trainer for the video-keypoint comparison matrix.

Every arch trains on identical inputs (ml.training.datasets.video_keypoints:
15 video glosses, fixed 30x1629 clips, deduped 336/109/86 splits) under the
shared loop in ml.training.torch_common, then registers in
models/registry/registry.json (framework=pytorch).

Usage:
    python -m ml.training.train_torch --arch tcn --model-name tcn_v1
    python -m ml.training.train_torch --arch lstm --model-name lstm_baseline_v1 --epochs 60
    python -m ml.training.train_torch --arch transformer --model-name transformer_v2 --set-active
    # ablations (config-only, no new files):
    python -m ml.training.train_torch --arch lstm --hidden-size 128 --num-layers 2 --model-name lstm_small_v1
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.common import create_run_dir, register_model, write_manifest  # noqa: E402
from ml.training.datasets.video_keypoints import (  # noqa: E402
    INPUT_FEATURES,
    VIDEO_LABELS,
    build_arrays,
)

ARCHES = ("lstm", "gru", "cnn_lstm", "tcn", "transformer")

ARCH_DEFAULTS = {
    "lstm": {"lr": 1e-3, "hidden_size": 256, "num_layers": 3, "dropout": 0.3},
    "gru": {"lr": 1e-3, "hidden_size": 256, "num_layers": 3, "dropout": 0.3},
    "cnn_lstm": {"lr": 1e-3, "hidden_size": 256, "num_layers": 2, "dropout": 0.3},
    "tcn": {"lr": 1e-3, "hidden_size": 128, "num_layers": 4, "dropout": 0.2},
    "transformer": {"lr": 1e-4, "hidden_size": 256, "num_layers": 3, "dropout": 0.1},
}


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(arch: str, *, hidden_size: int, num_layers: int, dropout: float,
                nhead: int, num_classes: int, input_features: int):
    """Instantiate one comparison arch. All take (batch, 30, 1629) -> logits."""
    if arch == "lstm":
        from model_lstm import ASLKeypointLSTM

        return ASLKeypointLSTM(input_size=input_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes,
                               dropout=dropout)
    if arch == "gru":
        from model_gru import ASLKeypointGRU

        return ASLKeypointGRU(input_size=input_features, hidden_size=hidden_size,
                              num_layers=num_layers, num_classes=num_classes,
                              dropout=dropout)
    if arch == "cnn_lstm":
        from model_cnn_lstm_torch import CNNKeypointLSTM

        return CNNKeypointLSTM(input_size=input_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes,
                               dropout=dropout)
    if arch == "tcn":
        from model_tcn import KeypointTCN

        channels = tuple([hidden_size] * num_layers)
        return KeypointTCN(input_size=input_features, num_classes=num_classes,
                           channels=channels, dropout=dropout)
    if arch == "transformer":
        from model_transformer import Config, SignLanguageTransformer

        config = Config()
        config.input_features = input_features
        config.num_classes = num_classes
        config.d_model = hidden_size
        config.num_encoder_layers = num_layers
        config.nhead = nhead
        config.dropout = dropout
        config.gloss_to_idx = {g: i for i, g in enumerate(VIDEO_LABELS)}
        config.idx_to_gloss = {i: g for g, i in config.gloss_to_idx.items()}
        return SignLanguageTransformer(config)
    raise ValueError(f"Unknown arch '{arch}'. Choices: {ARCHES}")


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _loader(X, y, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _loader(X_train, y_train, True),
        _loader(X_val, y_val, False),
        _loader(X_test, y_test, False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PyTorch video-keypoint trainer")
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--model-name", required=True, help="Registry key, e.g. tcn_v1")
    parser.add_argument("--set-active", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--nhead", type=int, default=8, help="transformer heads")
    args = parser.parse_args()

    defaults = ARCH_DEFAULTS[args.arch]
    lr = args.lr if args.lr is not None else defaults["lr"]
    hidden_size = args.hidden_size or defaults["hidden_size"]
    num_layers = args.num_layers or defaults["num_layers"]
    dropout = args.dropout if args.dropout is not None else defaults["dropout"]

    set_seeds(args.seed)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] arch={args.arch} device={device} seed={args.seed}")
    print(f"[Training] hidden={hidden_size} layers={num_layers} "
          f"dropout={dropout} lr={lr} epochs={args.epochs}")

    print("[Training] Loading unified video dataset (30x1629, deduped)...")
    X_train, y_train, X_val, y_val, X_test, y_test, labels = build_arrays()
    print(f"[Training] train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size
    )

    model = build_model(
        args.arch, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, nhead=args.nhead,
        num_classes=len(labels), input_features=INPUT_FEATURES,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Training] params={params:,}")

    from ml.training.torch_common import (
        evaluate_loader,
        save_history_plot,
        save_reports,
        train_model,
    )

    run_dir = create_run_dir(f"torch_{args.arch}")
    checkpoint = run_dir / "checkpoints" / f"{args.model_name}.pth"

    history = train_model(
        model, train_loader, val_loader, device=device,
        epochs=args.epochs, lr=lr, weight_decay=args.weight_decay,
        patience=args.patience, checkpoint_path=checkpoint,
    )
    save_history_plot(history, run_dir / "plots")
    import json as _json

    (run_dir / "history.json").write_text(
        _json.dumps({k: (v if not isinstance(v, np.generic) else float(v))
                     for k, v in history.items()
                     if isinstance(v, list)}, indent=2),
        encoding="utf-8",
    )

    print("[Training] Test evaluation...")
    test_metrics = evaluate_loader(model, test_loader, device)
    reports = save_reports(
        test_metrics["y_true"], test_metrics["y_pred"], labels, run_dir / "reports"
    )
    print(f"[Training] test acc={reports['accuracy']:.4f} f1={reports['f1_macro']:.4f}")

    # Stable serving path: outputs/models/<model-name>.pth
    serving_path = PROJECT_ROOT / "outputs" / "models" / f"{args.model_name}.pth"
    serving_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, serving_path)

    write_manifest(
        run_dir, model_name=args.model_name, framework="pytorch",
        model_path=serving_path, labels=labels, sequence_length=30,
        input_features=INPUT_FEATURES,
        description=f"PyTorch {args.arch} trained on unified video keypoints",
        extra={
            "arch": args.arch, "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "lr": lr, "seed": args.seed, "params": params,
            "test_accuracy": reports["accuracy"], "test_f1_macro": reports["f1_macro"],
            "best_val_acc": history["best_val_acc"],
            "stopped_epoch": history["stopped_epoch"],
        },
    )
    register_model(
        args.model_name, display_name=f"{args.arch.upper()} (PyTorch, video v2)",
        framework="pytorch", model_path=str(serving_path.relative_to(PROJECT_ROOT)),
        labels=labels, sequence_length=30, input_features=INPUT_FEATURES,
        description=f"PyTorch {args.arch} on unified video keypoints",
        set_active=args.set_active,
        arch=args.arch,
        hyperparams={
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "nhead": args.nhead, "lr": lr,
            "batch_size": args.batch_size, "seed": args.seed,
        },
        metrics={
            "test_accuracy": reports["accuracy"],
            "test_f1_macro": reports["f1_macro"],
            "best_val_acc": float(history["best_val_acc"]),
            "params": params,
        },
    )
    print(f"[Training] Done. Serving artifact: {serving_path}")


if __name__ == "__main__":
    main()
