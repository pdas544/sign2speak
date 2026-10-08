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
from ml.training.arch_factory import ARCHES, ARCH_DEFAULTS, build_model  # noqa: E402
from ml.training.datasets.video_keypoints import (  # noqa: E402
    INPUT_FEATURES,
    NOFACE_FEATURES,
    VIDEO_LABELS,
    build_arrays,
)


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    parser.add_argument("--augment-copies", type=int, default=0,
                        help="train-only augmented replicas per clip (0 = off)")
    parser.add_argument("--sampler", choices=("last", "uniform"), default="last",
                        help="last = most-recent 30 frames; uniform = 30 evenly spaced")
    parser.add_argument("--mask-face", action="store_true",
                        help="drop face block -> pose+hands features only")
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
          f"dropout={dropout} lr={lr} epochs={args.epochs} "
          f"augment_copies={args.augment_copies} sampler={args.sampler} "
          f"mask_face={args.mask_face}")

    features = NOFACE_FEATURES if args.mask_face else INPUT_FEATURES
    print("[Training] Loading unified video dataset "
          f"(30x{features}, deduped, sampler={args.sampler})...")
    X_train, y_train, X_val, y_val, X_test, y_test, labels = build_arrays(
        sampler=args.sampler, mask_face=args.mask_face,
    )
    print(f"[Training] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    if args.augment_copies > 0:
        from ml.preprocessing.augment import augment_sequence

        rng = np.random.default_rng(args.seed)
        aug_X = [X_train]
        aug_y = [y_train]
        for _ in range(args.augment_copies):
            aug_X.append(np.stack(
                [augment_sequence(seq, rng=rng) for seq in X_train]
            ).astype(np.float32))
            aug_y.append(y_train.copy())
        X_train = np.concatenate(aug_X, axis=0)
        y_train = np.concatenate(aug_y, axis=0)
        print(f"[Training] augmented train={X_train.shape} "
              f"({args.augment_copies} copies, train-only)")

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size
    )

    model = build_model(
        args.arch, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, nhead=args.nhead,
        num_classes=len(labels), input_features=features,
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
        input_features=features,
        description=f"PyTorch {args.arch} trained on unified video keypoints",
        extra={
            "arch": args.arch, "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "lr": lr, "seed": args.seed, "params": params,
            "augment_copies": args.augment_copies, "sampler": args.sampler,
            "mask_face": args.mask_face,
            "augment": {"rotation_deg": 7.0, "translate": 0.05, "hand_swap_prob": 0.5},
            "test_accuracy": reports["accuracy"], "test_f1_macro": reports["f1_macro"],
            "best_val_acc": history["best_val_acc"],
            "stopped_epoch": history["stopped_epoch"],
        },
    )
    # Unique, informative display name (registry keys alone are cryptic and
    # arch-only names collide across runs — see the 7× "TCN (PyTorch, video v2)" incident).
    _tags = []
    if args.sampler == "uniform":
        _tags.append("uniform")
    if args.mask_face:
        _tags.append("noface")
    if args.augment_copies:
        _tags.append(f"augx{args.augment_copies}")
    _tags.append(f"h{hidden_size}x{num_layers}")
    display_name = f"{args.arch.upper()} · {args.model_name} · {'+'.join(_tags)}"

    register_model(
        args.model_name, display_name=display_name,
        framework="pytorch", model_path=str(serving_path.relative_to(PROJECT_ROOT)),
        labels=labels, sequence_length=30, input_features=features,
        description=f"PyTorch {args.arch} on unified video keypoints",
        set_active=args.set_active,
        arch=args.arch,
        hyperparams={
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "nhead": args.nhead, "lr": lr,
            "batch_size": args.batch_size, "seed": args.seed,
            "augment_copies": args.augment_copies, "sampler": args.sampler,
            "mask_face": args.mask_face,
            "augment": {"rotation_deg": 7.0, "translate": 0.05, "hand_swap_prob": 0.5},
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
