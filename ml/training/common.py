"""
ml/training/common.py
─────────────────────
Shared utilities for all training scripts:
  - Deterministic run-id generation
  - Per-run output directory scaffolding (checkpoints, plots, reports)
  - Model manifest writing / reading
  - Automatic model registry update after a successful training run
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parents[2]      # repo root
OUTPUTS_DIR = BASE_DIR / "outputs" / "training"
REGISTRY_PATH = BASE_DIR / "models" / "registry" / "registry.json"


# --------------------------------------------------------------------------- #
# Run-directory factory                                                        #
# --------------------------------------------------------------------------- #

def create_run_dir(model_family: str) -> Path:
    """
    Create and return a timestamped run directory under outputs/training/<model_family>/.

    Structure:
        outputs/training/<model_family>/<run_id>/
            checkpoints/
            plots/
            reports/
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / model_family / run_id

    for sub in ("checkpoints", "plots", "reports"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[Training] Run directory: {run_dir}")
    return run_dir


# --------------------------------------------------------------------------- #
# Manifest                                                                     #
# --------------------------------------------------------------------------- #

def _environment_provenance() -> dict[str, Any]:
    """Best-effort record of the training environment (versions never fail the run)."""
    import platform

    info: dict[str, Any] = {"python": platform.python_version()}
    for pkg in ("torch", "tensorflow", "numpy", "sklearn"):
        try:
            module = __import__(pkg)
            info[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    try:
        import subprocess

        info["git_sha"] = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            ).stdout.strip() or None
        )
    except Exception:
        info["git_sha"] = None
    return info


def write_manifest(
    run_dir: Path,
    *,
    model_name: str,
    framework: str,
    model_path: str | Path,
    labels: list[str],
    sequence_length: int,
    input_features: int,
    description: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest.json into the run directory."""
    manifest: dict[str, Any] = {
        "model_name": model_name,
        "framework": framework,
        "model_path": str(model_path),
        "labels": labels,
        "sequence_length": sequence_length,
        "input_features": input_features,
        "description": description,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment_provenance(),
    }
    if extra:
        manifest.update(extra)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Training] Manifest written: {manifest_path}")
    return manifest_path


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Registry helpers                                                             #
# --------------------------------------------------------------------------- #

def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"active_model": None, "models": {}}
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _save_registry(registry: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def register_model(
    model_name: str,
    *,
    display_name: str,
    framework: str,
    model_path: str | Path,
    labels: list[str],
    sequence_length: int,
    input_features: int,
    description: str = "",
    set_active: bool = False,
    arch: str | None = None,
    hyperparams: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """
    Add or overwrite a model entry in models/registry/registry.json.

    Args:
        model_name:  Registry key (e.g. "cnn_lstm_v2").
        set_active:  If True, also update the "active_model" key.
        arch:        Architecture key for the generic torch backend
                     (lstm|gru|cnn_lstm|tcn|transformer). None = legacy entry.
        hyperparams: Arch hyperparameters needed to rebuild the model
                     before loading the state dict.
        metrics:     Latest known eval metrics, e.g.
                     {"test_accuracy": 0.86, "test_f1_macro": 0.85,
                      "eval_run_id": "20260928_120000", "evaluated_at": ...}.
    """
    registry = _load_registry()

    registry["models"][model_name] = {
        "display_name": display_name,
        "framework": framework,
        "model_path": str(model_path),
        "labels": labels,
        "sequence_length": sequence_length,
        "input_features": input_features,
        "description": description,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if arch is not None:
        registry["models"][model_name]["arch"] = arch
    if hyperparams is not None:
        registry["models"][model_name]["hyperparams"] = hyperparams
    if metrics is not None:
        registry["models"][model_name]["metrics"] = metrics

    if set_active or registry.get("active_model") is None:
        registry["active_model"] = model_name

    _save_registry(registry)
    print(f"[Registry] '{model_name}' registered (active={registry['active_model']})")


def update_model_metrics(model_name: str, metrics: dict[str, Any]) -> None:
    """Write back latest eval metrics onto an existing registry entry."""
    registry = _load_registry()
    if model_name not in registry.get("models", {}):
        raise KeyError(f"Model '{model_name}' not found in registry")
    entry = registry["models"][model_name]
    entry_metrics = dict(entry.get("metrics") or {})
    entry_metrics.update(metrics)
    entry["metrics"] = entry_metrics
    _save_registry(registry)
    print(f"[Registry] metrics updated for '{model_name}': {sorted(entry_metrics)}")
