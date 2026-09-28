from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()

health_bp = Blueprint("health", __name__)


def _dependency_status() -> dict[str, Any]:
    """Check status of all dependencies, including active model from registry."""
    audio_dir_exists = Path(settings.audio_output_dir).exists()
    logs_dir_exists = Path(settings.logs_dir).exists()

    # Load active model config from registry
    model_file_ok = False
    model_file_path = "unknown"
    
    try:
        from app.services.model_registry_service import ModelRegistryService
        registry = ModelRegistryService()
        model_config = registry.active_model_config()
        if model_config:
            model_file_path = model_config.get("model_path", "unknown")
            # Resolve relative to workspace root
            workspace_root = Path(settings.model_registry_path).parent.parent.parent
            full_path = workspace_root / model_file_path
            model_file_ok = full_path.exists()
    except Exception as exc:
        logger.warning("Failed to load active model config: %s", exc)

    dependencies = {
        "model_file": {
            "ok": model_file_ok,
            "path": model_file_path,
        },
        "audio_directory": {
            "ok": audio_dir_exists,
            "path": settings.audio_output_dir,
        },
        "logs_directory": {
            "ok": logs_dir_exists,
            "path": settings.logs_dir,
        },
    }

    return dependencies


@health_bp.get("/health")
def health_check() -> tuple[Response, int]:
    dependencies = _dependency_status()
    all_ok = all(item["ok"] for item in dependencies.values())

    payload = {
        "status": "ok" if all_ok else "degraded",
        "app": settings.app_name,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies,
    }

    return jsonify(payload), 200 if all_ok else 503


@health_bp.get("/ready")
def readiness_check() -> tuple[Response, int]:
    dependencies = _dependency_status()
    model_ready = dependencies["model_file"]["ok"]

    payload = {
        "ready": model_ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return jsonify(payload), 200 if model_ready else 503


@health_bp.get("/live")
def liveness_check() -> tuple[Response, int]:
    payload = {
        "live": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), 200


def register_health_controller(flask_app: Any) -> None:
    flask_app.register_blueprint(health_bp)
    logger.info("Health controller registered")
