from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template

if __package__ in {None, ""}:
    # Support direct script execution: python app/main.py
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.controllers import register_controllers


def create_app() -> Flask:
    settings = get_settings()
    configure_logging()

    app = Flask(
        __name__,
        template_folder=settings.templates_dir,
        static_folder=settings.static_dir,
    )
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_size_mb * 1024 * 1024

    register_controllers(app)

    logger = get_logger(__name__)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            app_name=settings.app_name,
            max_seq_length=settings.max_seq_length,
        )

    @app.get("/routes")
    def routes() -> tuple[object, int]:
        """Debug endpoint — lists all available API routes."""
        payload = {
            "app": settings.app_name,
            "environment": settings.environment,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routes": {
                "ui": "/",
                "health": "/health",
                "liveness": "/live",
                "readiness": "/ready",
                "media": {
                    "health": "/media/health",
                    "frame": "/media/frame",
                    "video_feed": "/media/video_feed",
                },
                "inference": {
                    "labels": "/inference/labels",
                    "predict": "/inference/predict",
                    "audio": "/inference/audio/<file_name>",
                    "models": "/inference/models",
                    "active_model": "/inference/models/active",
                },
            },
        }
        return jsonify(payload), 200

    logger.info(
        "Application created | env=%s | host=%s | port=%s",
        settings.environment,
        settings.host,
        settings.port,
    )
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
