from __future__ import annotations

from typing import Any

from app.config.logging import get_logger

from .health_controller import register_health_controller
from .inference_controller import register_inference_controller
from .media_controller import register_media_controller


logger = get_logger(__name__)


def register_controllers(flask_app: Any) -> None:
    register_health_controller(flask_app)
    register_media_controller(flask_app)
    register_inference_controller(flask_app)
    logger.info("All controllers registered")
