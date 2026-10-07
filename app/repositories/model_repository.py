from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


class ModelRepository:
    def __init__(
        self,
        model_path: str | Path | None = None,
        label_map_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else Path(settings.model_path)
        self._label_map_path = (
            Path(label_map_path) if label_map_path else Path(settings.label_map_path)
        )

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def label_map_path(self) -> Path:
        return self._label_map_path

    def model_exists(self) -> bool:
        return self._model_path.exists()

    def label_map_exists(self) -> bool:
        return self._label_map_path.exists()

    def load_state_dict(self, device: torch.device | str | None = None) -> dict[str, Any]:
        if not self.model_exists():
            raise FileNotFoundError(f"Model file not found at {self._model_path}")

        map_location = device if device is not None else "cpu"
        state_dict = torch.load(self._model_path, map_location=map_location)
        logger.info("Loaded model state dict from %s", self._model_path)
        return state_dict

    def load_label_map(self) -> dict[str, Any]:
        if not self.label_map_exists():
            raise FileNotFoundError(f"Label map file not found at {self._label_map_path}")

        with self._label_map_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Label map must be a JSON object")

        logger.info("Loaded label map from %s", self._label_map_path)
        return data

    def save_label_map(self, label_map: dict[str, Any]) -> Path:
        self._label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with self._label_map_path.open("w", encoding="utf-8") as handle:
            json.dump(label_map, handle, indent=2, ensure_ascii=False)

        logger.info("Saved label map to %s", self._label_map_path)
        return self._label_map_path
