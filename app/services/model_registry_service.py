"""
app/services/model_registry_service.py
────────────────────────────────────────
Manages the model registry stored at models/registry/registry.json.

Responsibilities:
  - Load/reload the registry from disk
  - Return the active model configuration
  - List all registered models
  - Switch the active model (also clears the cached backend in InferenceService)
  - Instantiate and return the correct InferenceBackend for a given config
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.config.logging import get_logger
from app.config.settings import get_settings
from app.services.inference_backends.base import InferenceBackend


logger = get_logger(__name__)
settings = get_settings()


class ModelRegistryService:
    def __init__(self, registry_path: str | Path | None = None) -> None:
        self._path = Path(registry_path) if registry_path else Path(settings.model_registry_path)
        self._lock = Lock()
        self._registry: dict[str, Any] = {}
        self._reload()

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _reload(self) -> None:
        """Load or re-load the registry from disk."""
        if not self._path.exists():
            logger.warning("Registry file not found at %s — using empty registry", self._path)
            self._registry = {"active_model": None, "models": {}}
            return

        with self._path.open(encoding="utf-8") as fh:
            self._registry = json.load(fh)
        logger.info(
            "Registry loaded: %d models, active=%s",
            len(self._registry.get("models", {})),
            self._registry.get("active_model"),
        )

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[dict[str, Any]]:
        """Return all registered model entries as a list."""
        models = self._registry.get("models", {})
        active = self._registry.get("active_model")
        return [
            {**entry, "name": name, "is_active": name == active}
            for name, entry in models.items()
        ]

    def active_model_name(self) -> str | None:
        """Return the key of the currently active model."""
        # Prefer runtime env override
        override = settings.model_name
        if override and override in self._registry.get("models", {}):
            return override
        return self._registry.get("active_model")

    def active_model_config(self) -> dict[str, Any]:
        """
        Return the registry entry dict for the currently active model.

        Raises:
            RuntimeError: if no active model is configured.
            KeyError: if the active model name is not found in the registry.
        """
        name = self.active_model_name()
        if not name:
            raise RuntimeError(
                "No active model is configured. "
                "Set MODEL_NAME env var or update models/registry/registry.json"
            )
        models = self._registry.get("models", {})
        if name not in models:
            raise KeyError(f"Active model '{name}' not found in registry")
        return dict(models[name])

    def get_model_config(self, name: str) -> dict[str, Any]:
        """Return the config for a specific named model."""
        models = self._registry.get("models", {})
        if name not in models:
            raise KeyError(f"Model '{name}' not found in registry")
        return dict(models[name])

    def set_active_model(self, name: str) -> None:
        """Persist a new active-model choice to the registry file."""
        with self._lock:
            self._reload()
            if name not in self._registry.get("models", {}):
                raise KeyError(f"Model '{name}' not found in registry")
            self._registry["active_model"] = name
            self._save()
            logger.info("Active model changed to '%s'", name)

    # ------------------------------------------------------------------ #
    # Backend factory                                                      #
    # ------------------------------------------------------------------ #

    def build_backend(self, model_config: dict[str, Any]) -> InferenceBackend:
        """
        Instantiate and load the correct backend for a given model config entry.

        Args:
            model_config: A registry entry dict (must contain 'framework').

        Returns:
            A loaded InferenceBackend ready to run predict().
        """
        from app.services.inference_backends.tf_backend import TFBackend
        from app.services.inference_backends.torch_backend import TorchBackend

        framework = model_config.get("framework", "").lower()

        if framework == "tf":
            backend: InferenceBackend = TFBackend()
        elif framework in ("pytorch", "torch"):
            backend = TorchBackend()
        else:
            raise ValueError(
                f"Unknown model framework '{framework}'. "
                "Supported values: 'tf', 'pytorch'"
            )

        backend.load(model_config)
        return backend
