from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    debug: bool

    host: str
    port: int

    model_framework: str
    model_path: str
    label_map_path: str
    max_seq_length: int
    prediction_threshold: float

    camera_index: int

    tts_enabled: bool
    audio_output_dir: str
    default_tts_language: str
    translation_target_language: str

    logs_dir: str
    log_level: str
    log_file: str

    max_upload_size_mb: int

    templates_dir: str
    static_dir: str

    @staticmethod
    def from_env() -> "Settings":
        environment = os.getenv("APP_ENV", "development")
        logs_dir = os.getenv("LOGS_DIR", str(BASE_DIR / "logs"))

        settings = Settings(
            app_name=os.getenv("APP_NAME", "Sign2Speak"),
            environment=environment,
            debug=_get_bool("DEBUG", environment == "development"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_get_int("PORT", 8000),
            model_framework=os.getenv("MODEL_FRAMEWORK", "pytorch"),
            model_path=os.getenv("MODEL_PATH", str(BASE_DIR / "outputs" / "models" / "model_transformer.pth")),
            label_map_path=os.getenv("LABEL_MAP_PATH", str(BASE_DIR / "outputs" / "gloss_mapping.json")),
            max_seq_length=_get_int("MAX_SEQ_LENGTH", 30),
            prediction_threshold=_get_float("PREDICTION_THRESHOLD", 0.7),
            camera_index=_get_int("CAMERA_INDEX", 0),
            tts_enabled=_get_bool("TTS_ENABLED", True),
            audio_output_dir=os.getenv("AUDIO_OUTPUT_DIR", str(BASE_DIR / "audio")),
            default_tts_language=os.getenv("DEFAULT_TTS_LANGUAGE", "en"),
            translation_target_language=os.getenv("TRANSLATION_TARGET_LANGUAGE", "hindi"),
            logs_dir=logs_dir,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=os.getenv("LOG_FILE", str(Path(logs_dir) / "app.log")),
            max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", 25),
            templates_dir=os.getenv("TEMPLATES_DIR", str(BASE_DIR / "app" / "views" / "templates")),
            static_dir=os.getenv("STATIC_DIR", str(BASE_DIR / "app" / "views" / "static")),
        )

        settings.ensure_runtime_dirs()
        return settings

    def ensure_runtime_dirs(self) -> None:
        Path(self.audio_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
