from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


class AudioRepository:
    def __init__(self, audio_dir: str | Path | None = None) -> None:
        self._audio_dir = Path(audio_dir) if audio_dir else Path(settings.audio_output_dir)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def audio_dir(self) -> Path:
        return self._audio_dir

    def path_for(self, file_name: str) -> Path:
        return self._audio_dir / file_name

    def exists(self, file_name: str) -> bool:
        return self.path_for(file_name).exists()

    def build_file_name(
        self,
        base_text: str,
        language: str,
        extension: str = "mp3",
    ) -> str:
        safe_text = "_".join((base_text or "speech").strip().split())[:50] or "speech"
        safe_lang = (language or "en").strip().lower()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = extension.lstrip(".")
        return f"{safe_text}_{safe_lang}_{ts}.{ext}"

    def list_files(self, suffix: str | None = None) -> list[str]:
        if suffix:
            pattern = f"*.{suffix.lstrip('.')}"
            items = sorted(self._audio_dir.glob(pattern), key=lambda p: p.name)
        else:
            items = sorted(self._audio_dir.iterdir(), key=lambda p: p.name)

        return [item.name for item in items if item.is_file()]

    def delete(self, file_name: str) -> bool:
        path = self.path_for(file_name)
        if not path.exists():
            return False

        path.unlink()
        logger.info("Deleted audio file %s", path)
        return True
