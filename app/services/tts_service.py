from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.config.logging import get_logger
from app.config.settings import get_settings
from app.services.translation_service import TranslationService


logger = get_logger(__name__)
settings = get_settings()


def _load_tts_helper():
    """Lazy-import legacy TTSHelper so Flask can boot without pyttsx3."""
    try:
        from tts_helper import TTSHelper
    except ImportError as exc:
        raise ImportError(
            "pyttsx3 is required for TTS. Install it with: pip install pyttsx3"
        ) from exc
    return TTSHelper


class TTSService:
    def __init__(self, translation_service: TranslationService | None = None) -> None:
        self._translation_service = translation_service or TranslationService()
        self._audio_dir = Path(settings.audio_output_dir)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            helper_cls = _load_tts_helper()
            self._tts_helper = helper_cls(
                audio_dir=str(self._audio_dir),
                target_lang=settings.translation_target_language,
            )
        except ImportError as exc:
            logger.warning("TTS disabled: %s", exc)
            self._tts_helper = None

    def _build_file_name(self, text: str, language: str, extension: str = "mp3") -> str:
        safe_text = "_".join(text.strip().split())[:50] or "speech"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{safe_text}_{language}_{timestamp}.{extension}"

    def generate_audio(self, text: str, language: str = "en") -> str | None:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text is required for TTS")

        if not settings.tts_enabled:
            logger.info("TTS is disabled by configuration")
            return None

        lang = language.strip().lower()
        if lang in {"hi", "hindi"}:
            speech_text = self._translation_service.translate(clean_text, "hindi")
            lang_code = "hi"
        else:
            speech_text = clean_text
            lang_code = "en"

        file_name = self._build_file_name(clean_text, lang_code)
        output_path = self._audio_dir / file_name

        try:
            if self._tts_helper is None or self._tts_helper.engine is None:
                logger.warning("TTS engine unavailable, skipping audio generation")
                return None
            # Reuse pyttsx3 engine from helper to avoid repeated engine initialization.
            self._tts_helper.engine.save_to_file(speech_text, str(output_path))
            self._tts_helper.engine.runAndWait()
            logger.info("Generated %s audio at %s", lang_code, output_path)
            return file_name
        except Exception as exc:
            logger.error("TTS generation failed for text='%s': %s", clean_text, exc)
            return None

    def generate_bilingual_audio(self, text: str) -> dict[str, str | None]:
        english_file = self.generate_audio(text, "en")
        hindi_file = self.generate_audio(text, "hi")
        return {
            "en": english_file,
            "hi": hindi_file,
        }

    def shutdown(self) -> None:
        try:
            if self._tts_helper is not None:
                self._tts_helper.shutdown()
        except Exception:
            logger.warning("TTS shutdown encountered an issue", exc_info=True)
