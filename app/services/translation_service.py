from __future__ import annotations

from threading import Lock

from deep_translator import GoogleTranslator

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


def _make_translator(target_lang: str):
    """Lazy-import deep_translator so Flask can boot without it installed."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise ImportError(
            "deep-translator is required for translation. "
            "Install it with: pip install deep-translator"
        ) from exc
    return GoogleTranslator(source="auto", target=target_lang)


class TranslationService:
    def __init__(self) -> None:
        self._target_lang = settings.translation_target_language
        try:
            self._translator = _make_translator(self._target_lang)
        except ImportError as exc:
            logger.warning("Translation disabled: %s", exc)
            self._translator = None
        self._cache: dict[str, str] = {}
        self._lock = Lock()

    def translate(self, text: str, target_language: str | None = None) -> str:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text is required for translation")

        language = (target_language or self._target_lang).strip().lower()
        cache_key = f"{language}:{clean_text}"

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                return cached

        try:
            if self._translator is None:
                raise RuntimeError("translator unavailable")
            translator = (
                self._translator
                if language == self._target_lang
                else _make_translator(language)
            )
            translated = translator.translate(clean_text)
            if not translated:
                logger.warning("Translation returned empty output for text='%s'", clean_text)
                return clean_text

            with self._lock:
                self._cache[cache_key] = translated

            return translated
        except Exception as exc:
            logger.warning("Translation failed for text='%s': %s", clean_text, exc)
            return clean_text

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)
