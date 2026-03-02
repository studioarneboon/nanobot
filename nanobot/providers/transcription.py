"""Voice transcription providers — local faster-whisper (preferred) with Groq API fallback."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TranscriptionProvider(ABC):
    """Common interface for all speech-to-text backends."""

    @abstractmethod
    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file and return the text."""


# ---------------------------------------------------------------------------
# Local faster-whisper provider
# ---------------------------------------------------------------------------

class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    """Local transcription using faster-whisper (CTranslate2).

    The model is loaded lazily on first transcribe() call so startup isn't
    slowed down.  The actual inference runs in a thread pool via
    ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, model_size: str = "base", language: str | None = "nl"):
        self.model_size = model_size
        self.language = language
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        """Load the Whisper model on first use."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model '{}' ...", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("faster-whisper model '{}' loaded", self.model_size)
        except Exception as e:
            logger.error("Failed to load faster-whisper model: {}", e)
            raise

    def _transcribe_sync(self, path: Path) -> str:
        """Run transcription synchronously (called inside a thread)."""
        self._ensure_model()
        segments, info = self._model.transcribe(
            str(path),
            language=self.language,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if info.language:
            logger.debug("Detected language: {} (prob {:.2f})", info.language, info.language_probability)
        return text

    async def transcribe(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        try:
            text = await asyncio.to_thread(self._transcribe_sync, path)
            if text:
                logger.info("faster-whisper transcribed: {}...", text[:60])
            return text
        except Exception as e:
            logger.error("faster-whisper transcription error: {}", e)
            return ""


# ---------------------------------------------------------------------------
# Groq API provider (fallback)
# ---------------------------------------------------------------------------

class GroqTranscriptionProvider(TranscriptionProvider):
    """Remote transcription via Groq's Whisper API.

    Used as fallback when faster-whisper is not installed.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    response = await client.post(
                        self.api_url, headers=headers, files=files, timeout=60.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    text = data.get("text", "")
                    if text:
                        logger.info("Groq transcribed: {}...", text[:60])
                    return text

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_transcriber(
    stt_model: str = "base",
    stt_language: str | None = "nl",
    stt_provider: str = "auto",
    groq_api_key: str = "",
) -> TranscriptionProvider | None:
    """Create the best available transcription provider.

    Provider selection (when *stt_provider* is ``"auto"``):
      1. faster-whisper (local) — if the ``faster_whisper`` package is installed
      2. Groq API — if a Groq API key is available
      3. ``None`` — no transcription available

    Args:
        stt_model: Whisper model size for faster-whisper (tiny/base/small/medium/large).
        stt_language: Language hint (ISO 639-1), e.g. ``"nl"``. ``None`` = auto-detect.
        stt_provider: ``"auto"``, ``"faster-whisper"``, or ``"groq"``.
        groq_api_key: Groq API key (only needed for Groq provider).

    Returns:
        A :class:`TranscriptionProvider` instance, or ``None`` if nothing is available.
    """
    if stt_provider in ("auto", "faster-whisper"):
        try:
            import faster_whisper as _fw  # noqa: F401

            logger.info(
                "Using faster-whisper for STT (model={}, language={})",
                stt_model,
                stt_language or "auto",
            )
            return FasterWhisperTranscriptionProvider(model_size=stt_model, language=stt_language)
        except ImportError:
            if stt_provider == "faster-whisper":
                logger.error("faster-whisper requested but not installed")
                return None
            logger.debug("faster-whisper not installed, trying Groq fallback")

    if stt_provider in ("auto", "groq"):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if api_key:
            logger.info("Using Groq API for STT")
            return GroqTranscriptionProvider(api_key=api_key)
        if stt_provider == "groq":
            logger.error("Groq STT requested but no API key available")
            return None

    logger.warning("No transcription provider available")
    return None
