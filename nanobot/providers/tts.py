"""Text-to-speech providers — KittenTTS (local, lightweight)."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TTSProvider(ABC):
    """Common interface for all text-to-speech backends."""

    @abstractmethod
    async def speak(self, text: str, output_path: str | Path) -> str | None:
        """Generate speech from text and save to file.
        
        Args:
            text: Text to speak.
            output_path: Path to save audio file.
            
        Returns:
            Path to generated audio file, or None on failure.
        """


# ---------------------------------------------------------------------------
# KittenTTS provider (local)
# ---------------------------------------------------------------------------

class KittenTTSProvider(TTSProvider):
    """Local TTS using KittenTTS (<25MB, CPU-friendly).
    
    Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    All English voices.
    """

    def __init__(self, voice: str = "Luna"):
        self.voice = voice
        self._model = None

    def _ensure_model(self):
        """Load the KittenTTS model on first use."""
        if self._model is not None:
            return
        try:
            from kittentts import KittenTTS
            logger.info("Loading KittenTTS model (voice={})...", self.voice)
            self._model = KittenTTS()
            logger.info("KittenTTS loaded with voice: {}", self.voice)
        except Exception as e:
            logger.error("Failed to load KittenTTS: {}", e)
            raise

    def _speak_sync(self, text: str, output_path: Path) -> str | None:
        """Generate speech synchronously (called inside a thread)."""
        self._ensure_model()
        try:
            self._model.generate_to_file(text, str(output_path), voice=self.voice)
            logger.debug("KittenTTS generated: {}", output_path)
            return str(output_path)
        except Exception as e:
            logger.error("KittenTTS generation error: {}", e)
            return None

    async def speak(self, text: str, output_path: str | Path) -> str | None:
        output_path = Path(output_path)
        try:
            result = await asyncio.to_thread(self._speak_sync, text, output_path)
            if result:
                logger.info("KittenTTS spoken: {}...", text[:50])
            return result
        except Exception as e:
            logger.error("KittenTTS speak error: {}", e)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tts_provider(
    tts_enabled: bool = False,
    tts_voice: str = "Luna",
) -> TTSProvider | None:
    """Create TTS provider if available.
    
    Args:
        tts_enabled: Whether TTS is enabled.
        tts_voice: Voice to use (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo).
        
    Returns:
        A :class:`TTSProvider` instance, or None if not available/disabled.
    """
    if not tts_enabled:
        return None
    
    try:
        import kittentts as _kt  # noqa: F401
        logger.info("Using KittenTTS for TTS (voice={})", tts_voice)
        return KittenTTSProvider(voice=tts_voice)
    except ImportError:
        logger.warning("KittenTTS not installed, TTS unavailable")
        return None
