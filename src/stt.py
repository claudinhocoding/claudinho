"""
Speech to Text
==============

Uses Groq Whisper API (cloud) for fast transcription (~0.3s).
Falls back to local Whisper.cpp if Groq is unavailable.

Returns transcribed text + detected language.
"""

import logging
import subprocess
import re
from typing import Tuple

import requests

import config

logger = logging.getLogger(__name__)


def transcribe(wav_path: str) -> Tuple[str, str]:
    """
    Transcribe audio to text.
    Tries Groq cloud first (fast), falls back to local Whisper.cpp.

    Returns:
        Tuple of (transcribed_text, detected_language).
        Language is 2-letter code: "en", "pt", etc.
    """
    # Try Groq cloud STT first
    groq_key = getattr(config, 'GROQ_API_KEY', None)
    if groq_key:
        try:
            text, lang = _groq_transcribe(wav_path, groq_key)
            if text.strip():
                return text, lang
        except Exception as e:
            logger.warning(f"Groq STT failed, falling back to local: {e}")

    # Fallback to local Whisper.cpp
    return _whisper_transcribe(wav_path)


def _groq_transcribe(wav_path: str, api_key: str) -> Tuple[str, str]:
    """
    Transcribe via Groq Whisper API.
    ~0.3s for short audio clips (228x real-time).
    """
    logger.info("STT (Groq cloud)...")

    with open(wav_path, "rb") as f:
        response = requests.post(
            config.GROQ_STT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            files={
                "file": ("audio.wav", f, "audio/wav"),
            },
            data={
                "model": config.GROQ_STT_MODEL,
                "response_format": "verbose_json",
            },
            timeout=15,
        )

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:200]}")

    data = response.json()
    text = data.get("text", "").strip()
    language = data.get("language", config.DEFAULT_LANGUAGE)

    # Groq returns full language names ("English", "Portuguese") — normalize to 2-letter codes
    language = _normalize_language(language)

    # Clean up common Whisper artifacts
    text = _clean_transcription(text)

    logger.info(f"STT: [{language}] {text}")
    return text, language


def _whisper_transcribe(wav_path: str) -> Tuple[str, str]:
    """
    Transcribe via local Whisper.cpp.
    Slower (~3-4s on Pi 5) but works offline.
    """
    logger.info("STT (local Whisper)...")

    result = subprocess.run(
        [
            str(config.WHISPER_CLI),
            "-m", str(config.WHISPER_MODEL),
            "-l", "auto",
            "-f", wav_path,
            "--no-timestamps",
            "-t", "4",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        logger.error(f"Whisper error: {result.stderr}")
        return "", config.DEFAULT_LANGUAGE

    language = _detect_language(result.stderr)
    text = _parse_transcription(result.stdout)

    logger.info(f"STT: [{language}] {text}")
    return text, language


def _detect_language(stderr: str) -> str:
    """Extract detected language from Whisper.cpp stderr."""
    match = re.search(r"auto-detected language:\s*(\w+)", stderr)
    if match:
        return match.group(1)
    match = re.search(r"lang\s*=\s*(\w+)", stderr)
    if match and match.group(1) != "auto":
        return match.group(1)
    return config.DEFAULT_LANGUAGE


def _parse_transcription(stdout: str) -> str:
    """Extract transcription text from Whisper.cpp stdout."""
    lines = stdout.strip().split("\n")
    text_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("[") and not line.startswith("whisper_"):
            text_lines.append(line)
    text = " ".join(text_lines).strip()
    return _clean_transcription(text)


LANGUAGE_MAP = {
    "english": "en", "portuguese": "pt", "spanish": "es",
    "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "chinese": "zh", "korean": "ko",
    "dutch": "nl", "russian": "ru", "arabic": "ar",
}


def _normalize_language(lang: str) -> str:
    """Normalize language to 2-letter code (Groq returns full names)."""
    if len(lang) <= 3:
        return lang.lower()
    return LANGUAGE_MAP.get(lang.lower(), lang[:2].lower())


def _clean_transcription(text: str) -> str:
    """Remove common Whisper artifacts."""
    text = re.sub(r"\(.*?\)", "", text)       # Remove (parenthetical notes)
    text = re.sub(r"\[.*?\]", "", text)       # Remove [bracketed notes]
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text
