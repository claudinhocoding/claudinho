"""
Speech to Text
==============

Uses Whisper.cpp for local speech recognition.
Returns transcribed text + detected language.
"""

import logging
import subprocess
import re
from typing import Tuple

import config

logger = logging.getLogger(__name__)


def transcribe(wav_path: str) -> Tuple[str, str]:
    """
    Transcribe audio to text using Whisper.cpp.
    
    Args:
        wav_path: Path to WAV file.
        
    Returns:
        Tuple of (transcribed_text, detected_language).
        Language is 2-letter code: "en", "pt", etc.
    """
    result = subprocess.run(
        [
            str(config.WHISPER_CLI),
            "-m", str(config.WHISPER_MODEL),
            "-l", "auto",           # auto language detection
            "-f", wav_path,
            "--no-timestamps",
            "-t", "4",              # 4 threads (Pi 5 = 4 cores)
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        logger.error(f"Whisper error: {result.stderr}")
        return "", config.DEFAULT_LANGUAGE
    
    # Parse language from stderr (whisper outputs processing info there)
    language = _detect_language(result.stderr)
    
    # Parse transcription from stdout
    text = _parse_transcription(result.stdout)
    
    logger.info(f"STT: [{language}] {text}")
    return text, language


def _detect_language(stderr: str) -> str:
    """Extract detected language from Whisper.cpp stderr output."""
    # Whisper outputs something like: "auto-detected language: en (p = 0.97)"
    match = re.search(r"auto-detected language:\s*(\w+)", stderr)
    if match:
        return match.group(1)
    
    # Fallback: check the "lang = XX" in processing line
    match = re.search(r"lang\s*=\s*(\w+)", stderr)
    if match:
        lang = match.group(1)
        if lang != "auto":
            return lang
    
    return config.DEFAULT_LANGUAGE


def _parse_transcription(stdout: str) -> str:
    """Extract transcription text from Whisper.cpp stdout."""
    lines = stdout.strip().split("\n")
    
    # Filter out processing info lines (start with [ or are empty)
    text_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("[") and not line.startswith("whisper_"):
            text_lines.append(line)
    
    text = " ".join(text_lines).strip()
    
    # Clean up common Whisper artifacts
    text = re.sub(r"\(.*?\)", "", text)       # Remove (parenthetical notes)
    text = re.sub(r"\[.*?\]", "", text)       # Remove [bracketed notes]
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    
    return text
