"""
Text to Speech
==============

Uses Piper TTS for local speech synthesis.
Switches voice based on detected language (EN/PT).
"""

import logging
import subprocess
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def synthesize(text: str, language: str = "en", output_path: str = None) -> str:
    """
    Synthesize speech from text using Piper.
    
    Automatically selects voice based on language.
    
    Args:
        text: Text to speak.
        language: 2-letter language code ("en", "pt").
        output_path: Output WAV path. None = auto-generate.
        
    Returns:
        Path to generated WAV file.
    """
    if not text.strip():
        logger.warning("Empty text, skipping TTS")
        return ""
    
    # Select voice based on language
    voice_path = config.PIPER_VOICES.get(language, config.PIPER_VOICES["en"])
    
    if not voice_path.exists():
        logger.error(f"Voice model not found: {voice_path}")
        # Fallback to any available voice
        for lang, path in config.PIPER_VOICES.items():
            if path.exists():
                voice_path = path
                logger.info(f"Falling back to {lang} voice")
                break
        else:
            raise FileNotFoundError("No Piper voice models found!")
    
    if output_path is None:
        output_path = str(config.TMP_DIR / "response.wav")
    
    logger.info(f"TTS [{language}]: {text[:80]}...")
    
    result = subprocess.run(
        [
            str(config.PIPER_BIN),
            "--model", str(voice_path),
            "--output_file", output_path,
        ],
        input=text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode != 0:
        logger.error(f"Piper error: {result.stderr}")
        return ""
    
    return output_path
