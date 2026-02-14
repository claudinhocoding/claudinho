"""
Text-to-Speech via Inworld AI
==============================

Cloud TTS using Inworld's API. Much better quality than local Piper.
Falls back to Piper if the API call fails.
"""

import base64
import json
import logging
import subprocess
import wave
from pathlib import Path

import requests

import config

logger = logging.getLogger(__name__)


def synthesize(text: str, language: str = "en") -> str:
    """
    Synthesize text to speech using Inworld TTS.
    Returns path to WAV file, or empty string on failure.
    """
    if not text.strip():
        return ""
    
    # Truncate very long responses for TTS
    if len(text) > 500:
        text = text[:497] + "..."
    
    output_path = str(config.TMP_DIR / "response.wav")
    
    try:
        result = _inworld_tts(text, output_path)
        if result:
            return result
    except Exception as e:
        logger.warning(f"Inworld TTS failed: {e}")
    
    # Fallback to Piper
    logger.info("Falling back to Piper TTS")
    return _piper_tts(text, language, output_path)


def _inworld_tts(text: str, output_path: str) -> str:
    """Call Inworld TTS API."""
    logger.info(f"TTS (Inworld): {text[:80]}...")
    
    response = requests.post(
        config.INWORLD_TTS_URL,
        headers={
            "Authorization": f"Basic {config.INWORLD_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "voiceId": config.INWORLD_VOICE_ID,
            "modelId": config.INWORLD_MODEL,
        },
        timeout=15,
    )
    
    if response.status_code != 200:
        logger.error(f"Inworld API error: {response.status_code} {response.text[:200]}")
        return ""
    
    content_type = response.headers.get("Content-Type", "")
    
    if "audio" in content_type:
        # Direct audio bytes
        audio_data = response.content
    elif "json" in content_type:
        # JSON response with base64 audio
        data = response.json()
        # Try common field names
        audio_b64 = (
            data.get("audioContent") or
            data.get("audio") or 
            data.get("audioData") or 
            data.get("data") or 
            data.get("output", {}).get("audio") or
            ""
        )
        if not audio_b64:
            logger.error(f"No audio in response: {list(data.keys())}")
            return ""
        audio_data = base64.b64decode(audio_b64)
    else:
        # Assume raw audio
        audio_data = response.content
    
    if not audio_data:
        return ""
    
    # Save audio — detect format and convert to WAV if needed
    # Check if it's already WAV
    if audio_data[:4] == b"RIFF":
        with open(output_path, "wb") as f:
            f.write(audio_data)
    else:
        # Likely MP3 or other format — save as temp then convert with ffmpeg
        tmp_path = str(config.TMP_DIR / "response_raw.mp3")
        with open(tmp_path, "wb") as f:
            f.write(audio_data)
        
        # Convert to WAV
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", output_path],
                capture_output=True, check=True, timeout=10,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try aplay-compatible format with sox or just save raw
            # If no ffmpeg, try playing mp3 directly
            output_path = tmp_path  # aplay might not handle mp3
            logger.warning("ffmpeg not found, trying raw audio")
    
    logger.info(f"TTS audio saved: {output_path}")
    return output_path


def _piper_tts(text: str, language: str, output_path: str) -> str:
    """Fallback: local Piper TTS."""
    voice = config.PIPER_VOICES.get(language, config.PIPER_VOICES.get("en"))
    if not voice or not voice.exists():
        logger.error(f"No Piper voice for language: {language}")
        return ""
    
    logger.info(f"TTS (Piper) [{language}]: {text[:80]}...")
    
    try:
        proc = subprocess.run(
            [
                str(config.PIPER_BIN),
                "--model", str(voice),
                "--output_file", output_path,
            ],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.error(f"Piper failed: {proc.stderr.decode()[:200]}")
            return ""
        return output_path
    except Exception as e:
        logger.error(f"Piper error: {e}")
        return ""
