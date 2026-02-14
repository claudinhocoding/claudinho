"""
Text-to-Speech via Inworld AI
==============================

Cloud TTS using Inworld's API. Falls back to Piper if API fails.
"""

import base64
import logging
import subprocess
from pathlib import Path

import requests

import config

logger = logging.getLogger(__name__)


def synthesize(text: str, language: str = "en") -> str:
    """Synthesize text to speech. Returns path to WAV file."""
    if not text.strip():
        return ""
    
    if len(text) > 500:
        text = text[:497] + "..."
    
    output_path = str(config.TMP_DIR / "response.wav")
    
    try:
        result = _inworld_tts(text, output_path)
        if result:
            return result
    except Exception as e:
        logger.warning(f"Inworld TTS failed: {e}")
    
    logger.info("Falling back to Piper TTS")
    return _piper_tts(text, language, output_path)


def _inworld_tts(text: str, output_path: str) -> str:
    """Call Inworld TTS API. Returns WAV directly (LINEAR16)."""
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
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 16000,
            },
        },
        timeout=15,
    )
    
    if response.status_code != 200:
        logger.error(f"Inworld API error: {response.status_code} {response.text[:200]}")
        return ""
    
    data = response.json()
    audio_b64 = data.get("audioContent", "")
    
    if not audio_b64:
        logger.error(f"No audioContent in response: {list(data.keys())}")
        return ""
    
    audio_bytes = base64.b64decode(audio_b64)
    
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    
    logger.info(f"TTS audio: {len(audio_bytes)} bytes")
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
            [str(config.PIPER_BIN), "--model", str(voice), "--output_file", output_path],
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
