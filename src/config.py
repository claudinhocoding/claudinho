"""
Configuration
=============

All hardware-specific paths and settings in one place.
Edit this file to match your Pi setup.
"""

from pathlib import Path

# ── Audio Devices (ALSA) ──────────────────────────────────────
MIC_DEVICE = "plughw:0,0"       # USB PnP Sound Device (card 0)
SPEAKER_DEVICE = "plughw:3,0"   # UACDemoV1.0 (card 3)
SAMPLE_RATE = 16000
CHANNELS = 1

# ── Whisper.cpp (STT) ────────────────────────────────────────
WHISPER_CLI = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-base.bin"

# ── Piper TTS ─────────────────────────────────────────────────
PIPER_BIN = Path.home() / "piper" / "piper"
PIPER_VOICES = {
    "en": Path.home() / "piper" / "en_US-norman-medium.onnx",
    "pt": Path.home() / "piper" / "pt_BR-edresson-low.onnx",
}
DEFAULT_LANGUAGE = "en"

# ── Wake Word ─────────────────────────────────────────────────
# Wake word model: file path to .onnx model
# Community models work directly with openWakeWord's Model class
WAKE_WORD_MODEL = str(Path.home() / "claudinho" / "models" / "hey_rick.onnx")
WAKE_WORD_FALLBACK = "hey_jarvis_v0.1"  # used if model file not found
WAKE_WORD_THRESHOLD = 0.5

# ── OpenClaw Gateway ──────────────────────────────────────────
OPENCLAW_URL = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "4e09f12594b3f0ba3ae73680d1c40bfe2d6d9abf8eafe790"

# ── Inworld TTS ───────────────────────────────────────────────
INWORLD_TTS_URL = "https://api.inworld.ai/tts/v1/voice"
INWORLD_API_KEY = "endGMkV6SzRoeGgyUVVYZlB1UWZEYnBBSnhYZlRFVzk6S1hQUjFQSm00b1hzT2liZjFGNjdCczJPMm1vdzUyaXg5WmFpaTVmWEt5eTJDOE1pWUx3ZW00cU96WlBkbjkxVw=="
INWORLD_VOICE_ID = "Theodore"
INWORLD_MODEL = "inworld-tts-1.5-max"

# ── Voice Activity Detection (Silero VAD) ─────────────────────
# Neural VAD — much more accurate than RMS with noisy USB mics
SILERO_VAD_MODEL = None      # auto: ~/claudinho/models/silero_vad.onnx (downloads on first use)
VAD_THRESHOLD = 0.4          # speech probability threshold (0.0-1.0, lower = more sensitive)

# ── Recording ─────────────────────────────────────────────────
SILENCE_DURATION = 1.5       # seconds of silence to stop recording
MAX_RECORD_DURATION = 30     # max seconds per recording
MIN_RECORD_DURATION = 0.8    # minimum seconds before silence detection kicks in
MIN_SPEECH_DURATION = 0.3    # minimum speech detected before allowing silence stop
NOISE_MULTIPLIER = 2.5       # RMS fallback: speech must be this many times louder than noise

# ── Paths ─────────────────────────────────────────────────────
TMP_DIR = Path("/tmp/claudinho")
