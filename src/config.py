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
# Custom model: set to a file path (e.g. Path.home() / "claudinho" / "models" / "claudinho.onnx")
# Built-in model: set to a string name (e.g. "hey_jarvis_v0.1")
WAKE_WORD_MODEL = str(Path.home() / "claudinho" / "models" / "claudinho.onnx")
WAKE_WORD_FALLBACK = "hey_jarvis_v0.1"  # used if custom model file not found
WAKE_WORD_THRESHOLD = 0.5

# ── OpenClaw Gateway ──────────────────────────────────────────
OPENCLAW_URL = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "4e09f12594b3f0ba3ae73680d1c40bfe2d6d9abf8eafe790"

# ── Recording ─────────────────────────────────────────────────
SILENCE_THRESHOLD = 500      # RMS threshold for silence detection
SILENCE_DURATION = 1.5       # seconds of silence to stop recording
MAX_RECORD_DURATION = 15     # max seconds per recording
MIN_RECORD_DURATION = 1.0    # minimum seconds before silence detection kicks in

# ── Paths ─────────────────────────────────────────────────────
TMP_DIR = Path("/tmp/claudinho")
