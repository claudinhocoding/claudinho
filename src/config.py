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
WAKE_WORD_MODEL = "hey_jarvis_v0.1"  # built-in; swap to custom .onnx later
WAKE_WORD_THRESHOLD = 0.5

# ── LLM ───────────────────────────────────────────────────────
# Reads from ANTHROPIC_API_KEY env var by default
ANTHROPIC_MODEL = "claude-haiku-4-5-20250929"
SYSTEM_PROMPT = """You are Claudinho, a friendly voice assistant running on a Raspberry Pi 5.
Keep responses concise and conversational — you're being spoken aloud.
2-3 sentences max unless asked for more detail.
You speak English and Portuguese. Reply in whatever language the user speaks to you."""

# ── Recording ─────────────────────────────────────────────────
SILENCE_THRESHOLD = 500      # RMS threshold for silence detection
SILENCE_DURATION = 1.5       # seconds of silence to stop recording
MAX_RECORD_DURATION = 15     # max seconds per recording
MIN_RECORD_DURATION = 1.0    # minimum seconds before silence detection kicks in

# ── Paths ─────────────────────────────────────────────────────
TMP_DIR = Path("/tmp/claudinho")
