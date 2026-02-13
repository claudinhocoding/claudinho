"""
Text to Speech
==============

Uses Piper TTS for local speech synthesis.
Fast, lightweight, runs great on Pi 5.

https://github.com/rhasspy/piper
"""

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Default voice - good quality, fast on Pi 5
DEFAULT_VOICE = "en_US-lessac-medium"


class TextToSpeech:
    """Synthesizes speech from text using Piper TTS."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        piper_path: Optional[str] = None,
        speed: float = 1.0,
    ):
        """
        Initialize Piper TTS.

        Args:
            voice: Piper voice name (e.g., 'en_US-lessac-medium').
                   See https://rhasspy.github.io/piper-samples/ for samples.
            piper_path: Path to piper binary. None = find in PATH.
            speed: Speech speed multiplier (1.0 = normal).
        """
        self.voice = voice
        self.speed = speed
        self.piper_path = piper_path or self._find_piper()
        self._sample_rate = 22050  # Piper default

    def _find_piper(self) -> str:
        """Find piper binary."""
        # Check if installed via pip (piper-tts package)
        result = subprocess.run(
            ["which", "piper"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # Check common install paths
        common_paths = [
            Path.home() / "piper" / "piper",
            Path("/usr/local/bin/piper"),
            Path("/opt/piper/piper"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "Piper TTS not found. Install with:\n"
            "  pip install piper-tts\n"
            "Or download from: https://github.com/rhasspy/piper/releases"
        )

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak.

        Returns:
            Audio data as numpy array (int16, 22050Hz).
        """
        if not text.strip():
            logger.warning("Empty text, returning silence")
            return np.zeros(self._sample_rate, dtype=np.int16)

        logger.debug(f"Synthesizing: '{text[:80]}...'")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Run piper
            cmd = [
                self.piper_path,
                "--model", self.voice,
                "--output_file", tmp_path,
            ]

            if self.speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self.speed)])

            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr}")
                return np.zeros(self._sample_rate, dtype=np.int16)

            # Read the generated audio
            with wave.open(tmp_path, "rb") as wf:
                self._sample_rate = wf.getframerate()
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                )

            return audio_data

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: str):
        """Synthesize and save directly to WAV file."""
        cmd = [
            self.piper_path,
            "--model", self.voice,
            "--output_file", output_path,
        ]

        result = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.error(f"Piper error: {result.stderr}")

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate

    @staticmethod
    def list_voices():
        """List installed Piper voices."""
        result = subprocess.run(
            ["piper", "--list-voices"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Run 'piper --list-voices' to see available voices")
            print("Download voices from: https://rhasspy.github.io/piper-samples/")


if __name__ == "__main__":
    print("Testing Piper TTS...")
    tts = TextToSpeech()
    audio = tts.synthesize("Hello! I am Claudinho, your voice assistant.")
    print(f"Generated {len(audio)} samples at {tts.sample_rate}Hz")
    print(f"Duration: {len(audio) / tts.sample_rate:.1f}s")
