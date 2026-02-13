"""
Wake Word Detection
===================

Uses openWakeWord for local wake word detection.
Listens for "Hey Claudinho" or configured wake phrase.
Runs continuously with minimal CPU usage (~1%).

https://github.com/dscripka/openWakeWord
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Audio parameters matching openWakeWord requirements
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # openWakeWord expects 80ms chunks at 16kHz


class WakeWordDetector:
    """Detects wake word using openWakeWord."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device_index: Optional[int] = None,
    ):
        """
        Initialize wake word detector.

        Args:
            model_path: Path to custom .onnx model, or None for built-in models.
                        Train custom "Hey Claudinho" at: https://github.com/dscripka/openWakeWord#training
            threshold: Detection threshold (0.0-1.0). Higher = fewer false positives.
            device_index: Audio input device index. None = default mic.
        """
        self.model_path = model_path
        self.threshold = threshold
        self.device_index = device_index

        self._oww_model = None
        self._pyaudio = None
        self._stream = None

    def _initialize(self):
        """Lazy initialization of openWakeWord and audio stream."""
        if self._oww_model is not None:
            return

        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "openWakeWord not installed. Run: pip install openwakeword"
            )

        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "PyAudio not installed. Run: pip install pyaudio"
            )

        # Download default models if needed
        openwakeword.utils.download_models()

        # Initialize model
        if self.model_path:
            # Custom trained model (e.g., "Hey Claudinho")
            self._oww_model = Model(
                wakeword_models=[self.model_path],
                inference_framework="onnx",
            )
            logger.info(f"Loaded custom wake word model: {self.model_path}")
        else:
            # Use built-in "hey jarvis" as placeholder until custom model is trained
            self._oww_model = Model(
                wakeword_models=["hey_jarvis_v0.1"],
                inference_framework="onnx",
            )
            logger.info("Using built-in 'hey jarvis' model (train custom model for 'Hey Claudinho')")

        # Initialize audio stream
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info(f"Wake word detector initialized (threshold={self.threshold})")

    def listen(self) -> bool:
        """
        Listen for one audio chunk and check for wake word.

        Call this in a loop. Returns True when wake word is detected.

        Returns:
            True if wake word detected, False otherwise.
        """
        self._initialize()

        # Read audio chunk
        audio_bytes = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        # Run detection
        prediction = self._oww_model.predict(audio_data)

        # Check all model predictions against threshold
        for model_name, score in prediction.items():
            if score > self.threshold:
                logger.info(f"Wake word detected: {model_name} (score={score:.3f})")
                # Reset model state to avoid repeated triggers
                self._oww_model.reset()
                return True

        return False

    def listen_blocking(self, timeout: Optional[float] = None) -> bool:
        """
        Block until wake word is detected or timeout.

        Args:
            timeout: Maximum seconds to listen. None = listen forever.

        Returns:
            True if wake word detected, False if timed out.
        """
        import time

        start = time.time()

        while True:
            if self.listen():
                return True

            if timeout and (time.time() - start) > timeout:
                return False

    def cleanup(self):
        """Release resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

        self._oww_model = None
        logger.info("Wake word detector cleaned up")

    def __del__(self):
        self.cleanup()


def list_audio_devices():
    """Print available audio input devices for debugging."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        print("Available audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} (rate={int(info['defaultSampleRate'])})")
        p.terminate()
    except ImportError:
        print("PyAudio not installed. Run: pip install pyaudio")


if __name__ == "__main__":
    list_audio_devices()
    print()
    print("Testing wake word detection (say 'Hey Jarvis')...")
    print("Press Ctrl+C to stop\n")

    detector = WakeWordDetector(threshold=0.5)
    try:
        while True:
            if detector.listen():
                print("🔔 Wake word detected!")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        detector.cleanup()
