"""
Voice Activity Detection (Silero VAD)
=====================================

Neural speech detection using Silero VAD ONNX model.
Much more accurate than RMS-based threshold, especially with noisy USB mics.

Requires: onnxruntime (already installed for openWakeWord)
Model:    silero_vad.onnx (~2MB, auto-downloaded on first use)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
WINDOW_SIZE = 512  # samples at 16kHz (32ms per window)
SAMPLE_RATE = 16000


def _download_model(dest: Path) -> bool:
    """Download Silero VAD ONNX model if not present."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Silero VAD model to {dest}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(dest))
        logger.info(f"Downloaded ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download VAD model: {e}")
        return False


class SileroVAD:
    """
    Silero VAD using ONNX runtime (no PyTorch needed).

    Usage:
        vad = SileroVAD("path/to/silero_vad.onnx")
        prob = vad.process_chunk(audio_16k_int16)
        if prob > 0.5:
            print("Speech detected!")
        vad.reset()  # call between utterances
    """

    def __init__(self, model_path: Optional[str] = None):
        import onnxruntime

        if model_path is None:
            model_path = str(Path.home() / "claudinho" / "models" / "silero_vad.onnx")

        path = Path(model_path)
        if not path.exists():
            if not _download_model(path):
                raise FileNotFoundError(f"VAD model not found: {model_path}")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(path), sess_options=opts)

        self._detect_version()
        self.reset()
        logger.info(f"Silero VAD v{self.version} loaded from {path.name}")

    def _detect_version(self):
        """Auto-detect model version from input names."""
        input_names = [i.name for i in self.session.get_inputs()]
        if "state" in input_names:
            self.version = 5
            self._state_dim = 128
        else:
            self.version = 4
            self._state_dim = 64

    def reset(self):
        """Reset internal state (call between utterances)."""
        if self.version == 5:
            self._state = np.zeros((2, 1, self._state_dim), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, self._state_dim), dtype=np.float32)
            self._c = np.zeros((2, 1, self._state_dim), dtype=np.float32)
        self._buffer = np.array([], dtype=np.float32)

    def process_chunk(self, audio_16k: np.ndarray) -> float:
        """
        Process a chunk of 16kHz int16 audio.
        Returns the max speech probability (0.0 to 1.0) across all windows.

        Handles chunks of any size by buffering leftover samples.
        """
        # Normalize to float32 [-1, 1]
        audio_f32 = audio_16k.astype(np.float32) / 32768.0

        # Append to buffer (handles leftover from previous chunk)
        self._buffer = np.concatenate([self._buffer, audio_f32])

        max_prob = 0.0
        # Process complete windows
        while len(self._buffer) >= WINDOW_SIZE:
            window = self._buffer[:WINDOW_SIZE]
            self._buffer = self._buffer[WINDOW_SIZE:]
            prob = self._infer(window)
            max_prob = max(max_prob, prob)

        return max_prob

    def _infer(self, window: np.ndarray) -> float:
        """Run one ONNX inference on a 512-sample window."""
        x = window.reshape(1, -1)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        if self.version == 5:
            inputs = {"input": x, "state": self._state, "sr": sr}
            out, self._state = self.session.run(None, inputs)
        else:
            inputs = {"input": x, "h": self._h, "c": self._c, "sr": sr}
            out, self._h, self._c = self.session.run(None, inputs)

        return float(np.squeeze(out))
