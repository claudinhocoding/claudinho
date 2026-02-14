"""
Voice Activity Detection
========================

Tries multiple VAD backends in order:
1. silero-vad package (official, most accurate)
2. webrtcvad (Google's C-based VAD, fast and reliable)
3. RMS threshold (last resort)

Install one: pip install silero-vad  OR  pip install webrtcvad
"""

import logging
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class VADBackend(Protocol):
    """Interface for VAD backends."""
    def is_speech(self, audio_16k_int16: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


class SileroVADBackend:
    """Official silero-vad package (ONNX, no PyTorch needed)."""

    def __init__(self, threshold: float = 0.4):
        from silero_vad import load_silero_vad, VADIterator
        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self.iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=16000,
            min_silence_duration_ms=100,
        )
        logger.info(f"Silero VAD loaded (threshold={threshold})")

    def is_speech(self, audio_16k_int16: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        import torch
        # Normalize to float32 [-1, 1]
        audio_f32 = audio_16k_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f32)

        # Process in 512-sample windows
        window_size = 512
        any_speech = False
        for start in range(0, len(tensor) - window_size + 1, window_size):
            window = tensor[start:start + window_size]
            prob = self.model(window, 16000).item()
            if prob > self.threshold:
                any_speech = True
                break

        return any_speech

    def reset(self):
        self.model.reset_states()


class WebRTCVADBackend:
    """Google's WebRTC VAD (C-based, fast, reliable)."""

    def __init__(self, aggressiveness: int = 3):
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness
        logger.info(f"WebRTC VAD loaded (aggressiveness={aggressiveness})")

    def is_speech(self, audio_16k_int16: np.ndarray) -> bool:
        """Check if audio chunk contains speech using majority vote."""
        pcm = audio_16k_int16.tobytes()
        frame_duration_ms = 30  # 10, 20, or 30 ms
        frame_size = int(16000 * frame_duration_ms / 1000) * 2  # bytes (int16)
        n_speech = 0
        n_frames = 0

        for start in range(0, len(pcm) - frame_size + 1, frame_size):
            frame = pcm[start:start + frame_size]
            try:
                if self.vad.is_speech(frame, 16000):
                    n_speech += 1
            except Exception:
                pass
            n_frames += 1

        if n_frames == 0:
            return False
        # Speech if >30% of frames contain speech
        return (n_speech / n_frames) > 0.3

    def reset(self):
        pass  # WebRTC VAD is stateless


class RMSVADBackend:
    """Simple RMS-based VAD (last resort fallback)."""

    def __init__(self, threshold: float = 500):
        self.threshold = threshold
        self._calibrated = False
        logger.info(f"RMS VAD fallback (threshold={threshold})")

    def is_speech(self, audio_16k_int16: np.ndarray) -> bool:
        samples = audio_16k_int16.astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        return rms > self.threshold

    def reset(self):
        pass


def create_vad(threshold: float = 0.4) -> VADBackend:
    """Create the best available VAD backend."""
    # Try silero-vad first
    try:
        return SileroVADBackend(threshold=threshold)
    except ImportError:
        logger.info("silero-vad not installed (pip install silero-vad)")
    except Exception as e:
        logger.warning(f"silero-vad failed: {e}")

    # Try webrtcvad
    try:
        return WebRTCVADBackend(aggressiveness=3)
    except ImportError:
        logger.info("webrtcvad not installed (pip install webrtcvad)")
    except Exception as e:
        logger.warning(f"webrtcvad failed: {e}")

    # Fall back to RMS
    logger.warning("No VAD backend available, using RMS threshold")
    return RMSVADBackend(threshold=500)
