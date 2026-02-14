"""
Wake Word Detection
===================

Supports two modes:
1. Custom .onnx model (trained from real recordings) — runs mel→embedding→classifier pipeline
2. Built-in openWakeWord model (e.g. "hey_jarvis_v0.1") — uses the standard Model class

Note: USB mic only supports 44100 Hz natively, so we record
at 44100 Hz and downsample to 16000 Hz for inference.
"""

import ctypes
import logging
from pathlib import Path

import numpy as np
from scipy.signal import resample

import config

# Suppress ALSA warnings before PyAudio gets imported
_ERROR_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def _null_error_handler(filename, line, function, err, fmt):
    pass
_c_null_handler = _ERROR_HANDLER(_null_error_handler)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_c_null_handler)
except OSError:
    pass

logger = logging.getLogger(__name__)

# Audio constants
MIC_RATE = 44100
MIC_CHUNK_SIZE = int(MIC_RATE * 0.08)  # 80ms at 44100 Hz

# openWakeWord embedding constants
WINDOW = 1280          # 80ms at 16kHz
MELS_PER_CHUNK = 5     # mel model outputs 5 frames per chunk
N_MEL_FRAMES = 76      # embedding model needs 76 mel frames


class WakeWordDetector:
    """Listens for wake word using either custom or built-in model."""
    
    def __init__(self):
        self._pyaudio = None
        self._stream = None
        self._custom_mode = False
        
        # Custom model components
        self._mel_session = None
        self._emb_session = None
        self._cls_session = None
        self._feat_mean = None
        self._feat_std = None
        self._mel_buffer = []
        
        # Built-in model
        self._model = None
    
    def _initialize(self):
        """Lazy init — only load models when first needed."""
        if self._mel_session is not None or self._model is not None:
            return
        
        import pyaudio
        
        model_path = config.WAKE_WORD_MODEL
        
        if Path(model_path).exists():
            # Community/custom .onnx models work with openWakeWord Model class
            self._init_oww_model(model_path)
        else:
            self._init_builtin()
        
        # Open mic stream
        self._pyaudio = pyaudio.PyAudio()
        device_index = self._find_usb_mic()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=MIC_CHUNK_SIZE,
        )
        logger.info("Audio stream opened for wake word detection (44100 Hz → 16000 Hz)")
    
    def _init_oww_model(self, model_path):
        """Initialize openWakeWord-compatible .onnx model (community or custom)."""
        import openwakeword
        from openwakeword.model import Model
        
        self._custom_mode = False
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )
        logger.info(f"Wake word model loaded: {model_path}")
    
    def _init_builtin(self):
        """Initialize built-in openWakeWord model."""
        import openwakeword
        from openwakeword.model import Model
        
        self._custom_mode = False
        model_name = getattr(config, 'WAKE_WORD_FALLBACK', 'hey_jarvis_v0.1')
        openwakeword.utils.download_models()
        self._model = Model(
            wakeword_models=[model_name],
            inference_framework="onnx",
        )
        logger.info(f"Built-in wake word model loaded: {model_name}")
    
    def _find_oww_models(self) -> Path:
        """Find openWakeWord's embedding model directory."""
        # Check common locations
        candidates = [
            Path.home() / ".local" / "lib" / "python3.13" / "site-packages" / "openwakeword" / "resources" / "models",
        ]
        
        # Try to find via import
        try:
            import openwakeword
            candidates.insert(0, Path(openwakeword.__file__).parent / "resources" / "models")
        except ImportError:
            pass
        
        for p in candidates:
            if (p / "melspectrogram.onnx").exists():
                logger.info(f"Found openWakeWord models at {p}")
                return p
        
        raise FileNotFoundError(
            "Cannot find openWakeWord embedding models (melspectrogram.onnx, embedding_model.onnx). "
            "Install openwakeword or download them manually."
        )
    
    def _find_usb_mic(self) -> int:
        """Find the USB microphone PyAudio device index."""
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if "usb" in name and info["maxInputChannels"] > 0:
                logger.info(f"Found USB mic: [{i}] {info['name']}")
                return i
        logger.warning("USB mic not found by name, using default input")
        return None
    
    def _downsample(self, audio_44100: np.ndarray) -> np.ndarray:
        """Downsample from 44100 Hz to 16000 Hz."""
        target_length = int(len(audio_44100) * 16000 / 44100)
        return resample(audio_44100, target_length).astype(np.int16)
    
    def _process_custom(self, audio_16k: np.ndarray) -> float:
        """
        Process 80ms of 16kHz audio through the custom pipeline.
        Returns max detection score.
        """
        audio_f = audio_16k.astype(np.float32) / 32768.0
        
        # Run mel spectrogram: (1, 1280) -> (1, 1, 5, 32)
        mel_out = self._mel_session.run(
            None, {"input": audio_f.reshape(1, -1)}
        )[0]
        frames = mel_out.reshape(-1, 32)  # (5, 32)
        self._mel_buffer.append(frames)
        
        # Keep only enough for one embedding window
        max_chunks = (N_MEL_FRAMES // MELS_PER_CHUNK) + 2
        if len(self._mel_buffer) > max_chunks:
            self._mel_buffer = self._mel_buffer[-max_chunks:]
        
        # Stack all mel frames
        all_mels = np.concatenate(self._mel_buffer, axis=0)  # (N*5, 32)
        
        if len(all_mels) < N_MEL_FRAMES:
            return 0.0
        
        # Take the last 76 mel frames
        mel_window = all_mels[-N_MEL_FRAMES:]  # (76, 32)
        mel_input = mel_window.reshape(1, 76, 32, 1).astype(np.float32)
        
        # Run embedding model: (1, 76, 32, 1) -> (1, 1, 1, 96)
        emb_out = self._emb_session.run(None, {"input_1": mel_input})[0]
        embedding = emb_out.flatten()  # (96,)
        
        # Normalize
        embedding_norm = (embedding - self._feat_mean) / self._feat_std
        
        # Run classifier: (1, 96) -> (1, 1) score
        score = self._cls_session.run(
            None, {"input": embedding_norm.reshape(1, -1).astype(np.float32)}
        )[0]
        
        return float(score.flatten()[0])
    
    def listen(self) -> bool:
        """
        Listen for one chunk and check for wake word.
        Call in a loop. Returns True when detected.
        """
        self._initialize()
        
        # Read at 44100 Hz
        audio_bytes = self._stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
        audio_44100 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Downsample to 16000 Hz
        audio_16000 = self._downsample(audio_44100)
        
        if self._custom_mode:
            # Process through custom pipeline
            # Split into 1280-sample (80ms) windows
            for start in range(0, len(audio_16000) - WINDOW + 1, WINDOW):
                chunk = audio_16000[start:start + WINDOW]
                score = self._process_custom(chunk)
                
                if score > config.WAKE_WORD_THRESHOLD:
                    logger.info(f"🔔 Wake word detected! (score={score:.3f})")
                    self._mel_buffer = []  # reset after detection
                    return True
        else:
            # Built-in openWakeWord model
            prediction = self._model.predict(audio_16000)
            for model_name, score in prediction.items():
                if score > config.WAKE_WORD_THRESHOLD:
                    logger.info(f"🔔 Wake word: {model_name} (score={score:.3f})")
                    self._model.reset()
                    return True
        
        return False
    
    def pause(self):
        """Release the mic so recording can use it."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        logger.debug("Mic released for recording")
    
    def resume(self):
        """Reopen the mic stream for wake word listening."""
        if self._mel_session is None and self._model is None:
            return  # Not initialized yet
        import pyaudio
        self._pyaudio = pyaudio.PyAudio()
        device_index = self._find_usb_mic()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=MIC_CHUNK_SIZE,
        )
        # Reset mel buffer on resume
        self._mel_buffer = []
        logger.debug("Mic reopened for wake word")

    def cleanup(self):
        """Release all resources."""
        self.pause()
        self._mel_session = None
        self._emb_session = None
        self._cls_session = None
        self._model = None
        logger.info("Wake word detector cleaned up")
