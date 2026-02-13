"""
Wake Word Detection
===================

Uses openWakeWord with PyAudio for continuous listening.
Detects "Hey Jarvis" (swap to custom "Hey Claudinho" later).

Note: USB mic only supports 44100 Hz natively, so we record
at 44100 Hz and downsample to 16000 Hz for openWakeWord.
"""

import logging
import numpy as np
from scipy.signal import resample

import config

logger = logging.getLogger(__name__)

# openWakeWord expects 80ms chunks at 16kHz = 1280 samples
OWW_CHUNK_SIZE = 1280
# Record at 44100 Hz (USB mic native rate)
MIC_RATE = 44100
# Equivalent chunk size at 44100 Hz for 80ms
MIC_CHUNK_SIZE = int(MIC_RATE * 0.08)  # 3528 samples


class WakeWordDetector:
    """Listens for wake word using openWakeWord."""
    
    def __init__(self):
        self._model = None
        self._pyaudio = None
        self._stream = None
    
    def _initialize(self):
        """Lazy init — only load models when first needed."""
        if self._model is not None:
            return
        
        import openwakeword
        from openwakeword.model import Model
        import pyaudio
        
        # Download default models if needed
        openwakeword.utils.download_models()
        
        # Load wake word model
        self._model = Model(
            wakeword_models=[config.WAKE_WORD_MODEL],
            inference_framework="onnx",
        )
        logger.info(f"Wake word model loaded: {config.WAKE_WORD_MODEL}")
        
        # Open mic stream at native 44100 Hz
        self._pyaudio = pyaudio.PyAudio()
        
        # Find the USB mic device index
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
    
    def _find_usb_mic(self) -> int:
        """Find the USB microphone PyAudio device index."""
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            # Match our USB PnP Sound Device
            if "usb" in name and info["maxInputChannels"] > 0:
                logger.info(f"Found USB mic: [{i}] {info['name']}")
                return i
        
        logger.warning("USB mic not found by name, using default input")
        return None
    
    def _downsample(self, audio_44100: np.ndarray) -> np.ndarray:
        """Downsample from 44100 Hz to 16000 Hz."""
        target_length = int(len(audio_44100) * 16000 / 44100)
        return resample(audio_44100, target_length).astype(np.int16)
    
    def listen(self) -> bool:
        """
        Listen for one chunk and check for wake word.
        Call in a loop. Returns True when detected.
        """
        self._initialize()
        
        # Read at 44100 Hz
        audio_bytes = self._stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
        audio_44100 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Downsample to 16000 Hz for openWakeWord
        audio_16000 = self._downsample(audio_44100)
        
        prediction = self._model.predict(audio_16000)
        
        for model_name, score in prediction.items():
            if score > config.WAKE_WORD_THRESHOLD:
                logger.info(f"🔔 Wake word: {model_name} (score={score:.3f})")
                self._model.reset()
                return True
        
        return False
    
    def pause(self):
        """Release the mic so arecord can use it."""
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
        if self._model is None:
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
        logger.debug("Mic reopened for wake word")

    def cleanup(self):
        """Release all resources."""
        self.pause()
        self._model = None
        logger.info("Wake word detector cleaned up")
