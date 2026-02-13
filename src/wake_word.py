"""
Wake Word Detection
===================

Uses openWakeWord with PyAudio for continuous listening.
Detects "Hey Jarvis" (swap to custom "Hey Claudinho" later).
"""

import logging
import numpy as np

import config

logger = logging.getLogger(__name__)

# openWakeWord expects 80ms chunks at 16kHz
OWW_CHUNK_SIZE = 1280


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
        
        # Open mic stream
        self._pyaudio = pyaudio.PyAudio()
        
        # Find the USB mic device index
        device_index = self._find_usb_mic()
        
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=OWW_CHUNK_SIZE,
        )
        logger.info("Audio stream opened for wake word detection")
    
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
    
    def listen(self) -> bool:
        """
        Listen for one chunk and check for wake word.
        Call in a loop. Returns True when detected.
        """
        self._initialize()
        
        audio_bytes = self._stream.read(OWW_CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        prediction = self._model.predict(audio_data)
        
        for model_name, score in prediction.items():
            if score > config.WAKE_WORD_THRESHOLD:
                logger.info(f"🔔 Wake word: {model_name} (score={score:.3f})")
                self._model.reset()
                return True
        
        return False
    
    def cleanup(self):
        """Release audio resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        self._model = None
        logger.info("Wake word detector cleaned up")
