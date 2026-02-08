"""
Wake Word Detection
===================

Uses Porcupine for wake word detection.
Listens for "Hey Claudinho" or configured wake phrase.
"""

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# Porcupine will be imported at runtime
# pip install pvporcupine pvrecorder


class WakeWordDetector:
    """Detects wake word using Porcupine."""
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        keyword: str = "jarvis",  # Built-in keyword, or path to custom .ppn
        sensitivity: float = 0.5,
        device_index: int = -1,  # -1 = default device
    ):
        self.access_key = access_key
        self.keyword = keyword
        self.sensitivity = sensitivity
        self.device_index = device_index
        
        self._porcupine = None
        self._recorder = None
    
    def _initialize(self):
        """Lazy initialization of Porcupine."""
        if self._porcupine is not None:
            return
        
        try:
            import pvporcupine
            from pvrecorder import PvRecorder
        except ImportError:
            raise ImportError(
                "Porcupine not installed. Run: pip install pvporcupine pvrecorder"
            )
        
        # Get access key from environment if not provided
        if not self.access_key:
            import os
            self.access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
        
        if not self.access_key:
            raise ValueError(
                "Porcupine access key required. "
                "Set PORCUPINE_ACCESS_KEY env var or pass access_key parameter. "
                "Get a free key at: https://picovoice.ai/"
            )
        
        # Initialize Porcupine
        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=[self.keyword] if not self.keyword.endswith('.ppn') else None,
            keyword_paths=[self.keyword] if self.keyword.endswith('.ppn') else None,
            sensitivities=[self.sensitivity],
        )
        
        # Initialize recorder
        self._recorder = PvRecorder(
            device_index=self.device_index,
            frame_length=self._porcupine.frame_length,
        )
        
        logger.info(f"Wake word detector initialized: '{self.keyword}'")
        logger.info(f"Using audio device: {self._recorder.selected_device}")
    
    def listen(self) -> bool:
        """
        Listen for wake word.
        
        Returns:
            True if wake word detected, False otherwise.
        """
        self._initialize()
        
        if not self._recorder.is_recording:
            self._recorder.start()
        
        # Read audio frame
        pcm = self._recorder.read()
        
        # Check for wake word
        keyword_index = self._porcupine.process(pcm)
        
        if keyword_index >= 0:
            return True
        
        return False
    
    def cleanup(self):
        """Release resources."""
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder.delete()
            self._recorder = None
        
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        
        logger.info("Wake word detector cleaned up")
    
    def __del__(self):
        self.cleanup()


# List available audio devices
def list_audio_devices():
    """Print available audio devices for debugging."""
    try:
        from pvrecorder import PvRecorder
        devices = PvRecorder.get_available_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device}")
    except ImportError:
        print("pvrecorder not installed")


if __name__ == "__main__":
    list_audio_devices()
