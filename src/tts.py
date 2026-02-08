"""
Text to Speech
==============

Uses KittenTTS for local speech synthesis.
Lightweight model optimized for CPU/Pi 5.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Synthesizes speech from text using KittenTTS."""
    
    # Available voices
    VOICES = [
        'expr-voice-2-m', 'expr-voice-2-f',
        'expr-voice-3-m', 'expr-voice-3-f',
        'expr-voice-4-m', 'expr-voice-4-f',
        'expr-voice-5-m', 'expr-voice-5-f',
    ]
    
    def __init__(
        self,
        model: str = "KittenML/kitten-tts-nano-0.2",
        voice: str = "expr-voice-2-f",
    ):
        self.model_name = model
        self.voice = voice
        self._model = None
        
        if voice not in self.VOICES:
            logger.warning(f"Unknown voice '{voice}', using default")
            self.voice = "expr-voice-2-f"
    
    def _initialize(self):
        """Lazy initialization of KittenTTS model."""
        if self._model is not None:
            return
        
        try:
            from kittentts import KittenTTS
        except ImportError:
            raise ImportError(
                "KittenTTS not installed. Run:\n"
                "pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl"
            )
        
        logger.info(f"Loading KittenTTS model: {self.model_name}")
        self._model = KittenTTS(self.model_name)
        logger.info("KittenTTS model loaded")
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            voice: Voice to use (optional, uses default if not specified).
            
        Returns:
            Audio data as numpy array (24kHz sample rate).
        """
        self._initialize()
        
        voice = voice or self.voice
        
        if not text.strip():
            logger.warning("Empty text, returning silence")
            return np.zeros(24000, dtype=np.float32)  # 1 second of silence
        
        logger.debug(f"Synthesizing: '{text[:50]}...' with voice {voice}")
        
        audio = self._model.generate(text, voice=voice)
        
        return audio
    
    def save(self, audio: np.ndarray, path: str):
        """Save audio to WAV file."""
        try:
            import soundfile as sf
            sf.write(path, audio, 24000)
            logger.debug(f"Saved audio to {path}")
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")
    
    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return 24000
    
    @classmethod
    def list_voices(cls):
        """Print available voices."""
        print("Available KittenTTS voices:")
        for voice in cls.VOICES:
            print(f"  - {voice}")


if __name__ == "__main__":
    # Test TTS
    TextToSpeech.list_voices()
    
    print("\nTesting synthesis...")
    tts = TextToSpeech()
    audio = tts.synthesize("Hello! I am Claudinho, your voice assistant.")
    tts.save(audio, "test_output.wav")
    print("Saved to test_output.wav")
