"""
Audio Utilities
===============

Handles audio recording and playback.
Works with USB microphone and speaker.
"""

import logging
import wave
import time
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024


class AudioRecorder:
    """Records audio from microphone."""
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self._pyaudio = None
        self._stream = None
    
    def _initialize(self):
        """Initialize PyAudio."""
        if self._pyaudio is not None:
            return
        
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        except ImportError:
            raise ImportError("PyAudio not installed. Run: pip install pyaudio")
    
    def record(self, duration: float) -> np.ndarray:
        """
        Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds.
            
        Returns:
            Audio data as numpy array.
        """
        self._initialize()
        import pyaudio
        
        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        frames = []
        num_chunks = int(self.sample_rate / CHUNK_SIZE * duration)
        
        logger.debug(f"Recording for {duration}s...")
        
        for _ in range(num_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
    
    def record_until_silence(
        self,
        silence_threshold: int = 500,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
    ) -> np.ndarray:
        """
        Record audio until silence is detected.
        
        Args:
            silence_threshold: RMS threshold for silence detection.
            silence_duration: Seconds of silence to stop recording.
            max_duration: Maximum recording duration.
            
        Returns:
            Audio data as numpy array.
        """
        self._initialize()
        import pyaudio
        
        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        frames = []
        silent_chunks = 0
        silent_chunks_needed = int(silence_duration * self.sample_rate / CHUNK_SIZE)
        max_chunks = int(max_duration * self.sample_rate / CHUNK_SIZE)
        
        logger.debug("Recording until silence...")
        
        for i in range(max_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            # Check for silence
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            # Stop if enough silence
            if silent_chunks >= silent_chunks_needed and len(frames) > silent_chunks_needed:
                logger.debug(f"Silence detected after {i * CHUNK_SIZE / self.sample_rate:.1f}s")
                break
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
    
    def cleanup(self):
        """Release resources."""
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    @staticmethod
    def list_devices():
        """List available audio input devices."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            print("Audio input devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']}")
            p.terminate()
        except ImportError:
            print("PyAudio not installed")


class AudioPlayer:
    """Plays audio through speaker."""
    
    def __init__(
        self,
        device_index: Optional[int] = None,
        sounds_dir: Optional[str] = None,
    ):
        self.device_index = device_index
        self.sounds_dir = Path(sounds_dir) if sounds_dir else Path(__file__).parent.parent / "sounds"
        self._pyaudio = None
    
    def _initialize(self):
        """Initialize PyAudio."""
        if self._pyaudio is not None:
            return
        
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        except ImportError:
            raise ImportError("PyAudio not installed. Run: pip install pyaudio")
    
    def play(self, audio: np.ndarray, sample_rate: int = 24000):
        """
        Play audio data.
        
        Args:
            audio: Audio data as numpy array (float32 or int16).
            sample_rate: Sample rate of the audio.
        """
        self._initialize()
        import pyaudio
        
        # Convert float32 to int16 if needed
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=self.device_index,
        )
        
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()
    
    def play_file(self, path: str):
        """Play a WAV file."""
        self._initialize()
        import pyaudio
        
        with wave.open(path, 'rb') as wf:
            stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=self.device_index,
            )
            
            data = wf.readframes(CHUNK_SIZE)
            while data:
                stream.write(data)
                data = wf.readframes(CHUNK_SIZE)
            
            stream.stop_stream()
            stream.close()
    
    def play_sound(self, name: str):
        """Play a named sound effect."""
        sound_path = self.sounds_dir / f"{name}.wav"
        if sound_path.exists():
            self.play_file(str(sound_path))
        else:
            logger.warning(f"Sound not found: {sound_path}")
    
    def cleanup(self):
        """Release resources."""
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    @staticmethod
    def list_devices():
        """List available audio output devices."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            print("Audio output devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    print(f"  [{i}] {info['name']}")
            p.terminate()
        except ImportError:
            print("PyAudio not installed")


if __name__ == "__main__":
    print("=== Audio Devices ===\n")
    AudioRecorder.list_devices()
    print()
    AudioPlayer.list_devices()
