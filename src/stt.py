"""
Speech to Text
==============

Uses Whisper.cpp for local speech recognition.
Optimized for Raspberry Pi 5.
"""

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribes speech to text using Whisper.cpp."""
    
    def __init__(
        self,
        model: str = "base",  # tiny, base, small, medium
        language: str = "en",
        whisper_cpp_path: Optional[str] = None,
    ):
        self.model = model
        self.language = language
        self.whisper_cpp_path = whisper_cpp_path or self._find_whisper_cpp()
        self.model_path = None
        
        self._validate_setup()
    
    def _find_whisper_cpp(self) -> str:
        """Find whisper.cpp installation."""
        common_paths = [
            Path.home() / "whisper.cpp",
            Path("/opt/whisper.cpp"),
            Path("/usr/local/whisper.cpp"),
        ]
        
        for path in common_paths:
            if (path / "main").exists():
                return str(path)
        
        # Try to find in PATH
        result = subprocess.run(
            ["which", "whisper-cpp"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return str(Path(result.stdout.strip()).parent)
        
        raise FileNotFoundError(
            "whisper.cpp not found. Install it with:\n"
            "  git clone https://github.com/ggerganov/whisper.cpp\n"
            "  cd whisper.cpp && make\n"
            "  ./models/download-ggml-model.sh base"
        )
    
    def _validate_setup(self):
        """Validate whisper.cpp installation."""
        whisper_path = Path(self.whisper_cpp_path)
        
        # Check for main executable
        main_path = whisper_path / "main"
        if not main_path.exists():
            raise FileNotFoundError(f"whisper.cpp main executable not found at {main_path}")
        
        # Check for model
        model_file = f"ggml-{self.model}.bin"
        self.model_path = whisper_path / "models" / model_file
        
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            logger.info(f"Download with: cd {whisper_path} && ./models/download-ggml-model.sh {self.model}")
    
    def transcribe(self, audio: Union[bytes, np.ndarray, str]) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as bytes, numpy array, or path to WAV file.
            
        Returns:
            Transcribed text.
        """
        # Handle different input types
        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, (bytes, np.ndarray)):
            audio_path = self._save_temp_wav(audio)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        
        try:
            # Run whisper.cpp
            result = subprocess.run(
                [
                    str(Path(self.whisper_cpp_path) / "main"),
                    "-m", str(self.model_path),
                    "-l", self.language,
                    "-f", audio_path,
                    "--no-timestamps",
                    "-t", "4",  # threads (Pi 5 has 4 cores)
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                logger.error(f"Whisper error: {result.stderr}")
                return ""
            
            # Parse output - whisper.cpp outputs text after processing info
            output = result.stdout.strip()
            
            # Extract just the transcription (skip processing info)
            lines = output.split('\n')
            text_lines = [l for l in lines if not l.startswith('[') and l.strip()]
            
            return ' '.join(text_lines).strip()
            
        finally:
            # Clean up temp file if we created one
            if isinstance(audio, (bytes, np.ndarray)):
                Path(audio_path).unlink(missing_ok=True)
    
    def _save_temp_wav(self, audio: Union[bytes, np.ndarray]) -> str:
        """Save audio to temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        
        if isinstance(audio, np.ndarray):
            # Assume 16kHz mono int16
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio.tobytes())
        else:
            # Raw bytes - write directly
            temp_file.write(audio)
        
        temp_file.close()
        return temp_file.name


if __name__ == "__main__":
    # Test STT
    stt = SpeechToText(model="base")
    print(f"Whisper.cpp path: {stt.whisper_cpp_path}")
    print(f"Model path: {stt.model_path}")
