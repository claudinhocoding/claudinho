"""
Audio Utilities
===============

Record and play audio via PyAudio (with silence detection) and aplay.
"""

import logging
import subprocess
import wave
import struct
import math
from pathlib import Path

import numpy as np
import pyaudio

import config

logger = logging.getLogger(__name__)

# Mic native rate
MIC_RATE = 44100
CHUNK_DURATION = 0.1  # 100ms chunks
MIC_CHUNK = int(MIC_RATE * CHUNK_DURATION)


def _open_mic():
    """Open the USB mic via PyAudio. Returns (pa, stream, device_index)."""
    pa = pyaudio.PyAudio()
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "usb" in info.get("name", "").lower() and info["maxInputChannels"] > 0:
            device_index = i
            break
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=MIC_CHUNK,
    )
    return pa, stream


def calibrate_noise(stream, duration: float = 0.8) -> float:
    """
    Measure ambient noise level for a short period.
    Returns a threshold = noise_floor * multiplier.
    """
    chunks = int(duration / CHUNK_DURATION)
    rms_values = []
    
    for _ in range(chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_values.append(rms)
    
    noise_floor = np.mean(rms_values)
    # Threshold = noise floor * 2.5 (speech is typically 3-10x louder than ambient)
    threshold = max(noise_floor * 2.5, 300)  # minimum 300 to avoid hypersensitivity
    logger.info(f"🔇 Noise floor: {noise_floor:.0f} RMS → speech threshold: {threshold:.0f}")
    return threshold


def record_until_silence(output_path: str) -> str:
    """
    Record from mic until user stops speaking.
    
    1. Calibrates ambient noise level
    2. Records at 44100 Hz (mic native)
    3. Stops when silence detected after speech
    4. Saves as 16kHz WAV for Whisper
    """
    logger.info("🎤 Listening...")
    
    pa, stream = _open_mic()
    
    # Auto-calibrate noise threshold
    threshold = calibrate_noise(stream)
    
    frames = []
    silent_chunks = 0
    has_speech = False
    chunks_needed_for_silence = int(config.SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(config.MAX_RECORD_DURATION / CHUNK_DURATION)
    min_chunks = int(config.MIN_RECORD_DURATION / CHUNK_DURATION)
    
    for i in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate RMS
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        
        if rms > threshold:
            has_speech = True
            silent_chunks = 0
        else:
            silent_chunks += 1
        
        # Stop if silence after speech (and past minimum duration)
        if has_speech and i >= min_chunks and silent_chunks >= chunks_needed_for_silence:
            logger.info(f"Silence detected after {i * CHUNK_DURATION:.1f}s")
            break
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    if not has_speech:
        logger.info("No speech detected")
        return ""
    
    # Combine frames, downsample 44100→16000, save as WAV
    audio_44k = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_16k = _downsample(audio_44k, MIC_RATE, 16000)
    
    _save_wav(output_path, audio_16k, 16000)
    logger.info(f"Recorded {len(audio_16k) / 16000:.1f}s of audio")
    
    return output_path


def _downsample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple downsample by taking every Nth sample."""
    from scipy.signal import resample
    target_length = int(len(audio) * to_rate / from_rate)
    return resample(audio, target_length).astype(np.int16)


def _save_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Save numpy int16 array as WAV."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def has_speech(wav_path: str, threshold: float = 200) -> bool:
    """Check if a WAV file contains speech."""
    try:
        with wave.open(wav_path, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            if len(data) < 100:
                return False
            samples = struct.unpack(f"<{len(data) // 2}h", data)
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            return rms > threshold
    except Exception:
        return False


def play(wav_path: str):
    """Play a WAV file through the USB speaker."""
    logger.info("🔊 Playing audio...")
    subprocess.run(
        ["aplay", "-D", config.SPEAKER_DEVICE, wav_path],
        check=True,
        capture_output=True,
    )


def play_beep():
    """Play a short beep to indicate listening."""
    # Generate a simple beep tone
    beep_path = str(config.TMP_DIR / "beep.wav")
    
    if not Path(beep_path).exists():
        _generate_beep(beep_path)
    
    play(beep_path)


def _generate_beep(path: str, freq: int = 880, duration: float = 0.15):
    """Generate a simple beep WAV file."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        for i in range(n_samples):
            t = i / sample_rate
            # Simple sine wave with fade out
            fade = 1.0 - (i / n_samples)
            sample = int(16000 * fade * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", sample))
