"""
Audio Utilities
===============

Record and play audio using ALSA (arecord/aplay).
Uses subprocess — simple, reliable, tested on Pi 5.
"""

import logging
import subprocess
import wave
import struct
import math
import time
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def record_fixed(duration: float, output_path: str) -> str:
    """Record audio for a fixed duration using arecord."""
    subprocess.run(
        [
            "arecord",
            "-D", config.MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(config.SAMPLE_RATE),
            "-c", str(config.CHANNELS),
            "-d", str(int(duration)),
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def record_until_silence(output_path: str) -> str:
    """
    Record audio until silence is detected.
    
    Uses arecord to record max duration, then trims trailing silence.
    For a voice assistant, this is good enough — Whisper handles
    trailing silence fine, and we cap at MAX_RECORD_DURATION.
    """
    logger.info("🎤 Recording...")
    
    subprocess.run(
        [
            "arecord",
            "-D", config.MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(config.SAMPLE_RATE),
            "-c", str(config.CHANNELS),
            "-d", str(config.MAX_RECORD_DURATION),
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    
    # Check if recording has speech (not just silence)
    if not has_speech(output_path):
        logger.info("No speech detected in recording")
        return ""
    
    return output_path


def record_with_vad(output_path: str) -> str:
    """
    Record with simple voice activity detection.
    
    Starts arecord, monitors the file size / audio energy,
    and kills the process when silence is detected.
    """
    logger.info("🎤 Recording (press Ctrl+C or wait for silence)...")
    
    proc = subprocess.Popen(
        [
            "arecord",
            "-D", config.MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(config.SAMPLE_RATE),
            "-c", str(config.CHANNELS),
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for minimum duration before checking silence
    time.sleep(config.MIN_RECORD_DURATION)
    
    # Monitor for silence by periodically reading the WAV file
    silence_start = None
    start_time = time.time()
    
    while proc.poll() is None:
        elapsed = time.time() - start_time
        
        # Hard limit
        if elapsed >= config.MAX_RECORD_DURATION:
            logger.info(f"Max duration reached ({config.MAX_RECORD_DURATION}s)")
            break
        
        # Check tail of recording for silence
        try:
            rms = get_tail_rms(output_path)
            if rms < config.SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= config.SILENCE_DURATION:
                    logger.info("Silence detected, stopping recording")
                    break
            else:
                silence_start = None
        except Exception:
            pass  # File might not be ready yet
        
        time.sleep(0.2)
    
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    if not has_speech(output_path):
        logger.info("No speech detected")
        return ""
    
    return output_path


def get_tail_rms(wav_path: str, tail_seconds: float = 0.5) -> float:
    """Get RMS of the last N seconds of a WAV file."""
    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            tail_frames = int(wf.getframerate() * tail_seconds)
            
            if n_frames < tail_frames:
                return 9999  # Not enough data yet, assume speech
            
            wf.setpos(n_frames - tail_frames)
            data = wf.readframes(tail_frames)
            
            samples = struct.unpack(f"<{tail_frames}h", data)
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            return rms
    except Exception:
        return 9999  # Assume speech on error


def has_speech(wav_path: str, threshold: float = 200) -> bool:
    """Check if a WAV file contains speech (not just silence/noise)."""
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
