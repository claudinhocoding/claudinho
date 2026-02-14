"""
Audio Utilities
===============

Record and play audio via PyAudio (with silence detection) and aplay.

Silence detection uses a sliding-window RMS approach with:
- Percentile-based noise calibration (robust to transient spikes)
- Smoothed RMS over multiple chunks (avoids false triggers)
- Trailing silence trimming (cleaner audio for Whisper)
"""

import collections
import ctypes
import logging
import math
import os
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np

# Suppress ALSA warnings before importing PyAudio
# (ALSA dumps "underrun occurred" / "Unknown PCM" spam to stderr)
_ERROR_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def _null_error_handler(filename, line, function, err, fmt):
    pass
_c_null_handler = _ERROR_HANDLER(_null_error_handler)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_c_null_handler)
except OSError:
    pass  # Not on Linux or ALSA not available

import pyaudio

import config

logger = logging.getLogger(__name__)

# Mic native rate
MIC_RATE = 44100
CHUNK_DURATION = 0.1  # 100ms chunks
MIC_CHUNK = int(MIC_RATE * CHUNK_DURATION)

# Sliding window size for RMS smoothing (number of chunks)
RMS_WINDOW = 5  # 500ms window


def _find_usb_device(direction: str = "input") -> str:
    """
    Find USB audio device card number dynamically.
    Returns ALSA device string like 'plughw:1,0'.
    Survives card number changes across reboots.
    """
    import re
    cmd = "arecord -l" if direction == "input" else "aplay -l"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB" in line and "card" in line:
                match = re.search(r"card (\d+)", line)
                if match:
                    card = match.group(1)
                    device = f"plughw:{card},0"
                    logger.info(f"Auto-detected {direction} device: {device} ({line.strip()})")
                    return device
    except Exception as e:
        logger.warning(f"Device detection failed: {e}")
    
    # Fallback to config
    fallback = config.MIC_DEVICE if direction == "input" else config.SPEAKER_DEVICE
    logger.warning(f"USB {direction} not found, using fallback: {fallback}")
    return fallback


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


def calibrate_noise(stream, duration: float = 1.5) -> float:
    """
    Measure ambient noise level over a short period.
    
    Uses 90th percentile of RMS values (robust to transient clicks/bumps)
    and applies a multiplier to set the speech threshold.
    
    Returns a threshold that speech must exceed.
    """
    chunks = int(duration / CHUNK_DURATION)
    rms_values = []
    
    for _ in range(chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_values.append(rms)
    
    # Use 90th percentile — more robust than mean (ignores transient spikes)
    noise_floor = np.percentile(rms_values, 90)
    multiplier = getattr(config, 'NOISE_MULTIPLIER', 2.5)
    threshold = max(noise_floor * multiplier, 300)  # minimum 300
    
    logger.info(
        f"🔇 Noise calibration: floor={noise_floor:.0f} (p90 of {len(rms_values)} samples), "
        f"multiplier={multiplier}x → threshold={threshold:.0f} RMS"
    )
    return threshold


def record_until_silence(output_path: str) -> str:
    """
    Record from mic until user stops speaking.
    
    Improvements over basic RMS threshold:
    1. Sliding-window RMS smoothing (avoids false triggers from noise spikes)
    2. Percentile-based noise calibration (robust to transient sounds)
    3. Minimum speech requirement before silence detection activates
    4. Trailing silence trimming (cleaner audio for Whisper)
    5. Generous max duration (30s) so it doesn't cut off mid-sentence
    
    Saves as 16kHz WAV for Whisper.
    """
    logger.info("🎤 Listening...")
    
    pa, stream = _open_mic()
    
    # Auto-calibrate noise threshold
    threshold = calibrate_noise(stream)
    
    frames = []
    rms_history = collections.deque(maxlen=RMS_WINDOW)  # sliding window
    silent_chunks = 0
    speech_chunks = 0
    has_speech = False
    last_speech_idx = 0  # index of last chunk with detected speech
    
    chunks_for_silence = int(config.SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(config.MAX_RECORD_DURATION / CHUNK_DURATION)
    min_speech_chunks = int(getattr(config, 'MIN_SPEECH_DURATION', 0.3) / CHUNK_DURATION)
    min_record_chunks = int(config.MIN_RECORD_DURATION / CHUNK_DURATION)
    
    for i in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate RMS for this chunk
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_history.append(rms)
        
        # Use smoothed RMS (average of sliding window) — much more stable
        smoothed_rms = np.mean(list(rms_history))
        
        is_speech = smoothed_rms > threshold
        
        if is_speech:
            speech_chunks += 1
            has_speech = True
            silent_chunks = 0
            last_speech_idx = i
        else:
            if has_speech:
                silent_chunks += 1
        
        # Debug logging every second
        if logger.isEnabledFor(logging.DEBUG) and i % 10 == 0:
            logger.debug(
                f"  chunk {i}: rms={rms:.0f} smoothed={smoothed_rms:.0f} "
                f"thresh={threshold:.0f} speech={is_speech} "
                f"silent={silent_chunks}/{chunks_for_silence}"
            )
        
        # Stop conditions:
        # 1. Have detected enough actual speech
        # 2. Past minimum recording duration
        # 3. Enough consecutive silence detected
        if (has_speech
                and speech_chunks >= min_speech_chunks
                and i >= min_record_chunks
                and silent_chunks >= chunks_for_silence):
            logger.info(
                f"✂️  Silence detected after {i * CHUNK_DURATION:.1f}s "
                f"({speech_chunks * CHUNK_DURATION:.1f}s speech, "
                f"{silent_chunks * CHUNK_DURATION:.1f}s silence)"
            )
            break
    else:
        if has_speech:
            logger.info(f"⏱️  Max duration reached ({config.MAX_RECORD_DURATION}s)")
        
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    if not has_speech:
        logger.info("No speech detected")
        return ""
    
    # Trim trailing silence (keep ~200ms for natural cutoff)
    keep_silent_chunks = int(0.2 / CHUNK_DURATION)
    trim_point = last_speech_idx + 1 + keep_silent_chunks
    if trim_point < len(frames):
        trimmed = len(frames) - trim_point
        frames = frames[:trim_point]
        logger.debug(f"Trimmed {trimmed} silent chunks ({trimmed * CHUNK_DURATION:.1f}s)")
    
    # Combine frames, downsample 44100→16000, save as WAV
    audio_44k = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_16k = _downsample(audio_44k, MIC_RATE, 16000)
    
    _save_wav(output_path, audio_16k, 16000)
    duration = len(audio_16k) / 16000
    logger.info(f"📼 Recorded {duration:.1f}s of audio (trimmed)")
    
    return output_path


def _downsample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Downsample audio using scipy's resample (high quality)."""
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
    """Check if a WAV file contains speech (basic energy check)."""
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


_speaker_device = None

def play(wav_path: str):
    """Play a WAV file through the USB speaker."""
    global _speaker_device
    if _speaker_device is None:
        _speaker_device = _find_usb_device("output")
    
    logger.info(f"🔊 Playing audio on {_speaker_device}...")
    try:
        subprocess.run(
            ["aplay", "-D", _speaker_device, wav_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"aplay failed: {e.stderr.decode() if e.stderr else e}")


def play_beep():
    """Play a short beep to indicate listening."""
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
            fade = 1.0 - (i / n_samples)
            sample = int(16000 * fade * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", sample))
