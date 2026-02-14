"""
Audio Utilities
===============

Record and play audio via PyAudio + Silero VAD for silence detection.

Uses neural Voice Activity Detection (Silero VAD) instead of RMS thresholds.
Falls back to RMS-based detection if the VAD model is unavailable.
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

import pyaudio

import config

logger = logging.getLogger(__name__)

# Mic native rate
MIC_RATE = 44100
CHUNK_DURATION = 0.1  # 100ms chunks
MIC_CHUNK = int(MIC_RATE * CHUNK_DURATION)


def _find_usb_device(direction: str = "input") -> str:
    """Find USB audio device card number dynamically."""
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
    fallback = config.MIC_DEVICE if direction == "input" else config.SPEAKER_DEVICE
    logger.warning(f"USB {direction} not found, using fallback: {fallback}")
    return fallback


def _open_mic():
    """Open the USB mic via PyAudio."""
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


def _load_vad():
    """Load the best available VAD backend."""
    try:
        from vad import create_vad
        threshold = getattr(config, 'VAD_THRESHOLD', 0.4)
        return create_vad(threshold=threshold)
    except Exception as e:
        logger.warning(f"VAD not available ({e}), falling back to RMS")
        return None


def _fast_downsample(audio_44k: np.ndarray) -> np.ndarray:
    """Fast downsample 44100→16000 for real-time VAD (linear interpolation)."""
    ratio = 16000 / 44100
    target_len = int(len(audio_44k) * ratio)
    indices = np.linspace(0, len(audio_44k) - 1, target_len)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(audio_44k) - 1)
    frac = indices - idx_floor
    return (audio_44k[idx_floor] * (1 - frac) + audio_44k[idx_ceil] * frac).astype(np.int16)


def record_until_silence(output_path: str) -> str:
    """
    Record from mic until user stops speaking.

    Uses Silero VAD (neural) for speech detection if available,
    falls back to smoothed RMS if not.

    Saves as 16kHz WAV for Whisper.
    """
    logger.info("🎤 Listening...")

    pa, stream = _open_mic()
    vad = _load_vad()

    if vad:
        result = _record_with_vad(stream, vad, output_path)
    else:
        result = _record_with_rms(stream, output_path)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return result


def _record_with_vad(stream, vad, output_path: str) -> str:
    """Record using VAD backend for speech/silence detection."""
    silence_duration = config.SILENCE_DURATION
    max_duration = config.MAX_RECORD_DURATION
    min_speech = getattr(config, 'MIN_SPEECH_DURATION', 0.3)

    frames = []
    silent_time = 0.0
    speech_time = 0.0
    has_speech = False
    last_speech_idx = 0

    max_chunks = int(max_duration / CHUNK_DURATION)
    min_speech_time = min_speech

    logger.info(f"🧠 Using VAD: {type(vad).__name__}")

    for i in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)

        # Downsample chunk for VAD (44100→16000)
        samples_44k = np.frombuffer(data, dtype=np.int16)
        samples_16k = _fast_downsample(samples_44k.astype(np.float64))

        is_speech = vad.is_speech(samples_16k)

        if is_speech:
            speech_time += CHUNK_DURATION
            has_speech = True
            silent_time = 0.0
            last_speech_idx = i
        else:
            if has_speech:
                silent_time += CHUNK_DURATION

        # Debug logging every second
        if logger.isEnabledFor(logging.DEBUG) and i % 10 == 0:
            logger.debug(
                f"  [{i * CHUNK_DURATION:.1f}s] "
                f"speech={is_speech} silent={silent_time:.1f}s"
            )

        # Stop: enough speech detected + sustained silence
        if (has_speech
                and speech_time >= min_speech_time
                and silent_time >= silence_duration):
            logger.info(
                f"✂️  Silence detected after {i * CHUNK_DURATION:.1f}s "
                f"({speech_time:.1f}s speech, {silent_time:.1f}s silence)"
            )
            break
    else:
        if has_speech:
            logger.info(f"⏱️  Max duration reached ({max_duration}s)")

    vad.reset()

    if not has_speech:
        logger.info("No speech detected")
        return ""

    # Trim trailing silence (keep ~200ms for natural cutoff)
    keep_chunks = int(0.2 / CHUNK_DURATION)
    trim_point = last_speech_idx + 1 + keep_chunks
    if trim_point < len(frames):
        trimmed = len(frames) - trim_point
        frames = frames[:trim_point]
        logger.debug(f"Trimmed {trimmed * CHUNK_DURATION:.1f}s of trailing silence")

    # Save as 16kHz WAV
    audio_44k = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_16k = _hq_downsample(audio_44k, MIC_RATE, 16000)
    _save_wav(output_path, audio_16k, 16000)
    duration = len(audio_16k) / 16000
    logger.info(f"📼 Recorded {duration:.1f}s of audio")
    return output_path


def _record_with_rms(stream, output_path: str) -> str:
    """Fallback: record using smoothed RMS for silence detection."""
    # Calibrate noise
    threshold = _calibrate_noise(stream)

    frames = []
    rms_history = collections.deque(maxlen=5)
    silent_chunks = 0
    speech_chunks = 0
    has_speech = False
    last_speech_idx = 0

    chunks_for_silence = int(config.SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(config.MAX_RECORD_DURATION / CHUNK_DURATION)
    min_speech_chunks = int(getattr(config, 'MIN_SPEECH_DURATION', 0.3) / CHUNK_DURATION)
    min_record_chunks = int(config.MIN_RECORD_DURATION / CHUNK_DURATION)

    logger.info(f"📊 Using RMS detection (threshold={threshold:.0f})")

    for i in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_history.append(rms)
        smoothed_rms = np.mean(list(rms_history))

        if smoothed_rms > threshold:
            speech_chunks += 1
            has_speech = True
            silent_chunks = 0
            last_speech_idx = i
        else:
            if has_speech:
                silent_chunks += 1

        if (has_speech
                and speech_chunks >= min_speech_chunks
                and i >= min_record_chunks
                and silent_chunks >= chunks_for_silence):
            logger.info(f"✂️  Silence after {i * CHUNK_DURATION:.1f}s")
            break

    if not has_speech:
        logger.info("No speech detected")
        return ""

    # Trim trailing silence
    keep_chunks = int(0.2 / CHUNK_DURATION)
    trim_point = last_speech_idx + 1 + keep_chunks
    if trim_point < len(frames):
        frames = frames[:trim_point]

    audio_44k = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_16k = _hq_downsample(audio_44k, MIC_RATE, 16000)
    _save_wav(output_path, audio_16k, 16000)
    duration = len(audio_16k) / 16000
    logger.info(f"📼 Recorded {duration:.1f}s of audio")
    return output_path


def _calibrate_noise(stream, duration: float = 1.5) -> float:
    """Measure ambient noise using 90th percentile RMS."""
    chunks = int(duration / CHUNK_DURATION)
    rms_values = []
    for _ in range(chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_values.append(rms)
    noise_floor = np.percentile(rms_values, 90)
    multiplier = getattr(config, 'NOISE_MULTIPLIER', 2.5)
    threshold = max(noise_floor * multiplier, 300)
    logger.info(f"🔇 Noise: p90={noise_floor:.0f} → threshold={threshold:.0f}")
    return threshold


def _hq_downsample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """High-quality downsample for final WAV output."""
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
