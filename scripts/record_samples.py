#!/usr/bin/env python3
"""
Record wake word training samples.

Plays a beep, records ~2 seconds, saves as numbered WAV files.
Say the wake word once per recording. Aim for 50-100 samples.

Usage:
    python scripts/record_samples.py --word claudinho --count 75

Vary your delivery naturally:
  - Normal, loud, soft, whispered
  - Close to mic, further away
  - Different speeds
  - Different intonation
"""

import argparse
import os
import sys
import wave
import struct
import math
import time
import subprocess

import numpy as np

# Add src/ to path for config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config

# Suppress ALSA warnings
import ctypes
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

MIC_RATE = 44100
RECORD_SECONDS = 2.5  # seconds per sample
CHUNK = 1024


def find_usb_speaker():
    """Find USB speaker card number."""
    import re
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB" in line and "card" in line:
                match = re.search(r"card (\d+)", line)
                if match:
                    return f"plughw:{match.group(1)},0"
    except Exception:
        pass
    return config.SPEAKER_DEVICE


def generate_beep(path, freq=660, duration=0.12):
    """Generate a short beep WAV."""
    sr = 16000
    n = int(sr * duration)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            t = i / sr
            fade = 1.0 - (i / n)
            sample = int(12000 * fade * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", sample))


def play_beep(beep_path, speaker):
    """Play the beep sound."""
    subprocess.run(
        ["aplay", "-D", speaker, beep_path],
        capture_output=True,
    )


def record_sample(pa, device_index, duration=RECORD_SECONDS):
    """Record a single sample, return raw audio bytes."""
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    
    frames = []
    for _ in range(int(MIC_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    return b"".join(frames)


def save_wav(path, audio_bytes, rate=16000):
    """Save audio as 16kHz WAV (downsampled from 44100)."""
    from scipy.signal import resample
    
    audio_44k = np.frombuffer(audio_bytes, dtype=np.int16)
    target_len = int(len(audio_44k) * rate / MIC_RATE)
    audio_16k = resample(audio_44k, target_len).astype(np.int16)
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio_16k.tobytes())


def find_usb_mic(pa):
    """Find USB mic device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "usb" in info.get("name", "").lower() and info["maxInputChannels"] > 0:
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Record wake word samples")
    parser.add_argument("--word", default="claudinho", help="Wake word being recorded")
    parser.add_argument("--count", type=int, default=75, help="Number of samples to record")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()
    
    # Setup output directory
    out_dir = args.output or os.path.join(
        os.path.dirname(__file__), "..", "training", "positive", args.word
    )
    os.makedirs(out_dir, exist_ok=True)
    
    # Find existing samples to continue numbering
    existing = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    start_num = len(existing)
    
    print(f"\n🎤 Wake Word Sample Recorder")
    print(f"   Word: \"{args.word}\"")
    print(f"   Samples to record: {args.count}")
    print(f"   Output: {out_dir}")
    if start_num > 0:
        print(f"   Continuing from sample #{start_num}")
    print(f"\n   Tips:")
    print(f"   - Say \"{args.word}\" once per recording")
    print(f"   - Vary: normal, loud, soft, whispered")
    print(f"   - Vary: close, medium, far from mic")
    print(f"   - Vary: fast, normal, slow")
    print(f"   - Press Ctrl+C to stop early\n")
    
    # Setup audio
    speaker = find_usb_speaker()
    beep_path = "/tmp/claudinho/rec_beep.wav"
    os.makedirs("/tmp/claudinho", exist_ok=True)
    generate_beep(beep_path)
    
    pa = pyaudio.PyAudio()
    mic_idx = find_usb_mic(pa)
    
    if mic_idx is not None:
        info = pa.get_device_info_by_index(mic_idx)
        print(f"   Mic: [{mic_idx}] {info['name']}")
    print(f"   Speaker: {speaker}\n")
    
    input("Press Enter to start recording...")
    print()
    
    recorded = 0
    try:
        for i in range(args.count):
            num = start_num + i
            print(f"  [{num + 1}/{start_num + args.count}] ", end="", flush=True)
            
            # Beep
            play_beep(beep_path, speaker)
            time.sleep(0.15)  # tiny gap after beep
            
            # Record
            print(f"🔴 Say \"{args.word}\"...", end="", flush=True)
            audio = record_sample(pa, mic_idx)
            
            # Save
            filename = f"{args.word}_{num:04d}.wav"
            filepath = os.path.join(out_dir, filename)
            save_wav(filepath, audio)
            
            print(f" ✅ saved {filename}")
            recorded += 1
            
            time.sleep(0.3)  # small pause between samples
            
    except KeyboardInterrupt:
        print(f"\n\nStopped early.")
    finally:
        pa.terminate()
    
    total = start_num + recorded
    print(f"\n🎉 Done! Recorded {recorded} new samples ({total} total)")
    print(f"   Saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
