#!/usr/bin/env python3
"""Quick mic diagnostic — shows RMS values for silence vs speech."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import audio
import config

config.TMP_DIR.mkdir(parents=True, exist_ok=True)
pa, stream = audio._open_mic()

print("=== SILENCE (5s) — stay quiet ===")
silence_rms = []
for i in range(50):
    data = stream.read(audio.MIC_CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    silence_rms.append(rms)
    print(f"  chunk {i:2d}: RMS={rms:.0f}")

print()
print("=== SPEECH (5s) — talk now! ===")
speech_rms = []
for i in range(50):
    data = stream.read(audio.MIC_CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    speech_rms.append(rms)
    print(f"  chunk {i:2d}: RMS={rms:.0f}")

stream.close()
pa.terminate()

print()
print("=== SUMMARY ===")
print(f"Silence: mean={np.mean(silence_rms):.0f}, p90={np.percentile(silence_rms, 90):.0f}, max={np.max(silence_rms):.0f}")
print(f"Speech:  mean={np.mean(speech_rms):.0f}, p90={np.percentile(speech_rms, 90):.0f}, max={np.max(speech_rms):.0f}")
ratio = np.mean(speech_rms) / max(np.mean(silence_rms), 1)
print(f"Ratio:   {ratio:.1f}x (need >2.5x for RMS detection to work)")
if ratio < 2.0:
    print("⚠️  Ratio too low — RMS detection won't work well. Need Silero VAD.")
elif ratio < 3.0:
    print("⚠️  Ratio marginal — may need tuning.")
else:
    print("✅ Ratio looks good for RMS detection.")
