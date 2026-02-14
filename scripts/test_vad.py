#!/usr/bin/env python3
"""Diagnose Silero VAD — check model, test with sine wave + mic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# 1. Check model
print("=== MODEL INFO ===")
try:
    import onnxruntime
    from vad import SileroVAD
    vad = SileroVAD()
    print(f"Version: v{vad.version}")
    print(f"State dim: {vad._state_dim}")
    for inp in vad.session.get_inputs():
        print(f"  Input: {inp.name} shape={inp.shape} type={inp.type}")
    for out in vad.session.get_outputs():
        print(f"  Output: {out.name} shape={out.shape} type={out.type}")
except Exception as e:
    print(f"FAILED to load VAD: {e}")
    sys.exit(1)

# 2. Test with silence (zeros)
print("\n=== TEST: SILENCE (zeros) ===")
silence = np.zeros(1600, dtype=np.int16)
vad.reset()
prob = vad.process_chunk(silence)
print(f"  prob={prob:.4f} (should be ~0.0)")

# 3. Test with sine wave (fake speech-like signal)
print("\n=== TEST: SINE WAVE 440Hz ===")
t = np.linspace(0, 0.1, 1600, dtype=np.float32)
sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
vad.reset()
prob = vad.process_chunk(sine)
print(f"  prob={prob:.4f} (might be low - sine != speech)")

# 4. Test with noise (random)
print("\n=== TEST: RANDOM NOISE ===")
noise = np.random.randint(-5000, 5000, 1600, dtype=np.int16)
vad.reset()
prob = vad.process_chunk(noise)
print(f"  prob={prob:.4f}")

# 5. Test raw inference with known good input
print("\n=== TEST: RAW INFERENCE ===")
# Create a window directly
window = np.random.randn(1, 512).astype(np.float32) * 0.1
sr = np.array(16000, dtype=np.int64)
if vad.version == 5:
    state = np.zeros((2, 1, 128), dtype=np.float32)
    inputs = {"input": window, "state": state, "sr": sr}
else:
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    inputs = {"input": window, "h": h, "c": c, "sr": sr}

try:
    results = vad.session.run(None, inputs)
    print(f"  Raw output: {results[0]} (shape={results[0].shape})")
    print(f"  State shape: {results[1].shape}")
except Exception as e:
    print(f"  INFERENCE FAILED: {e}")
    # Try to understand what the model actually wants
    print("\n  Trying to discover correct input shapes...")
    for inp in vad.session.get_inputs():
        print(f"    {inp.name}: shape={inp.shape} type={inp.type}")

# 6. Test with actual mic audio (5 seconds)
print("\n=== TEST: LIVE MIC (5s) — speak during this! ===")
import audio
import config
config.TMP_DIR.mkdir(parents=True, exist_ok=True)

pa, stream = audio._open_mic()
vad.reset()

for i in range(50):
    data = stream.read(audio.MIC_CHUNK, exception_on_overflow=False)
    samples_44k = np.frombuffer(data, dtype=np.int16)

    # Show raw audio stats
    rms = np.sqrt(np.mean(samples_44k.astype(np.float32) ** 2))

    # Downsample
    samples_16k = audio._fast_downsample(samples_44k.astype(np.float64))

    # Show downsampled stats
    rms_16k = np.sqrt(np.mean(samples_16k.astype(np.float32) ** 2))

    # VAD
    prob = vad.process_chunk(samples_16k)

    if i % 5 == 0 or prob > 0.1:
        print(f"  [{i:2d}] rms_44k={rms:.0f} rms_16k={rms_16k:.0f} vad_prob={prob:.3f}")

stream.close()
pa.terminate()
print("\nDone!")
