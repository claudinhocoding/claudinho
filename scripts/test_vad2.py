#!/usr/bin/env python3
"""VAD diagnosis round 2: try different chunk sizes + official package."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import onnxruntime

# Load model directly
model_path = os.path.expanduser("~/claudinho/models/silero_vad.onnx")
print(f"Model size: {os.path.getsize(model_path)} bytes")
session = onnxruntime.InferenceSession(model_path)

# Check model file isn't HTML/garbage
with open(model_path, "rb") as f:
    header = f.read(8)
    print(f"File header (hex): {header.hex()}")
    # Valid ONNX starts with \x08 or protobuf marker
    if header[:2] == b'<!':
        print("ERROR: Model file is HTML, not ONNX! Download failed.")
        sys.exit(1)

# Test 1: Different chunk sizes with synthetic speech-like audio
print("\n=== TEST: DIFFERENT CHUNK SIZES ===")
state = np.zeros((2, 1, 128), dtype=np.float32)
sr = np.array(16000, dtype=np.int64)

for chunk_size in [512, 1024, 1536, 4096, 8000, 16000]:
    state = np.zeros((2, 1, 128), dtype=np.float32)
    # Create speech-like signal (modulated noise)
    t = np.linspace(0, chunk_size / 16000, chunk_size)
    signal = (np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    x = signal.reshape(1, -1)
    out, state = session.run(None, {"input": x, "state": state, "sr": sr})
    prob = float(np.squeeze(out))
    print(f"  chunk_size={chunk_size:5d}: prob={prob:.4f}")

# Test 2: Amplified audio from mic
print("\n=== TEST: MIC WITH AMPLIFICATION (5s) — speak! ===")
import audio
import config
config.TMP_DIR.mkdir(parents=True, exist_ok=True)

pa, stream = audio._open_mic()
state = np.zeros((2, 1, 128), dtype=np.float32)

for i in range(50):
    data = stream.read(audio.MIC_CHUNK, exception_on_overflow=False)
    samples_44k = np.frombuffer(data, dtype=np.int16)
    samples_16k = audio._fast_downsample(samples_44k.astype(np.float64))

    # Normalize to float32 [-1, 1]
    audio_f32 = samples_16k.astype(np.float32) / 32768.0
    peak = np.max(np.abs(audio_f32))

    # Try full chunk at once (not 512-sample windows)
    x = audio_f32.reshape(1, -1)
    out, state = session.run(None, {"input": x, "state": state, "sr": sr})
    prob = float(np.squeeze(out))

    if i % 5 == 0 or prob > 0.1:
        print(f"  [{i:2d}] peak={peak:.4f} prob={prob:.4f}")

stream.close()
pa.terminate()

# Test 3: Try official silero-vad package
print("\n=== TEST: OFFICIAL PACKAGE ===")
try:
    from silero_vad import load_silero_vad
    model = load_silero_vad(onnx=True)
    print(f"Official model loaded: {type(model)}")

    # Test with 1s of synthetic audio
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    signal = np.sin(2 * np.pi * 300 * t) * 0.5
    import torch
    tensor = torch.from_numpy(signal)
    prob = model(tensor, 16000)
    print(f"  Synthetic speech prob: {prob}")
except ImportError:
    print("  silero-vad not installed. Install with: pip install silero-vad")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone!")
