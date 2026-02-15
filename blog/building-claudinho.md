# Building a $190 AI Voice Assistant on Raspberry Pi 5

*How I built a voice assistant that controls my home — and the AI helped design itself.*

---

I wanted a voice assistant that actually worked for me. Not one locked into a corporate ecosystem. Not one that sends everything to the cloud. Something I could hack on, extend, and own.

So I built **Claudinho** — a DIY voice assistant running on a Raspberry Pi 5, powered by Claude through [OpenClaw](https://github.com/openclaw/openclaw). It listens for a wake word, transcribes my speech, thinks with Claude, speaks back, and controls my smart home lights. Total hardware cost: **$190**.

This is the story of building it — the decisions, the debugging, and the surprises along the way.

## The Hardware: Keep It Stupid Simple

I spent way too long researching audio hardware before landing on the simplest possible setup:

| Component | Price |
|-----------|-------|
| Raspberry Pi 5 (8GB) | $135 |
| Mini USB Microphone (Adafruit #3367) | $6 |
| Mini USB Speaker (Adafruit #3369) | $13 |
| Official 27W USB-C PSU | $14 |
| Official Case + Fan | $12 |
| 32GB microSD card | $10 |
| **Total** | **$190** |

Why USB audio? Because I2S DACs and HAT-based audio boards mean soldering, custom drivers, and debugging ALSA configs. USB devices just show up as ALSA cards. Plug them in and they work.

The Pi 5 was chosen over the Pi 4 because — well, the Pi 4 was literally out of stock everywhere. The extra horsepower turned out to be nice for running Whisper locally.

## The Software Stack

The architecture is a pipeline with each stage handling one job:

```
Wake Word → Record → STT → Claude → TTS → Speaker
```

Here's what runs each stage:

- **Wake Word:** [openWakeWord](https://github.com/dscripka/openWakeWord) — open-source, ONNX-based, runs on-device. Started with "Hey Jarvis" as the trigger word.
- **Voice Activity Detection:** [webrtcvad](https://github.com/wiseman/py-webrtcvad) — Google's battle-tested VAD. More on why this was critical later.
- **Speech-to-Text:** [Groq](https://groq.com) Whisper API — cloud transcription in under a second. Falls back to local [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) if offline.
- **Intelligence:** [OpenClaw](https://github.com/openclaw/openclaw) gateway → Claude Sonnet. OpenClaw runs directly on the Pi as a systemd service, giving Claude session memory and tool access.
- **Text-to-Speech:** [Inworld AI](https://inworld.ai) cloud TTS. Falls back to local [Piper TTS](https://github.com/rhasspy/piper) if the API is down.
- **Smart Home:** [python-kasa](https://github.com/python-kasa/python-kasa) for TP-Link Kasa device control over local network.

The key insight: **the Pi handles the ears (wake word + VAD) locally, but the brain and voice are in the cloud.** This gives you the responsiveness of local detection with the intelligence of Claude.

## Day 1: "Hello World" on a Pi

Flashing Raspberry Pi OS Lite (headless, no desktop — we don't need pixels) was the easy part. Raspberry Pi Imager lets you pre-configure WiFi, SSH, and hostname before first boot. I set the hostname to `claudinho` and SSH'd right in:

```bash
ssh claudinho@claudinho.local
```

Then the dependency adventure began.

### The Python 3.13 Problem

Raspberry Pi OS (Debian Trixie) ships with Python 3.13. This is too new for `tflite-runtime`, which openWakeWord normally uses for inference. The fix was straightforward — openWakeWord also supports ONNX Runtime:

```bash
pip install openwakeword --no-deps
pip install onnxruntime numpy scipy tqdm scikit-learn
```

No TFLite, no problem. The ONNX backend works identically.

### Building Whisper.cpp

Whisper.cpp compiled cleanly on the Pi 5 with ARM NEON optimizations:

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp && make -j4
./models/download-ggml-model.sh base
```

The `base` model (multilingual) transcribes 3 seconds of audio in about 3.7 seconds on the Pi 5. Not real-time, but good enough — especially since we later moved to cloud STT.

I initially tried the `tiny` model for speed, but it couldn't reliably detect Portuguese. The `base` model handles both English and Portuguese well with `-l auto` for automatic language detection.

### Audio: The Deceptively Simple Part

Getting audio working on Linux is one of those things that sounds trivial and isn't. The USB mic only supports 44100Hz (not the 16000Hz that Whisper expects), so every recording needs to be downsampled. USB device card numbers change across reboots — card 0 might become card 2 after a restart.

I solved both problems:

```python
def _find_usb_device(direction="input"):
    """Scan ALSA for USB devices — survives card number changes."""
    cmd = "arecord -l" if direction == "input" else "aplay -l"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "USB" in line and "card" in line:
            card = re.search(r"card (\d+)", line).group(1)
            return f"plughw:{card},0"
```

Auto-detection at runtime. No hardcoded card numbers. The Pi can reboot all day and the right devices will be found.

## The First Working Loop

After a day of wrestling with dependencies and audio configs, the magic moment arrived. I said "Hey Jarvis," the Pi beeped, I spoke, and Claude responded through the speaker.

```
👂 Listening for wake word...
👋 Wake word detected!
🎤 Listening...
✂️  Silence detected after 3.2s
📝 [en] User: What's the weather like?
🤔 Thinking...
💬 Claudinho: Not sure about the weather right now, but it's always sunny in my circuits.
🔊 Playing audio...
👂 Listening for wake word...
```

It worked. It was slow (about 10 seconds end-to-end), the mic was terrible, and the silence detection was flaky — but it worked.

## The Mic Problem: When Simple RMS Fails

The $6 USB mic was the weakest link. Here's what the signal looked like:

```
Silence: RMS ~310, peaks to 460
Speech:  RMS ~620, peaks to 1600
Ratio:   2.0x
```

A 2x signal-to-noise ratio is awful. Normal speech frequently dips back to noise-floor levels between syllables. My initial approach — "if RMS drops below threshold for 1.5 seconds, stop recording" — simply didn't work. The noise floor was too close to speech levels.

I tried three approaches:

### Attempt 1: Smoothed RMS (failed)

Sliding window average over 500ms of RMS values. Better than per-chunk RMS, but still couldn't reliably distinguish silence from quiet speech. A 2x SNR is just too narrow.

### Attempt 2: Silero VAD ONNX (failed spectacularly)

Silero VAD is a neural network trained specifically for voice activity detection. I downloaded the ONNX model and wrote a custom inference wrapper. The model loaded fine, ran without errors, and returned... `0.001` probability for everything. Speech, silence, noise — all 0.001.

Turns out the model from the GitHub raw URL didn't match the inference API I'd written. Processing chunks larger than 512 samples crashed with LSTM shape errors. After an hour of debugging tensor shapes, I gave up on the manual approach.

### Attempt 3: webrtcvad (worked immediately)

```bash
pip install webrtcvad
```

Google's WebRTC Voice Activity Detection. Written in C, battle-tested in millions of video calls. Aggressiveness level 3 (most aggressive filtering). It just worked. On the first try. With a $6 USB mic with a 2x SNR.

The lesson: sometimes the boring, proven technology beats the fancy neural network.

I built a multi-backend VAD system that tries silero-vad (pip package), then webrtcvad, then falls back to RMS. In practice, webrtcvad is all you need.

## The Speed Problem: 10 Seconds Is an Eternity

With the working pipeline, I measured each stage:

```
Silence detection:  1.5s
Local Whisper STT:  4.0s  ← biggest bottleneck
Claude thinking:    4.0s
Inworld TTS:        2.0s
Total:            ~10s after you stop talking
```

10 seconds between finishing your sentence and hearing a response. For a voice assistant, that's a lifetime. People expect responses in 1-2 seconds. ChatGPT's voice mode has spoiled everyone.

### Fix 1: Cloud STT with Groq (~4s saved)

Local Whisper on the Pi takes 4 seconds. Groq's Whisper API (large-v3-turbo model) takes under 0.5 seconds for the same audio, at $0.04/hour — basically free.

```python
response = requests.post(
    "https://api.groq.com/openai/v1/audio/transcriptions",
    headers={"Authorization": f"Bearer {api_key}"},
    files={"file": ("audio.wav", f, "audio/wav")},
    data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
)
```

Local Whisper.cpp stays as an automatic fallback if the network is down. Best of both worlds.

### Fix 2: Reduce silence wait (~0.7s saved)

With webrtcvad accurately detecting speech boundaries, I could safely reduce the silence duration from 1.5s to 0.8s. The VAD is reliable enough that we don't need a long confirmation window.

### Fix 3: Streaming LLM + Sentence-Level TTS (~3s saved)

This was the big architectural change. Instead of waiting for Claude's complete response before starting TTS, I stream the response and play each sentence as it arrives:

```
BEFORE (sequential):
  Wait for full LLM [======6s======] → TTS [==2s==] → Play
  First audio: 8 seconds

AFTER (streaming):
  Stream LLM → first sentence ready [==2s==] → TTS [=1s=] → PLAY
               second sentence [==2s==] → TTS → PLAY
  First audio: 3 seconds
```

The implementation splits the SSE stream at sentence boundaries (periods, question marks, exclamation points), synthesizes each sentence independently, and plays them sequentially:

```python
for sentence in assistant.chat_stream_sentences(text):
    wav = tts.synthesize(sentence, language=language)
    audio.play(wav)
```

Each sentence gets its own WAV file (rotating `response_0.wav` through `response_9.wav`) to avoid overwriting a file that's still playing.

**Result: ~3-4 seconds to first audio.** Down from 10. The assistant now *feels* responsive.

## Smart Home: "Turn On the Lights"

With the voice pipeline solid, adding smart home control was natural. I use TP-Link Kasa lights, which can be controlled over the local network with `python-kasa`.

The approach is simple: on startup, discover all Kasa devices on the network. Include the device list in Claude's system prompt. Tell Claude to use `<<action:device>>` tags for smart home commands. The assistant strips these tags before TTS and executes them:

```python
# Claude responds: "Sure, turning on the sofa light. <<turn_on:Sofa>>"
clean_text, actions = extract_actions(response)
execute_actions(actions)  # Actually turns on the light
tts.synthesize(clean_text)  # Speaks: "Sure, turning on the sofa light."
```

No complicated tool-calling infrastructure. No separate home automation server. Just a structured tag format that Claude follows reliably, and 150 lines of python-kasa wrapper code.

"Hey Jarvis, turn on the lights" → lights turn on → "Done, all lights are on." The whole thing takes about 4 seconds.

## What I'd Do Differently

**Start with cloud STT.** I spent a day building and testing local Whisper before realizing it was the biggest bottleneck. For a voice assistant that needs internet for the LLM anyway, local STT doesn't buy you much.

**Don't cheap out on the mic.** The $6 Adafruit mic works, but barely. A $20-30 USB mic with better SNR would have saved hours of VAD debugging. The mic is the most critical component in the pipeline — everything downstream depends on clean audio.

**Stream from day one.** The sequential pipeline (record → transcribe → think → synthesize → play) is easier to build but gives terrible latency. Streaming changes everything. Build the pipeline with streaming in mind from the start.

## The Meta Moment

Here's the weird part: Claudinho — the voice assistant — was largely designed and built by **Claudinho** — the AI agent. The same Claude that powers the voice responses also wrote most of the code, debugged the ALSA issues, implemented the VAD system, and optimized the pipeline.

The agent runs on my Mac via OpenClaw, with access to the codebase on GitHub. I'd describe what I wanted ("the mic doesn't detect when I stop talking"), it would diagnose the problem ("your USB mic has a 2.0x SNR — RMS detection can't work"), write the fix, push to GitHub, and I'd `git pull` on the Pi.

An AI assistant that builds itself. We're living in interesting times.

## Try It Yourself

Everything is open source: **[github.com/claudinhocoding/claudinho](https://github.com/claudinhocoding/claudinho)**

The total cost is about $190 in hardware, plus whatever you spend on API calls (Groq STT is basically free, Claude is a few cents per conversation, Inworld TTS has a free tier).

The setup takes about 2-3 hours if you follow the README. The hardest part is waiting for Whisper.cpp to compile.

---

*Built by Thiago and Claudinho 🛠️ — a human and the AI that designed itself.*
