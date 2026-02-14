# Claudinho 🎤🤖

A DIY voice assistant powered by [OpenClaw](https://github.com/openclaw/openclaw) + Claude, running on Raspberry Pi 5.

**Local wake word. Local STT. Cloud TTS. Cloud intelligence.**

Say the wake word → speak → get a spoken response from Claude. Audio processing (wake word + STT) happens on-device — text goes to the cloud for intelligence and TTS.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌───────────┐    ┌───────────┐           │
│   │ USB Mic  │───▶│openWakeWrd│───▶│ Whisper   │           │
│   │          │    │ Wake Word │    │ STT       │           │
│   └──────────┘    │ Detection │    │ (base)    │           │
│                   └───────────┘    └───────────┘           │
│                                          │                  │
│        ┌──────────┐                      ▼                  │
│        │ webrtcvad│─── silence ──▶ stop recording          │
│        │ (VAD)    │                      │                  │
│        └──────────┘                      ▼                  │
│                                    ┌───────────┐           │
│                                    │ OpenClaw  │◀──────┐   │
│                                    │ Gateway   │       │   │
│                                    └───────────┘   Claude  │
│                                          │          API    │
│   ┌──────────┐    ┌───────────┐    ┌───────────┐   │      │
│   │ Speaker  │◀───│ Inworld   │◀───│ Response  │───┘      │
│   │          │    │ TTS       │    │           │           │
│   └──────────┘    └───────────┘    └───────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Model | Price |
|-----------|-------|-------|
| Computer | Raspberry Pi 5 (8GB) | ~$135 |
| Microphone | [Adafruit Mini USB Mic #3367](https://www.adafruit.com/product/3367) | ~$6 |
| Speaker | [Adafruit Mini USB Speaker #3369](https://www.adafruit.com/product/3369) | ~$13 |
| Power | Official Raspberry Pi 27W USB-C PSU | ~$14 |
| Case | Official Raspberry Pi 5 Case + Fan | ~$12 |
| Storage | 32GB microSD card | ~$10 |
| **Total** | | **~$190** |

## Software Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| OS | Raspberry Pi OS 64-bit Lite (Debian Trixie) | Base system |
| Wake Word | [openWakeWord](https://github.com/dscripka/openWakeWord) | Wake word detection (ONNX) |
| VAD | [webrtcvad](https://github.com/wiseman/py-webrtcvad) | Voice activity detection for silence detection |
| STT | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Speech-to-text (base model, ~3.7s for 3s audio) |
| LLM | [OpenClaw](https://github.com/openclaw/openclaw) + Claude | Intelligence |
| TTS | [Inworld AI](https://inworld.ai/) | Cloud text-to-speech (Theodore voice) |
| TTS Fallback | [Piper](https://github.com/rhasspy/piper) | Local TTS fallback (EN + PT-BR voices) |
| Audio | PyAudio + ALSA | Mic input / speaker output |

## Project Structure

```
claudinho/
├── src/
│   ├── main.py           # Entry point (wake word loop + no-wake mode)
│   ├── config.py          # Hardware paths, device settings, thresholds
│   ├── wake_word.py       # openWakeWord detection (44100→16kHz resample)
│   ├── audio.py           # Record with VAD silence detection, play via aplay
│   ├── vad.py             # Voice Activity Detection (webrtcvad / silero / RMS)
│   ├── stt.py             # Whisper.cpp CLI with auto language detection
│   ├── tts.py             # Inworld TTS with Piper fallback
│   └── assistant.py       # OpenClaw gateway chat completions API
├── scripts/
│   ├── install.sh         # Full Pi setup script
│   ├── record_samples.py  # Record wake word training samples
│   └── claudinho.service  # systemd unit (auto-start on boot)
├── training/              # Wake word training data
├── notebooks/             # Colab notebooks for model training
├── requirements.txt
└── README.md
```

## Setup

### 1. Flash Raspberry Pi OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/):
- Choose **Raspberry Pi OS (64-bit) Lite**
- Set hostname to `claudinho`, enable SSH, configure WiFi
- Flash to microSD card and boot

### 2. SSH into the Pi

```bash
ssh claudinho@claudinho.local
```

### 3. Install dependencies

```bash
# Clone the repo
git clone https://github.com/claudinhocoding/claudinho.git
cd claudinho

# Create Python virtual environment
python3 -m venv ~/claudinho/venv
source ~/claudinho/venv/bin/activate

# Install Python packages
pip install pyaudio numpy scipy requests webrtcvad
pip install openwakeword --no-deps
pip install onnxruntime scikit-learn tqdm

# Build Whisper.cpp
cd ~
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j4
./models/download-ggml-model.sh base

# (Optional) Install Piper TTS for local fallback
cd ~
mkdir piper && cd piper
# Download the Piper binary for aarch64 from:
# https://github.com/rhasspy/piper/releases
# Then download voice models:
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx.json

# Install OpenClaw
npm install -g openclaw
openclaw configure
```

### 4. Configure

Edit `src/config.py` with your OpenClaw gateway token:

```python
OPENCLAW_URL = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "your-gateway-token-here"
```

### 5. Test it

```bash
source ~/claudinho/venv/bin/activate
cd ~/claudinho

# Quick test (press Enter to talk, no wake word needed)
python3 src/main.py --no-wake

# Full mode with wake word
python3 src/main.py

# Debug mode (shows VAD decisions, RMS values, timing)
python3 src/main.py --no-wake --debug
python3 src/main.py --debug
```

### 6. Run as a service (auto-start on boot)

```bash
sudo cp scripts/claudinho.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable claudinho
sudo systemctl start claudinho
```

## Managing the Service

```bash
# Check status
sudo systemctl status claudinho

# Follow logs in real-time
journalctl -u claudinho -f

# View recent logs
journalctl -u claudinho --no-pager -n 50

# Restart (after code changes / git pull)
cd ~/claudinho && git pull
sudo systemctl restart claudinho

# Stop
sudo systemctl stop claudinho

# Start
sudo systemctl start claudinho
```

## Tuning the VAD (Voice Activity Detection)

The voice activity detection uses **webrtcvad** to determine when you start and stop speaking. This is critical for a good experience — if it's too aggressive, it cuts you off; too loose, and it records forever.

### Aggressiveness (0–3)

Set in `src/vad.py` → `WebRTCVADBackend(aggressiveness=3)`:

| Level | Behavior |
|-------|----------|
| 0 | Least aggressive — keeps recording through pauses, less likely to cut off |
| 1 | Mild filtering |
| 2 | Moderate — good balance for most environments |
| **3** | **Most aggressive** — stops quickly after speech ends (current default) |

- If it's **cutting you off too early** (mid-sentence), lower to `2`
- If it's **recording too long after you stop**, it's already at max — adjust `SILENCE_DURATION` in `config.py` instead (lower = stops sooner)

### Other tuning knobs (in `src/config.py`)

| Setting | Default | What it does |
|---------|---------|-------------|
| `SILENCE_DURATION` | `1.5` | Seconds of silence needed to stop recording |
| `MAX_RECORD_DURATION` | `30` | Max recording length in seconds |
| `MIN_SPEECH_DURATION` | `0.3` | Minimum speech before silence detection kicks in |
| `MIN_RECORD_DURATION` | `0.8` | Minimum recording length before stopping |

### VAD Backend Priority

The system tries VAD backends in this order:

1. **silero-vad** — Neural VAD (most accurate, `pip install silero-vad`)
2. **webrtcvad** — Google's C-based VAD (fast, reliable, `pip install webrtcvad`) ← recommended
3. **RMS threshold** — Simple volume-based (last resort, no install needed)

## SSH Quick Reference

```bash
# Connect from your Mac/PC
ssh claudinho@claudinho.local

# Check voice assistant
sudo systemctl status claudinho
journalctl -u claudinho -f

# Quick manual test
source ~/claudinho/venv/bin/activate
cd ~/claudinho
sudo systemctl stop claudinho          # free the mic
python3 src/main.py --no-wake --debug  # test interactively
sudo systemctl start claudinho         # back to service mode

# Update code
cd ~/claudinho && git pull
sudo systemctl restart claudinho
```

## How It Works

1. **Wake word** — openWakeWord listens continuously at 44100Hz (mic native rate), downsamples to 16kHz for inference. Detects the configured wake word with ONNX runtime.
2. **Recording** — After wake word triggers, records speech at 44100Hz. Uses webrtcvad (neural voice activity detection) to detect when you stop talking — much more reliable than volume thresholds with noisy USB mics.
3. **Transcription** — Whisper.cpp `base` model with `-l auto` for automatic language detection (English and Portuguese).
4. **LLM** — Sends transcribed text to Claude via OpenClaw's `/v1/chat/completions` endpoint. OpenClaw provides session memory and tool access.
5. **TTS** — Inworld AI synthesizes the response with the Theodore voice. Falls back to local Piper TTS if the cloud API is unavailable.
6. **Playback** — `aplay` outputs to auto-detected USB speaker.

USB audio device card numbers can change across reboots — both mic and speaker are auto-detected at runtime by scanning for "USB" in ALSA device names.

## Roadmap

- [x] Wake word → STT → Claude → TTS pipeline
- [x] Auto language detection (EN/PT)
- [x] OpenClaw gateway integration
- [x] systemd service (auto-start)
- [x] webrtcvad silence detection
- [x] USB device auto-detection
- [x] Inworld cloud TTS
- [ ] Custom "Claudinho" wake word
- [ ] Home automation integration
- [ ] LED/display status feedback
- [ ] 3D-printable enclosure

## License

MIT

---

*Built by Claudinho 🛠️ — the AI assistant who designed and coded itself.*
