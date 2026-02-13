# Claudinho 🎤🤖

A DIY voice assistant powered by [OpenClaw](https://github.com/openclaw/openclaw) + Claude, running on Raspberry Pi 5.

**Local wake word. Local STT. Local TTS. Cloud intelligence.**

Say "Hey Jarvis" → speak → get a spoken response from Claude. All audio processing happens on-device — only text goes to the cloud.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌───────────┐    ┌───────────┐           │
│   │ USB Mic  │───▶│openWakeWrd│───▶│ Whisper   │           │
│   │          │    │ "Hey      │    │ STT       │           │
│   └──────────┘    │  Jarvis"  │    │ (base)    │           │
│                   └───────────┘    └───────────┘           │
│                                          │                  │
│                                          ▼                  │
│                                    ┌───────────┐           │
│                                    │ OpenClaw  │◀──────┐   │
│                                    │ Gateway   │       │   │
│                                    └───────────┘   Claude  │
│                                          │          API    │
│   ┌──────────┐    ┌───────────┐    ┌───────────┐   │      │
│   │ Speaker  │◀───│ Piper TTS │◀───│ Response  │───┘      │
│   │          │    │ EN / PT   │    │           │           │
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
| Wake Word | [openWakeWord](https://github.com/dscripka/openWakeWord) | "Hey Jarvis" detection (ONNX) |
| STT | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Speech-to-text (base model, ~3.7s for 3s audio) |
| LLM | [OpenClaw](https://github.com/openclaw/openclaw) + Claude | Intelligence |
| TTS | [Piper](https://github.com/rhasspy/piper) | Text-to-speech (EN + PT-BR voices) |
| Audio | PyAudio + ALSA | Mic input / speaker output |

## Project Structure

```
claudinho/
├── src/
│   ├── main.py           # Entry point (wake word loop + no-wake mode)
│   ├── config.py          # Hardware paths, device settings, thresholds
│   ├── wake_word.py       # openWakeWord detection (44100→16kHz resample)
│   ├── audio.py           # Record with silence detection, play via aplay
│   ├── stt.py             # Whisper.cpp CLI with auto language detection
│   ├── tts.py             # Piper TTS with per-language voice selection
│   └── assistant.py       # OpenClaw gateway chat completions API
├── scripts/
│   ├── install.sh         # Full Pi setup script
│   └── claudinho.service  # systemd unit (auto-start on boot)
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
pip install pyaudio numpy scipy requests openwakeword --no-deps
pip install onnxruntime scikit-learn tqdm

# Build Whisper.cpp
cd ~
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j4
./models/download-ggml-model.sh base

# Install Piper TTS
cd ~
mkdir piper && cd piper
# Download the Piper binary for aarch64 from:
# https://github.com/rhasspy/piper/releases
# Then download voice models:
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx.json
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json

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
python src/main.py --no-wake

# Full mode with wake word
python src/main.py
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

# Follow logs
journalctl -u claudinho -f

# Restart after code changes
cd ~/claudinho && git pull
sudo systemctl restart claudinho

# Stop
sudo systemctl stop claudinho
```

## SSH Quick Reference

```bash
# Connect from your Mac/PC
ssh claudinho@claudinho.local

# Check voice assistant
sudo systemctl status claudinho
journalctl -u claudinho -f

# Check OpenClaw gateway
sudo systemctl status openclaw-gateway

# Activate Python env (only needed for manual runs)
source ~/claudinho/venv/bin/activate
```

## How It Works

1. **Wake word** — openWakeWord listens continuously at 44100Hz (mic native rate), downsamples to 16kHz for inference. Detects "Hey Jarvis" with ONNX runtime.
2. **Recording** — Auto-calibrates ambient noise (~0.8s), records speech, stops on silence (1.5s of quiet after speech detected). Saves as 16kHz WAV.
3. **Transcription** — Whisper.cpp `base` model with `-l auto` for automatic language detection (English and Portuguese).
4. **LLM** — Sends transcribed text to Claude via OpenClaw's `/v1/chat/completions` endpoint. OpenClaw provides session memory and tool access.
5. **TTS** — Piper synthesizes the response using language-matched voices (Norman for EN, Edresson for PT-BR).
6. **Playback** — `aplay` outputs to auto-detected USB speaker.

USB audio device card numbers can change across reboots — both mic and speaker are auto-detected at runtime by scanning for "USB" in ALSA device names.

## Roadmap

- [x] Wake word → STT → Claude → TTS pipeline
- [x] Auto language detection (EN/PT)
- [x] OpenClaw gateway integration
- [x] systemd service (auto-start)
- [x] Auto noise calibration
- [x] USB device auto-detection
- [ ] Custom "Hey Claudinho" wake word
- [ ] Home automation integration
- [ ] LED/display status feedback
- [ ] 3D-printable enclosure

## License

MIT

---

*Built by Claudinho 🛠️ — the AI assistant who designed and coded itself.*
