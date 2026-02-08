# Claudinho 🎤🐱

A DIY voice assistant powered by [OpenClaw](https://github.com/openclaw/openclaw) + Claude, running on Raspberry Pi 5.

**Local wake word. Local STT. Local TTS. Cloud intelligence.**

## Features

- 🎯 **Wake word detection** — "Hey Claudinho" (Porcupine)
- 🎤 **Local STT** — Whisper.cpp running on-device
- 🧠 **Cloud LLM** — Claude via OpenClaw
- 🐱 **Local TTS** — KittenTTS (15M params, runs on CPU)
- 🔒 **Privacy-first** — Audio processed locally, only text goes to cloud

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ USB Mic  │───▶│ Porcupine│───▶│ Whisper  │             │
│   │          │    │ Wake Word│    │ STT      │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                         │                   │
│                                         ▼                   │
│                                   ┌──────────┐             │
│                                   │ OpenClaw │◀────────┐   │
│                                   └──────────┘         │   │
│                                         │         Claude   │
│                                         ▼          API     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐     │       │
│   │ Speaker  │◀───│ KittenTTS│◀───│ Response │─────┘       │
│   │          │    │    🐱    │    │          │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Model | Price |
|-----------|-------|-------|
| Computer | Raspberry Pi 5 (8GB) | ~$135 |
| Microphone | Adafruit Mini USB Mic | ~$6 |
| Speaker | Adafruit Mini USB Speaker | ~$13 |
| Power | USB-C 27W PSU | ~$14 |
| Storage | 32GB microSD card | ~$10 |
| **Total** | | **~$178** |

## Software Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| OS | Raspberry Pi OS (64-bit) | Base system |
| Wake Word | [Porcupine](https://picovoice.ai/platform/porcupine/) | "Hey Claudinho" detection |
| STT | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Speech to text |
| LLM | [OpenClaw](https://github.com/openclaw/openclaw) + Claude | Intelligence |
| TTS | [KittenTTS](https://github.com/KittenML/KittenTTS) | Text to speech |

## Project Structure

```
claudinho/
├── src/
│   ├── main.py              # Main entry point
│   ├── wake_word.py         # Porcupine wake word detection
│   ├── stt.py               # Whisper.cpp integration
│   ├── assistant.py         # OpenClaw/Claude integration
│   ├── tts.py               # KittenTTS integration
│   └── audio.py             # Audio I/O utilities
├── config/
│   └── config.yaml          # Configuration
├── scripts/
│   ├── install.sh           # Installation script
│   └── setup_pi.sh          # Pi setup script
├── requirements.txt
└── README.md
```

## Setup

### 1. Flash Raspberry Pi OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/):
- Choose Raspberry Pi OS (64-bit)
- Configure WiFi and enable SSH in settings
- Flash to SD card

### 2. SSH into Pi

```bash
ssh pi@claudinho.local
```

### 3. Clone and install

```bash
git clone https://github.com/claudinhocoding/claudinho.git
cd claudinho
./scripts/install.sh
```

### 4. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit with your settings
```

### 5. Run

```bash
python src/main.py
```

## Configuration

```yaml
# config/config.yaml
wake_word:
  keyword: "hey claudinho"  # or use built-in: "jarvis", "computer"
  sensitivity: 0.5

stt:
  model: "base"  # tiny, base, small, medium
  language: "en"

tts:
  voice: "expr-voice-2-f"  # KittenTTS voice

openclaw:
  # Uses existing OpenClaw installation
  gateway_url: "http://localhost:18789"
```

## Roadmap

- [x] Project setup and architecture
- [ ] Basic wake word → STT → Claude → TTS pipeline
- [ ] Audio I/O handling
- [ ] Conversation context/memory
- [ ] LED/display status feedback
- [ ] Home Assistant integration
- [ ] Custom wake word training
- [ ] 3D-printable enclosure

## Development

```bash
# Run in development mode
python src/main.py --debug

# Test individual components
python -m pytest tests/
```

## License

MIT

---

*Named after Claudinho — the AI assistant who helped design it.* 🛠️
