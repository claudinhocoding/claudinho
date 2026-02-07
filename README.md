# Claudinho 🎤🤖

A DIY voice assistant powered by [OpenClaw](https://github.com/openclaw/openclaw) + Claude, running on Raspberry Pi 5.

**Local wake word. Local speech-to-text. Cloud intelligence.**

## Features

- 🎯 **Wake word detection** — "Hey Claudinho" (Porcupine)
- 🎤 **Local STT** — Whisper.cpp running on-device
- 🧠 **Cloud LLM** — Claude via OpenClaw
- 🔊 **TTS** — Piper (local) or ElevenLabs (cloud)
- 🔒 **Privacy-first** — Audio stays local, only text goes to the cloud

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ ReSpeaker│───▶│ Porcupine│───▶│ Whisper  │             │
│   │ 2-Mic    │    │ Wake Word│    │ STT      │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                         │                   │
│                                         ▼                   │
│                                   ┌──────────┐             │
│                                   │ OpenClaw │◀────────┐   │
│                                   └──────────┘         │   │
│                                         │         Claude   │
│                                         ▼          API     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐     │       │
│   │ Speaker  │◀───│ Piper    │◀───│ Response │─────┘       │
│   │          │    │ TTS      │    │          │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Model | Price |
|-----------|-------|-------|
| Computer | Raspberry Pi 5 (8GB) | ~$135 |
| Microphone | ReSpeaker 2-Mic HAT | ~$15 |
| Speaker | JBL Go 3 (or similar) | ~$40 |
| Power | USB-C 27W supply | ~$15 |
| Storage | 32GB SD card | ~$12 |
| **Total** | | **~$215** |

## Software Stack

- **OS**: Raspberry Pi OS (64-bit)
- **Wake Word**: [Porcupine](https://picovoice.ai/platform/porcupine/) (Picovoice)
- **STT**: [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- **Assistant**: [OpenClaw](https://github.com/openclaw/openclaw) + Claude
- **TTS**: [Piper](https://github.com/rhasspy/piper)

## Setup

> 🚧 Coming soon — project in early development

### 1. Flash Raspberry Pi OS

### 2. Install dependencies

### 3. Configure OpenClaw

### 4. Run Claudinho

## Roadmap

- [ ] Basic wake word → STT → Claude → TTS pipeline
- [ ] ReSpeaker LED feedback (listening/thinking/speaking states)
- [ ] Conversation context/memory
- [ ] Home Assistant integration
- [ ] Custom wake word training
- [ ] 3D-printable enclosure

## License

MIT

---

*Named after Claudinho — the AI assistant who helped design it.* 🛠️
