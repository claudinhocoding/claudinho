# Claudinho 🎤🤖

A DIY voice assistant powered by [OpenClaw](https://github.com/openclaw/openclaw) + Claude, running on Raspberry Pi 5.

**Wake word → Cloud STT → Streaming LLM → Sentence-level TTS → Speaker**

Say the wake word → speak → get a spoken response from Claude in ~3-4 seconds. Controls your smart home lights and plays Spotify music — all by voice.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Raspberry Pi 5                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌────────────┐              │
│  │ USB Mic  │───▶│openWakeWord│───▶│  webrtcvad │              │
│  │ (44.1kHz)│    │  Wake Word │    │    (VAD)   │              │
│  └──────────┘    └────────────┘    └────────────┘              │
│                                          │                      │
│                                    stop recording               │
│                                          │                      │
│                                          ▼                      │
│                                    ┌────────────┐              │
│                               ┌───▶│ Groq STT   │ (<1s)       │
│                               │    │ Whisper API │              │
│                               │    └────────────┘              │
│                               │          │                      │
│                               │          ▼                      │
│                               │    ┌────────────┐              │
│                               │    │  OpenClaw  │◀── Claude    │
│                               │    │  Gateway   │   (streaming)│
│                               │    └────────────┘              │
│                               │          │                      │
│                               │    sentence-by-sentence         │
│                               │          │                      │
│                               │    ┌─────┴──────┐              │
│                               │    │            │              │
│  ┌──────────┐    ┌─────────┐  │  ┌──────┐  ┌────────┐         │
│  │ Speaker  │◀───│ Inworld │◀─┘  │ Kasa │  │Spotify │         │
│  │ (USB)    │    │  TTS    │     │Lights│  │  API   │         │
│  └──────────┘    └─────────┘     └──────┘  └────────┘         │
│                                                │                │
│                                          ┌─────┴───┐           │
│                                          │spotifyd │           │
│                                          │(Connect)│           │
│                                          └─────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- 🎙️ **Wake word detection** — openWakeWord with custom ONNX models
- ⚡ **~3-4s latency** — Groq cloud STT + streaming LLM + sentence-level TTS
- 🏠 **Smart home control** — TP-Link Kasa lights (on/off/brightness/toggle)
- 🎵 **Spotify music** — Play, pause, skip, queue via voice through Spotify Connect
- 🌍 **Multilingual** — Auto-detects English and Portuguese
- 🔄 **Always on** — systemd service with auto-start on boot
- 🔌 **Auto-detection** — USB mic/speaker detected at runtime (survives card number changes)

## Hardware

| Component | Model | Price |
|-----------|-------|-------|
| Computer | Raspberry Pi 5 (8GB) | ~$135 |
| Microphone | [Adafruit Mini USB Mic #3367](https://www.adafruit.com/product/3367) | ~$6 |
| Speaker | [Adafruit Mini USB Speaker #3369](https://www.adafruit.com/product/3369) | ~$13 |
| Power | Official Raspberry Pi 27W USB-C PSU | ~$14 |
| Case | Official Raspberry Pi 5 Case + Fan | ~$12 |
| Storage | 128GB microSD card | ~$25 |
| **Total** | | **~$200** |

## Software Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| OS | Raspberry Pi OS 64-bit Lite (Debian Trixie) | Base system |
| Wake Word | [openWakeWord](https://github.com/dscripka/openWakeWord) | Local wake word detection (ONNX) |
| VAD | [webrtcvad](https://github.com/wiseman/py-webrtcvad) / [silero-vad](https://github.com/snakers4/silero-vad) | Voice activity detection for silence detection |
| STT (primary) | [Groq Whisper API](https://groq.com/) | Cloud STT — whisper-large-v3-turbo, <1s (~$0.04/hr) |
| STT (fallback) | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Local STT — base model, ~3.7s for 3s audio |
| LLM | [OpenClaw](https://github.com/openclaw/openclaw) + Claude | Streaming intelligence via SSE |
| TTS (primary) | [Inworld AI](https://inworld.ai/) | Cloud TTS — Theodore voice |
| TTS (fallback) | [Piper](https://github.com/rhasspy/piper) | Local TTS — EN (Norman) + PT-BR (Edresson) |
| Smart Home | [python-kasa](https://github.com/python-kasa/python-kasa) | TP-Link Kasa device control |
| Music | [spotipy](https://github.com/spotipy-dev/spotipy) + [spotifyd](https://github.com/Spotifyd/spotifyd) | Spotify Web API + Spotify Connect daemon |
| Audio | PyAudio + ALSA | Mic input / speaker output |

## Project Structure

```
claudinho/
├── src/
│   ├── main.py           # Entry point — wake word loop, action extraction, streaming pipeline
│   ├── config.py          # Hardware paths, API keys, thresholds
│   ├── wake_word.py       # openWakeWord detection (44.1kHz → 16kHz resample)
│   ├── audio.py           # Record with VAD silence detection, play via aplay
│   ├── vad.py             # Multi-backend VAD (silero → webrtcvad → RMS fallback)
│   ├── stt.py             # Groq cloud STT with local Whisper.cpp fallback
│   ├── tts.py             # Inworld cloud TTS with Piper local fallback
│   ├── assistant.py       # OpenClaw streaming chat (pinned to main session)
│   ├── lights.py          # TP-Link Kasa smart home control
│   └── music.py           # Spotify playback via spotipy + spotifyd
├── scripts/
│   ├── install.sh         # Full Pi setup script
│   ├── claudinho.service  # systemd unit (auto-start on boot)
│   ├── spotifyd.conf      # Spotify Connect daemon config
│   ├── spotifyd.service   # spotifyd systemd unit
│   ├── spotify_auth.py    # One-time Spotify OAuth authorization
│   └── record_samples.py  # Record wake word training samples
├── blog/                  # Blog post about building Claudinho
├── training/              # Wake word training data
├── notebooks/             # Colab notebooks for model training
├── models/                # Custom wake word ONNX models
├── .env                   # API keys (GROQ_API_KEY, SPOTIFY_*, KASA_*) — gitignored
├── requirements.txt
└── README.md
```

## Setup

### 1. Flash Raspberry Pi OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/):
- Choose **Raspberry Pi OS (64-bit) Lite**
- Set hostname to `claudinho`, enable SSH, configure WiFi
- Flash to microSD card and boot

### 2. Install dependencies

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
pip install python-kasa spotipy

# Build Whisper.cpp (local STT fallback)
cd ~
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j4
./models/download-ggml-model.sh base

# Install OpenClaw
npm install -g openclaw
openclaw configure
```

### 3. Configure environment

Create a `.env` file in the project root:

```bash
# Groq API (cloud STT)
GROQ_API_KEY=your-groq-api-key

# Spotify (optional)
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback

# TP-Link Kasa (optional)
KASA_USERNAME=your-tplink-email
KASA_PASSWORD=your-tplink-password
```

Edit `src/config.py` with your OpenClaw gateway token:

```python
OPENCLAW_URL = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "your-gateway-token-here"
```

### 4. Spotify setup (optional)

```bash
# Install spotifyd (Spotify Connect daemon)
wget https://github.com/Spotifyd/spotifyd/releases/latest/download/spotifyd-linux-aarch64-slim.tar.gz
tar xzf spotifyd-linux-aarch64-slim.tar.gz
sudo mv spotifyd /usr/local/bin/
sudo chmod +x /usr/local/bin/spotifyd

# Install spotifyd service
sudo cp scripts/spotifyd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable spotifyd
sudo systemctl start spotifyd

# Authorize Spotify (one-time — follow the URL prompt)
source venv/bin/activate
set -a; source .env; set +a
python3 scripts/spotify_auth.py
```

> **Note:** spotifyd requires `libssl1.1` on Debian Trixie. Install from the Bullseye security repo:
> ```bash
> wget http://security.debian.org/debian-security/pool/updates/main/o/openssl/libssl1.1_1.1.1w-0+deb11u4_arm64.deb
> sudo dpkg -i libssl1.1_1.1.1w-0+deb11u4_arm64.deb
> ```

### 5. Test it

```bash
source ~/claudinho/venv/bin/activate
cd ~/claudinho

# Quick test (press Enter to talk, no wake word needed)
python3 src/main.py --no-wake

# Full mode with wake word
python3 src/main.py

# Debug mode
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
```

## Voice Commands

### Smart Home
- "Turn on the lights" / "Turn off the sofa light"
- "Set the entrance light to 50%"
- "Toggle all lights"

### Music
- "Play Bad Bunny" / "Play some chill jazz"
- "Pause the music" / "Resume"
- "Skip this song" / "Previous track"
- "Set the volume to 70"
- "Queue Bohemian Rhapsody"

### Combined
- "Set the mood — dim the lights and play some bossa nova"

## How It Works

### Latency Pipeline (~3-4s to first audio)

| Stage | Time | What happens |
|-------|------|-------------|
| Silence detection | ~0.9s | webrtcvad detects end of speech |
| Groq STT | <1s | Cloud Whisper large-v3-turbo transcription |
| Claude first sentence | ~2s | Streaming SSE via OpenClaw gateway |
| Inworld TTS | ~1s | First sentence synthesized |
| **Total** | **~3-4s** | From end of speech to first audio |

### Action Tags

Claude includes inline action tags in responses that are extracted and executed before TTS:

```
"Playing Bad Bunny for you. <<spotify_play:Bad Bunny>>"
→ TTS speaks: "Playing Bad Bunny for you."
→ Executes: spotify_play("Bad Bunny")
```

Tags are invisible to the user — they only hear the natural speech.

### Session Pinning

Voice interactions are pinned to the main OpenClaw agent session (`agent:main:main`) via the `x-openclaw-session-key` header. This means the voice assistant shares memory and context with other OpenClaw interactions — one agent, one brain.

## Tuning

### VAD (Voice Activity Detection)

Set in `src/vad.py` → `WebRTCVADBackend(aggressiveness=3)`:

| Level | Behavior |
|-------|----------|
| 0 | Least aggressive — keeps recording through pauses |
| 1-2 | Moderate filtering |
| **3** | **Most aggressive** — stops quickly after speech (default) |

### Recording (in `src/config.py`)

| Setting | Default | What it does |
|---------|---------|-------------|
| `SILENCE_DURATION` | `0.8` | Seconds of silence to stop recording |
| `MAX_RECORD_DURATION` | `30` | Max recording length |
| `MIN_SPEECH_DURATION` | `0.3` | Min speech before silence detection |
| `VAD_THRESHOLD` | `0.4` | Speech probability threshold (Silero VAD) |

### VAD Backend Priority

1. **silero-vad** — Neural VAD (most accurate, `pip install silero-vad`)
2. **webrtcvad** — Google's C-based VAD (fast, reliable) ← recommended for low-SNR USB mics
3. **RMS threshold** — Simple volume-based (last resort)

## Roadmap

- [x] Wake word → STT → Claude → TTS pipeline
- [x] Auto language detection (EN/PT)
- [x] OpenClaw gateway integration (streaming)
- [x] systemd service (auto-start)
- [x] webrtcvad silence detection
- [x] USB device auto-detection
- [x] Groq cloud STT (<1s)
- [x] Streaming LLM + sentence-level TTS
- [x] Inworld cloud TTS (Theodore voice)
- [x] Smart home control (TP-Link Kasa lights)
- [x] Spotify music playback (spotifyd + spotipy)
- [x] Session pinning (shared memory with OpenClaw)
- [ ] Custom "Claudinho" wake word (training in progress)
- [ ] LED/display status feedback
- [ ] 3D-printable enclosure

## License

MIT

---

*Built by Claudinho 🛠️ — the AI assistant who designed and coded itself.*
