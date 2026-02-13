#!/bin/bash
# ============================================
# Claudinho Voice Assistant - Pi 5 Setup
# ============================================
# Run this after first SSH into your Pi:
#   curl -sSL https://raw.githubusercontent.com/claudinhocoding/claudinho/main/scripts/install.sh | bash
#
# Hardware:
#   - Raspberry Pi 5 (8GB)
#   - Mini USB Microphone (Adafruit #3367)
#   - Mini USB Stereo Speaker (Adafruit #3369)
# ============================================

set -e

echo "🐱 Claudinho Voice Assistant - Installation"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ---- System updates ----
echo "📦 Step 1: System updates..."
sudo apt update && sudo apt upgrade -y
step "System updated"

# ---- System dependencies ----
echo ""
echo "📦 Step 2: Installing system dependencies..."
sudo apt install -y \
    python3 python3-pip python3-venv \
    portaudio19-dev \
    alsa-utils \
    git cmake build-essential \
    curl wget
step "System dependencies installed"

# ---- Node.js (for OpenClaw) ----
echo ""
echo "📦 Step 3: Installing Node.js..."
if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    step "Node.js already installed: $NODE_VER"
else
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt install -y nodejs
    step "Node.js $(node --version) installed"
fi

# ---- OpenClaw ----
echo ""
echo "📦 Step 4: Installing OpenClaw..."
if command -v openclaw &>/dev/null; then
    step "OpenClaw already installed"
else
    sudo npm install -g openclaw
    step "OpenClaw installed"
fi

# ---- Clone Claudinho repo ----
echo ""
echo "📦 Step 5: Setting up Claudinho..."
CLAUDINHO_DIR="$HOME/claudinho"
if [ -d "$CLAUDINHO_DIR" ]; then
    cd "$CLAUDINHO_DIR" && git pull
    step "Claudinho repo updated"
else
    git clone https://github.com/claudinhocoding/claudinho.git "$CLAUDINHO_DIR"
    step "Claudinho repo cloned"
fi

# ---- Python virtual environment ----
echo ""
echo "📦 Step 6: Setting up Python environment..."
cd "$CLAUDINHO_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
step "Python environment ready"

# ---- Whisper.cpp ----
echo ""
echo "📦 Step 7: Installing Whisper.cpp..."
WHISPER_DIR="$HOME/whisper.cpp"
if [ -d "$WHISPER_DIR" ]; then
    step "Whisper.cpp already installed"
else
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    cd "$WHISPER_DIR"
    make -j4
    # Download base model (~142MB) - good balance of speed/accuracy for Pi 5
    bash models/download-ggml-model.sh base
    step "Whisper.cpp installed with base model"
fi

# ---- Piper TTS ----
echo ""
echo "📦 Step 8: Installing Piper TTS..."
if command -v piper &>/dev/null; then
    step "Piper already installed"
else
    # piper-tts Python package includes the binary
    pip install piper-tts
    # Download a voice
    echo "Downloading voice model..."
    piper --model en_US-lessac-medium --download-dir "$HOME/.local/share/piper-voices" \
        <<< "test" > /dev/null 2>&1 || true
    step "Piper TTS installed"
fi

# ---- Audio test ----
echo ""
echo "🔊 Step 9: Testing audio..."
echo ""

# Check for USB audio devices
if arecord -l 2>/dev/null | grep -q "USB"; then
    step "USB microphone detected"
else
    warn "USB microphone not detected. Plug it in and re-run."
fi

if aplay -l 2>/dev/null | grep -q "USB"; then
    step "USB speaker detected"
else
    warn "USB speaker not detected. Plug it in and re-run."
fi

# ---- ALSA config for USB audio ----
echo ""
echo "🔧 Step 10: Configuring USB audio as default..."
cat > "$HOME/.asoundrc" << 'EOF'
# Use USB devices as default audio
# Adjust card numbers if needed (run: arecord -l / aplay -l)

pcm.!default {
    type asym
    playback.pcm "plug:usb_speaker"
    capture.pcm "plug:usb_mic"
}

pcm.usb_speaker {
    type hw
    card 1
    device 0
}

pcm.usb_mic {
    type hw
    card 1
    device 0
}

ctl.!default {
    type hw
    card 1
}
EOF
warn "Default audio set to USB devices (card 1). Run 'arecord -l' to verify card numbers."

# ---- OpenClaw setup reminder ----
echo ""
echo "============================================"
echo -e "${GREEN}🐱 Claudinho installation complete!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Set up OpenClaw:"
echo "     openclaw onboard"
echo ""
echo "  2. Test audio:"
echo "     arecord -d 3 test.wav && aplay test.wav"
echo ""
echo "  3. Test wake word:"
echo "     cd ~/claudinho && source venv/bin/activate"
echo "     python src/wake_word.py"
echo ""
echo "  4. Run Claudinho:"
echo "     cd ~/claudinho && source venv/bin/activate"
echo "     python src/main.py"
echo ""
echo "  5. (Optional) Auto-start on boot:"
echo "     sudo cp scripts/claudinho.service /etc/systemd/system/"
echo "     sudo systemctl enable claudinho"
echo "     sudo systemctl start claudinho"
echo ""
