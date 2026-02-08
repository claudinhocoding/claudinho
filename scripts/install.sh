#!/bin/bash
# Claudinho Installation Script
# =============================
# Run this on your Raspberry Pi 5

set -e

echo "🐱 Installing Claudinho..."
echo ""

# Check if running on Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "   Some components may not work correctly."
    echo ""
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    git \
    cmake \
    build-essential

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install KittenTTS
echo "🐱 Installing KittenTTS..."
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl

# Install whisper.cpp
echo "🎤 Installing whisper.cpp..."
cd ~
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
make clean
make -j4

# Download Whisper model
echo "📥 Downloading Whisper base model..."
./models/download-ggml-model.sh base

cd -

# Create config from example
if [ ! -f "config/config.yaml" ]; then
    echo "📝 Creating config file..."
    cp config/config.example.yaml config/config.yaml
    echo "   Edit config/config.yaml with your settings"
fi

# Create sounds directory
mkdir -p sounds

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Get a free Porcupine access key: https://picovoice.ai/"
echo "  2. Edit config/config.yaml with your settings"
echo "  3. Run: source venv/bin/activate && python src/main.py"
echo ""
echo "🐱 Claudinho is ready!"
