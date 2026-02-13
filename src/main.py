#!/usr/bin/env python3
"""
Claudinho - DIY Voice Assistant
================================

Wake word → Record → Whisper STT → Claude → Piper TTS → Speaker

Hardware: Raspberry Pi 5, USB mic, USB speaker
"""

import argparse
import logging
import signal
import sys

import config
import audio
import stt
import tts
from assistant import Assistant
from wake_word import WakeWordDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup():
    """Create temp dirs and verify components exist."""
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check whisper
    if not config.WHISPER_CLI.exists():
        logger.error(f"Whisper CLI not found: {config.WHISPER_CLI}")
        logger.error("Build it: cd ~/whisper.cpp && make -j4")
        sys.exit(1)
    
    if not config.WHISPER_MODEL.exists():
        logger.error(f"Whisper model not found: {config.WHISPER_MODEL}")
        logger.error("Download: cd ~/whisper.cpp && ./models/download-ggml-model.sh base")
        sys.exit(1)
    
    # Check piper
    if not config.PIPER_BIN.exists():
        logger.error(f"Piper not found: {config.PIPER_BIN}")
        sys.exit(1)
    
    # Check at least one voice
    has_voice = any(p.exists() for p in config.PIPER_VOICES.values())
    if not has_voice:
        logger.error("No Piper voice models found!")
        sys.exit(1)
    
    logger.info("✅ All components verified")


def conversation_turn(assistant: Assistant):
    """Handle one conversation turn: record → transcribe → think → speak."""
    
    # Record user speech (stops on silence)
    wav_path = str(config.TMP_DIR / "input.wav")
    result = audio.record_until_silence(wav_path)
    
    if not result:
        logger.info("No speech detected, going back to listening")
        return
    
    # Transcribe with language detection
    text, language = stt.transcribe(wav_path)
    
    if not text.strip():
        logger.info("Empty transcription, going back to listening")
        return
    
    logger.info(f"📝 [{language}] User: {text}")
    
    # Get response from Claude
    logger.info("🤔 Thinking...")
    response = assistant.chat(text)
    logger.info(f"💬 Claudinho: {response}")
    
    # Synthesize and play response
    response_wav = tts.synthesize(response, language=language)
    if response_wav:
        audio.play(response_wav)


def run_assistant():
    """Main loop: wake word → conversation → repeat."""
    setup()
    
    logger.info("🐱 Claudinho starting up...")
    
    # Initialize components
    assistant = Assistant()
    detector = WakeWordDetector()
    
    # Play startup sound
    audio.play_beep()
    
    logger.info(f"👂 Listening for wake word (say '{config.WAKE_WORD_MODEL}')...")
    logger.info("Press Ctrl+C to stop\n")
    
    running = True
    
    def stop(sig, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    
    try:
        while running:
            if detector.listen():
                logger.info("👋 Wake word detected!")
                detector.pause()  # release mic for arecord
                audio.play_beep()
                conversation_turn(assistant)
                detector.resume()  # reopen mic for wake word
                logger.info(f"👂 Listening for wake word...\n")
    finally:
        detector.cleanup()
        logger.info("👋 Claudinho stopped")


def run_no_wake():
    """Run without wake word — just press Enter to talk."""
    setup()
    
    logger.info("🐱 Claudinho (no wake word mode)")
    logger.info("Press Enter to start recording, Ctrl+C to quit\n")
    
    assistant = Assistant()
    
    try:
        while True:
            input("⏎  Press Enter to speak...")
            audio.play_beep()
            conversation_turn(assistant)
            print()
    except (KeyboardInterrupt, EOFError):
        logger.info("👋 Claudinho stopped")


def main():
    parser = argparse.ArgumentParser(description="Claudinho Voice Assistant")
    parser.add_argument(
        "--no-wake", action="store_true",
        help="Skip wake word — press Enter to talk"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.no_wake:
        run_no_wake()
    else:
        run_assistant()


if __name__ == "__main__":
    main()
