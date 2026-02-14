#!/usr/bin/env python3
"""
Claudinho - DIY Voice Assistant
================================

Wake word → Record → Groq STT → Claude (streaming) → Inworld TTS → Speaker

Pipeline streams Claude's response sentence-by-sentence:
first sentence plays while the rest is still generating.
"""

import argparse
import logging
import signal
import sys
import time

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

    # Whisper.cpp is optional (Groq cloud STT is primary)
    if not config.WHISPER_CLI.exists():
        logger.warning(f"Whisper CLI not found (Groq cloud STT will be used)")
    if not config.WHISPER_MODEL.exists():
        logger.warning(f"Whisper model not found (Groq cloud STT will be used)")

    # Piper is optional (Inworld cloud TTS is primary)
    if not config.PIPER_BIN.exists():
        logger.warning(f"Piper not found (Inworld cloud TTS will be used)")

    logger.info("✅ Components verified")


def conversation_turn(assistant: Assistant):
    """
    Handle one conversation turn with streaming.

    Pipeline:
    1. Record until silence (webrtcvad)
    2. Transcribe (Groq cloud, <1s)
    3. Stream Claude's response sentence by sentence
    4. TTS + play each sentence as it arrives
    """
    # Record user speech
    wav_path = str(config.TMP_DIR / "input.wav")
    result = audio.record_until_silence(wav_path)

    if not result:
        logger.info("No speech detected, going back to listening")
        return

    # Transcribe
    text, language = stt.transcribe(wav_path)

    if not text.strip():
        logger.info("Empty transcription, going back to listening")
        return

    logger.info(f"📝 [{language}] User: {text}")

    # Stream response sentence-by-sentence
    logger.info("🤔 Thinking...")
    t_start = time.monotonic()
    first_audio = True
    full_response = []

    for sentence in assistant.chat_stream_sentences(text):
        full_response.append(sentence)

        if first_audio:
            elapsed = time.monotonic() - t_start
            logger.info(f"⚡ First sentence in {elapsed:.1f}s: {sentence}")
            first_audio = False
        else:
            logger.info(f"   ➜ {sentence}")

        # Synthesize and play this sentence
        wav = tts.synthesize(sentence, language=language)
        if wav:
            audio.play(wav)

    logger.info(f"💬 Full response: {' '.join(full_response)}")


def run_assistant():
    """Main loop: wake word → conversation → repeat."""
    setup()

    logger.info("🐱 Claudinho starting up...")

    assistant = Assistant()
    detector = WakeWordDetector()

    audio.play_beep()

    logger.info(f"👂 Listening for wake word...")
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
                detector.pause()
                audio.play_beep()
                conversation_turn(assistant)
                detector.resume()
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
