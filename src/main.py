#!/usr/bin/env python3
"""
Claudinho - DIY Voice Assistant
================================

Wake word → Record → Groq STT → Claude (streaming) → Inworld TTS → Speaker
With smart home control via TP-Link Kasa devices.
"""

import argparse
import logging
import re
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

# Global smart home + music instances
smart_home = None
music_player = None


def setup():
    """Create temp dirs, verify components, discover smart home devices."""
    global smart_home
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Whisper.cpp is optional (Groq cloud STT is primary)
    if not config.WHISPER_CLI.exists():
        logger.warning("Whisper CLI not found (Groq cloud STT will be used)")
    if not config.WHISPER_MODEL.exists():
        logger.warning("Whisper model not found (Groq cloud STT will be used)")

    # Piper is optional (Inworld cloud TTS is primary)
    if not config.PIPER_BIN.exists():
        logger.warning("Piper not found (Inworld cloud TTS will be used)")

    # Discover smart home devices
    try:
        from lights import SmartHome
        smart_home = SmartHome()
        smart_home.discover()
    except ImportError:
        logger.info("python-kasa not installed — smart home disabled")
    except Exception as e:
        logger.warning(f"Smart home discovery failed: {e}")

    # Initialize Spotify
    try:
        from music import MusicPlayer
        music_player = MusicPlayer()
    except ImportError:
        logger.info("spotipy not installed — music disabled")
    except Exception as e:
        logger.warning(f"Spotify init failed: {e}")

    logger.info("✅ Components verified")


def extract_actions(text: str):
    """
    Extract <<action:device:param>> tags from text.
    Returns (clean_text, list_of_actions).
    """
    actions = re.findall(r'<<(.+?)>>', text)
    clean_text = re.sub(r'<<.+?>>', '', text).strip()
    # Clean up extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text, actions


def execute_actions(actions: list):
    """Execute smart home and music actions."""
    if not actions:
        return
    for action in actions:
        try:
            cmd = action.split(":")[0].strip().lower()
            if cmd.startswith("spotify_") and music_player:
                result = music_player.execute_action(action)
                logger.info(f"🎵 Action: {action} → {result}")
            elif smart_home:
                result = smart_home.execute_action(action)
                logger.info(f"🏠 Action: {action} → {result}")
            else:
                logger.warning(f"No handler for action: {action}")
        except Exception as e:
            logger.error(f"Action failed: {action} → {e}")


def conversation_turn(assistant: Assistant):
    """
    Handle one conversation turn with streaming + smart home actions.

    Pipeline:
    1. Record until silence (webrtcvad)
    2. Transcribe (Groq cloud, <1s)
    3. Stream Claude's response sentence by sentence
    4. Extract & execute smart home actions
    5. TTS + play each sentence (without action tags)
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

        # Extract smart home actions from sentence
        clean_sentence, actions = extract_actions(sentence)

        # Execute any actions immediately
        execute_actions(actions)

        # Skip TTS if the sentence was only an action tag
        if not clean_sentence:
            continue

        if first_audio:
            elapsed = time.monotonic() - t_start
            logger.info(f"⚡ First sentence in {elapsed:.1f}s: {clean_sentence}")
            first_audio = False
        else:
            logger.info(f"   ➜ {clean_sentence}")

        # Synthesize and play
        wav = tts.synthesize(clean_sentence, language=language)
        if wav:
            audio.play(wav)

    logger.info(f"💬 Full response: {' '.join(full_response)}")


def run_assistant():
    """Main loop: wake word → conversation → repeat."""
    setup()

    logger.info("🐱 Claudinho starting up...")

    # Initialize assistant with device list + music
    device_list = smart_home.get_device_list() if smart_home else []
    music_status = music_player.get_status() if music_player and music_player.available else None
    assistant = Assistant(device_list=device_list, music_status=music_status)
    detector = WakeWordDetector()

    audio.play_beep()

    logger.info("👂 Listening for wake word...")
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
                logger.info("👂 Listening for wake word...\n")
    finally:
        detector.cleanup()
        logger.info("👋 Claudinho stopped")


def run_no_wake():
    """Run without wake word — just press Enter to talk."""
    setup()

    logger.info("🐱 Claudinho (no wake word mode)")
    logger.info("Press Enter to start recording, Ctrl+C to quit\n")

    device_list = smart_home.get_device_list() if smart_home else []
    music_status = music_player.get_status() if music_player and music_player.available else None
    assistant = Assistant(device_list=device_list, music_status=music_status)

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
