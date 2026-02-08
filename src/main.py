#!/usr/bin/env python3
"""
Claudinho - Voice Assistant
===========================

Main entry point for the Claudinho voice assistant.
Orchestrates wake word detection, STT, LLM, and TTS.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

from wake_word import WakeWordDetector
from stt import SpeechToText
from assistant import Assistant
from tts import TextToSpeech
from audio import AudioPlayer, AudioRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Claudinho:
    """Main voice assistant class."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.running = False
        
        # Initialize components (lazy loaded)
        self._wake_word = None
        self._stt = None
        self._assistant = None
        self._tts = None
        self._recorder = None
        self._player = None
    
    @property
    def wake_word(self) -> WakeWordDetector:
        if self._wake_word is None:
            self._wake_word = WakeWordDetector()
        return self._wake_word
    
    @property
    def stt(self) -> SpeechToText:
        if self._stt is None:
            self._stt = SpeechToText()
        return self._stt
    
    @property
    def assistant(self) -> Assistant:
        if self._assistant is None:
            self._assistant = Assistant()
        return self._assistant
    
    @property
    def tts(self) -> TextToSpeech:
        if self._tts is None:
            self._tts = TextToSpeech()
        return self._tts
    
    @property
    def recorder(self) -> AudioRecorder:
        if self._recorder is None:
            self._recorder = AudioRecorder()
        return self._recorder
    
    @property
    def player(self) -> AudioPlayer:
        if self._player is None:
            self._player = AudioPlayer()
        return self._player
    
    def handle_conversation(self):
        """Handle a single conversation turn."""
        logger.info("🎤 Listening...")
        
        # Record user speech
        audio_data = self.recorder.record_until_silence()
        
        # Convert speech to text
        logger.info("🔄 Transcribing...")
        text = self.stt.transcribe(audio_data)
        logger.info(f"📝 User said: {text}")
        
        if not text.strip():
            logger.info("No speech detected")
            return
        
        # Get response from Claude via OpenClaw
        logger.info("🤔 Thinking...")
        response = self.assistant.chat(text)
        logger.info(f"💬 Response: {response}")
        
        # Convert response to speech
        logger.info("🔊 Speaking...")
        audio = self.tts.synthesize(response)
        self.player.play(audio)
    
    def run(self):
        """Main loop - listen for wake word and handle conversations."""
        self.running = True
        logger.info("🐱 Claudinho is starting up...")
        logger.info("Say 'Hey Claudinho' to start a conversation")
        
        try:
            while self.running:
                # Wait for wake word
                if self.wake_word.listen():
                    logger.info("👋 Wake word detected!")
                    self.player.play_sound("listening")
                    self.handle_conversation()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self._wake_word:
            self._wake_word.cleanup()
        if self._recorder:
            self._recorder.cleanup()
        logger.info("👋 Claudinho stopped")


def main():
    parser = argparse.ArgumentParser(description="Claudinho Voice Assistant")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle graceful shutdown
    claudinho = Claudinho(config_path=args.config)
    signal.signal(signal.SIGINT, lambda s, f: setattr(claudinho, 'running', False))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(claudinho, 'running', False))
    
    claudinho.run()


if __name__ == "__main__":
    main()
