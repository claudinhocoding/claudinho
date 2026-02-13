"""
Assistant
=========

Interfaces with OpenClaw running locally on the Pi.
Sends transcribed text, receives response text.
"""

import logging
import subprocess
import json
from typing import Optional

logger = logging.getLogger(__name__)


class Assistant:
    """Communicates with OpenClaw for LLM responses."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:18789",
        gateway_token: Optional[str] = None,
    ):
        """
        Initialize assistant.

        OpenClaw runs on the Pi as a gateway service.
        We send messages via the local API.

        Args:
            gateway_url: OpenClaw gateway URL (local).
            gateway_token: Gateway auth token.
        """
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token or self._read_token()
        self._conversation_history = []

    def _read_token(self) -> str:
        """Read gateway token from OpenClaw config."""
        import yaml
        from pathlib import Path

        config_path = Path.home() / ".openclaw" / "openclaw.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            token = data.get("gateway", {}).get("auth", {}).get("token", "")
            if token and token != "__OPENCLAW_REDACTED__":
                return token

        # Fallback: try environment variable
        import os
        return os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

    def chat(self, text: str) -> str:
        """
        Send text to OpenClaw and get a response.

        Uses the OpenClaw wake API to inject a message
        and get a response from the active agent.

        Args:
            text: User's transcribed speech.

        Returns:
            Assistant's response text.
        """
        try:
            import requests

            # Use the wake endpoint to send a voice message
            response = requests.post(
                f"{self.gateway_url}/api/wake",
                headers={
                    "Authorization": f"Bearer {self.gateway_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": f"[Voice] {text}",
                    "mode": "now",
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("reply", data.get("message", "I heard you but couldn't process that."))
            else:
                logger.error(f"OpenClaw API error: {response.status_code} {response.text}")
                return "Sorry, I'm having trouble connecting to my brain."

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to OpenClaw gateway. Is it running?")
            return "Sorry, my brain isn't responding right now."
        except Exception as e:
            logger.error(f"Assistant error: {e}")
            return "Sorry, something went wrong."

    def chat_cli(self, text: str) -> str:
        """
        Alternative: use openclaw CLI directly.
        Simpler but spawns a process each time.
        """
        try:
            result = subprocess.run(
                ["openclaw", "gateway", "wake", "--text", f"[Voice] {text}", "--mode", "now"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"CLI error: {result.stderr}")
                return "Sorry, something went wrong."

        except FileNotFoundError:
            logger.error("openclaw CLI not found")
            return "OpenClaw is not installed."
        except Exception as e:
            logger.error(f"CLI error: {e}")
            return "Sorry, something went wrong."


if __name__ == "__main__":
    assistant = Assistant()
    print("Testing assistant (type 'quit' to exit):")
    while True:
        text = input("\nYou: ")
        if text.lower() in ("quit", "exit", "q"):
            break
        response = assistant.chat(text)
        print(f"Claudinho: {response}")
