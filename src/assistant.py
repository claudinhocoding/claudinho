"""
Assistant
=========

Talks to Claude via OpenClaw's OpenAI-compatible HTTP API.
This gives us memory, tools, and conversation persistence.
"""

import logging
import json
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GATEWAY_URL = "http://127.0.0.1:18789"


class Assistant:
    """Chat with Claude through the local OpenClaw gateway."""
    
    def __init__(self):
        self.token = self._read_token()
        self.session_user = "claudinho-voice"  # stable session key
        
        if not self.token:
            raise ValueError(
                "No OpenClaw gateway token found.\n"
                "Check ~/.openclaw/openclaw.json for gateway.auth.token"
            )
        
        logger.info("Connected to OpenClaw gateway")
    
    def _read_token(self) -> str:
        """Read gateway token from OpenClaw config."""
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            token = data.get("gateway", {}).get("auth", {}).get("token", "")
            if token and token != "__OPENCLAW_REDACTED__":
                return token
        return ""
    
    def chat(self, text: str) -> str:
        """
        Send a message through OpenClaw and get a response.
        
        Uses the OpenAI-compatible chat completions endpoint.
        The 'user' field gives us a persistent session.
        """
        try:
            response = requests.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "x-openclaw-agent-id": "main",
                },
                json={
                    "model": "openclaw",
                    "user": self.session_user,
                    "messages": [{"role": "user", "content": text}],
                },
                timeout=60,
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"]
                return reply
            else:
                logger.error(f"OpenClaw API error: {response.status_code} {response.text}")
                return "Sorry, I'm having trouble thinking right now."
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to OpenClaw gateway. Is it running?")
            return "Sorry, my brain isn't responding right now."
        except Exception as e:
            logger.error(f"Assistant error: {e}")
            return "Sorry, something went wrong."
    
    def reset(self):
        """Reset session (start fresh conversation)."""
        self.session_user = f"claudinho-voice-{id(self)}"
        logger.info("Conversation reset")
