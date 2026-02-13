"""
Assistant
=========

Talks to Claude via OpenClaw's OpenAI-compatible HTTP API.
This gives us memory, tools, and conversation persistence.
"""

import logging

import requests

from config import OPENCLAW_URL, OPENCLAW_TOKEN

logger = logging.getLogger(__name__)


class Assistant:
    """Chat with Claude through the local OpenClaw gateway."""
    
    def __init__(self):
        self.session_user = "claudinho-voice"  # stable session key
        
        if not OPENCLAW_TOKEN:
            raise ValueError(
                "No OpenClaw gateway token configured.\n"
                "Set OPENCLAW_TOKEN in config.py"
            )
        
        logger.info("Connected to OpenClaw gateway")
    
    def chat(self, text: str) -> str:
        """
        Send a message through OpenClaw and get a response.
        
        Uses the OpenAI-compatible chat completions endpoint.
        The 'user' field gives us a persistent session.
        """
        try:
            response = requests.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                    "Content-Type": "application/json",
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
