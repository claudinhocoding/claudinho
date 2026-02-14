"""
Assistant
=========

Talks to Claude via OpenClaw's OpenAI-compatible HTTP API.
Supports both blocking and streaming responses.
"""

import json
import logging
import re
from typing import Generator

import requests

from config import OPENCLAW_URL, OPENCLAW_TOKEN

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Claudinho, a voice assistant running on a Raspberry Pi 5. "
    "You hear the user through a microphone and speak back through a speaker. "
    "Keep responses short and conversational — this is spoken audio, not text chat. "
    "Avoid markdown, bullet points, URLs, or long lists. "
    "Be natural, helpful, and concise. 1-3 sentences is ideal."
)


class Assistant:
    """Chat with Claude through the local OpenClaw gateway."""

    def __init__(self):
        self.session_user = "claudinho-voice"

        if not OPENCLAW_TOKEN:
            raise ValueError(
                "No OpenClaw gateway token configured.\n"
                "Set OPENCLAW_TOKEN in config.py"
            )

        logger.info("Connected to OpenClaw gateway")

    # ── Streaming chat ────────────────────────────────────────

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        """
        Stream a response from Claude, yielding text chunks as they arrive.
        Uses Server-Sent Events (SSE) via the OpenAI streaming API.
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
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                },
                timeout=60,
                stream=True,
            )

            if response.status_code != 200:
                logger.error(f"OpenClaw stream error: {response.status_code}")
                yield "Sorry, I'm having trouble thinking right now."
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to OpenClaw gateway.")
            yield "Sorry, my brain isn't responding right now."
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield "Sorry, something went wrong."

    def chat_stream_sentences(self, text: str) -> Generator[str, None, None]:
        """
        Stream Claude's response and yield complete sentences.
        Buffers text until a sentence boundary is found.
        """
        buffer = ""

        for chunk in self.chat_stream(text):
            buffer += chunk

            # Try to extract complete sentences
            while True:
                # Look for sentence-ending punctuation followed by space or end
                match = re.search(r'([.!?])\s', buffer)
                if match:
                    end = match.end()
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]
                    if sentence:
                        yield sentence
                else:
                    break

        # Yield any remaining text
        if buffer.strip():
            yield buffer.strip()

    # ── Blocking chat (fallback) ──────────────────────────────

    def chat(self, text: str) -> str:
        """Blocking chat — waits for full response."""
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
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenClaw API error: {response.status_code}")
                return "Sorry, I'm having trouble thinking right now."

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to OpenClaw gateway.")
            return "Sorry, my brain isn't responding right now."
        except Exception as e:
            logger.error(f"Assistant error: {e}")
            return "Sorry, something went wrong."

    def reset(self):
        """Reset session."""
        self.session_user = f"claudinho-voice-{id(self)}"
        logger.info("Conversation reset")
