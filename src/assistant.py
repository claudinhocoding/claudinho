"""
Assistant
=========

Talks to Claude via OpenClaw's OpenAI-compatible HTTP API.
Supports streaming responses and smart home actions.
"""

import json
import logging
import re
from typing import Generator, List

import requests

from config import OPENCLAW_URL, OPENCLAW_TOKEN

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are Claudinho, a voice assistant running on a Raspberry Pi 5. "
    "You hear the user through a microphone and speak back through a speaker. "
    "Keep responses short and conversational — this is spoken audio, not text chat. "
    "Avoid markdown, bullet points, URLs, or long lists. "
    "Be natural, helpful, and concise. 1-3 sentences is ideal."
)

SMART_HOME_PROMPT = """

You can control smart home devices. Available devices:
{devices}

To control a device, include an action tag in your response:
<<turn_on:Device Name>>
<<turn_off:Device Name>>
<<brightness:Device Name:50>>
<<toggle:Device Name>>

Always include natural speech along with the action. The action tags will be removed before speaking.
Example: "Sure, turning on the living room light. <<turn_on:Living Room Light>>"
You can include multiple actions in one response.
If the user asks to turn on/off "the lights" or "all lights", control all available devices."""


class Assistant:
    """Chat with Claude through the local OpenClaw gateway."""

    def __init__(self, device_list: List[str] = None):
        self.session_user = "claudinho-voice"
        self.system_prompt = BASE_SYSTEM_PROMPT

        if device_list:
            devices_str = "\n".join(device_list)
            self.system_prompt += SMART_HOME_PROMPT.format(devices=devices_str)
            logger.info(f"🏠 Smart home enabled with {len(device_list)} device(s)")

        if not OPENCLAW_TOKEN:
            raise ValueError(
                "No OpenClaw gateway token configured.\n"
                "Set OPENCLAW_TOKEN in config.py"
            )

        logger.info("Connected to OpenClaw gateway")

    # ── Streaming chat ────────────────────────────────────────

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        """Stream a response from Claude, yielding text chunks."""
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
                        {"role": "system", "content": self.system_prompt},
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
        """Stream response and yield complete sentences."""
        buffer = ""

        for chunk in self.chat_stream(text):
            buffer += chunk

            while True:
                match = re.search(r'([.!?])\s', buffer)
                if match:
                    end = match.end()
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]
                    if sentence:
                        yield sentence
                else:
                    break

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
                        {"role": "system", "content": self.system_prompt},
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
