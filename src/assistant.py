"""
Assistant
=========

Integrates with OpenClaw to communicate with Claude.
Handles conversation context and message formatting.
"""

import logging
import requests
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class Assistant:
    """Communicates with Claude via OpenClaw Gateway."""
    
    def __init__(
        self,
        gateway_url: str = "http://localhost:18789",
        gateway_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.gateway_url = gateway_url.rstrip('/')
        self.gateway_token = gateway_token
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation_history: List[Dict] = []
    
    def _default_system_prompt(self) -> str:
        return """You are Claudinho, a friendly voice assistant running on a Raspberry Pi.

Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail.
Be helpful, warm, and occasionally playful.

Remember: you're speaking, not writing. Use natural speech patterns."""
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.gateway_token:
            headers["Authorization"] = f"Bearer {self.gateway_token}"
        return headers
    
    def chat(self, message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: User's message.
            
        Returns:
            Assistant's response text.
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Try OpenClaw Gateway API
            response = self._chat_via_gateway(message)
        except Exception as e:
            logger.warning(f"Gateway error: {e}, falling back to direct API")
            response = self._chat_fallback(message)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _chat_via_gateway(self, message: str) -> str:
        """Chat via OpenClaw Gateway."""
        # Use the sessions_send endpoint or direct chat
        endpoint = f"{self.gateway_url}/api/chat"
        
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history
            ]
        }
        
        response = requests.post(
            endpoint,
            json=payload,
            headers=self._get_headers(),
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Gateway returned {response.status_code}: {response.text}")
        
        data = response.json()
        return data.get("response", data.get("content", ""))
    
    def _chat_fallback(self, message: str) -> str:
        """Fallback: direct Anthropic API call."""
        import os
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "I'm having trouble connecting. Please check the OpenClaw gateway."
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "system": self.system_prompt,
                "messages": self.conversation_history
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"API error: {response.text}")
            return "Sorry, I'm having trouble thinking right now."
        
        data = response.json()
        return data["content"][0]["text"]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


if __name__ == "__main__":
    # Test assistant
    assistant = Assistant()
    
    print("Testing assistant...")
    response = assistant.chat("Hello! What can you help me with?")
    print(f"Response: {response}")
