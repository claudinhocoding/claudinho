"""
Assistant
=========

Handles conversation with Claude via Anthropic SDK.
Maintains conversation history within a session.
"""

import logging
import os

import anthropic

import config

logger = logging.getLogger(__name__)


class Assistant:
    """Chat with Claude using the Anthropic Python SDK."""
    
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation: list[dict] = []
        self.max_history = 20  # keep last N turns to manage context
    
    def chat(self, text: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            text: User's transcribed speech.
            
        Returns:
            Claude's response text.
        """
        # Add user message
        self.conversation.append({"role": "user", "content": text})
        
        # Trim history if too long
        if len(self.conversation) > self.max_history:
            self.conversation = self.conversation[-self.max_history:]
        
        try:
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=300,  # keep responses short for voice
                system=config.SYSTEM_PROMPT,
                messages=self.conversation,
            )
            
            reply = response.content[0].text
            
            # Add assistant response to history
            self.conversation.append({"role": "assistant", "content": reply})
            
            return reply
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return "Sorry, I'm having trouble thinking right now."
        except Exception as e:
            logger.error(f"Assistant error: {e}")
            return "Sorry, something went wrong."
    
    def reset(self):
        """Clear conversation history."""
        self.conversation.clear()
        logger.info("Conversation history cleared")
