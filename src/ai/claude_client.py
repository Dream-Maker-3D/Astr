import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import time

from anthropic import AsyncAnthropic
from anthropic.types import Message

from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import AIServiceError, NetworkError, ConfigurationError


class ClaudeClient:
    """
    Claude API client wrapper with conversation management and error handling.
    Implements natural conversation patterns for voice assistant use.
    """
    
    def __init__(self):
        self.config = ConfigurationManager()
        self._client: Optional[AsyncAnthropic] = None
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        
        # Conversation context
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt = ""
        
        # Configuration
        ai_config = self.config.get_ai_config()
        self.model = ai_config.get('model', 'claude-3-5-haiku-20241022')
        self.max_tokens = ai_config.get('max_tokens', 150)
        self.temperature = ai_config.get('temperature', 0.8)
        self.timeout = ai_config.get('timeout', 20)
        
        # Conversation settings
        conversation_config = self.config.get_conversation_config()
        self.max_context_length = conversation_config.get('max_context_length', 15)
        self.context_window_seconds = conversation_config.get('context_window_seconds', 600)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms minimum between requests
    
    async def initialize(self) -> bool:
        """Initialize Claude client with API key and system prompt."""
        try:
            ai_config = self.config.get_ai_config()
            api_key_env = ai_config.get('api_key_env', 'ANTHROPIC_API_KEY')
            api_key = self.config.get_env(api_key_env)
            
            if not api_key:
                raise ConfigurationError(f"API key not found in environment variable: {api_key_env}")
            
            # Initialize Anthropic client
            self._client = AsyncAnthropic(api_key=api_key)
            
            # Set up system prompt for natural conversation
            self._system_prompt = ai_config.get('system_prompt', self._get_default_system_prompt())
            
            self._is_initialized = True
            self._logger.info(f"Claude client initialized with model: {self.model}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Claude client: {e}")
            raise AIServiceError(f"Failed to initialize Claude client: {e}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for natural conversation."""
        return """You are having a natural spoken conversation. Be concise and 
        conversational. Respond like a knowledgeable person, not an AI assistant.
        Use natural speech patterns, contractions, and keep responses brief 
        unless asked for details. Handle interruptions and corrections gracefully 
        without acknowledging them explicitly."""
    
    async def send_message(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to Claude and get response.
        
        Args:
            message: User message text
            conversation_id: Optional conversation identifier
            
        Returns:
            Dictionary with response data
        """
        if not self._is_initialized or not self._client:
            raise AIServiceError("Claude client not initialized")
        
        if not message.strip():
            raise AIServiceError("Empty message provided")
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Prepare conversation context
            messages = self._prepare_conversation_context(message)
            
            start_time = time.time()
            
            # Send request to Claude
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._system_prompt,
                messages=messages
            )
            
            response_time = time.time() - start_time
            
            # Extract response text
            response_text = ""
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
            
            # Update conversation history
            self._add_to_conversation_history(message, response_text)
            
            # Clean up old conversation history
            self._cleanup_conversation_history()
            
            result = {
                'text': response_text,
                'conversation_id': conversation_id,
                'model': self.model,
                'response_time': response_time,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
            
            self._logger.debug(f"Claude response generated in {response_time:.2f}s")
            return result
            
        except Exception as e:
            self._logger.error(f"Claude API request failed: {e}")
            raise AIServiceError(f"Claude API request failed: {e}")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _prepare_conversation_context(self, current_message: str) -> List[Dict[str, str]]:
        """Prepare conversation context for Claude API."""
        messages = []
        
        # Add conversation history
        for entry in self._conversation_history[-self.max_context_length:]:
            messages.append({
                "role": "user",
                "content": entry["user"]
            })
            if entry["assistant"]:
                messages.append({
                    "role": "assistant", 
                    "content": entry["assistant"]
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _add_to_conversation_history(self, user_message: str, assistant_response: str) -> None:
        """Add exchange to conversation history."""
        self._conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
    
    def _cleanup_conversation_history(self) -> None:
        """Remove old conversation history based on time and count limits."""
        current_time = time.time()
        
        # Remove entries older than context window
        self._conversation_history = [
            entry for entry in self._conversation_history
            if current_time - entry["timestamp"] < self.context_window_seconds
        ]
        
        # Keep only most recent entries if still too many
        if len(self._conversation_history) > self.max_context_length:
            self._conversation_history = self._conversation_history[-self.max_context_length:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history."""
        return self._conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        self._logger.debug("Conversation history cleared")
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update system prompt for conversation behavior."""
        self._system_prompt = prompt
        self._logger.debug("System prompt updated")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt."""
        return self._system_prompt
    
    async def stream_message(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from Claude for real-time conversation.
        
        Args:
            message: User message text
            conversation_id: Optional conversation identifier
            
        Yields:
            Chunks of response text as they arrive
        """
        if not self._is_initialized or not self._client:
            raise AIServiceError("Claude client not initialized")
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Prepare conversation context
            messages = self._prepare_conversation_context(message)
            
            # Start streaming request
            stream = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._system_prompt,
                messages=messages,
                stream=True
            )
            
            complete_response = ""
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        text_chunk = chunk.delta.text
                        complete_response += text_chunk
                        yield text_chunk
            
            # Add to conversation history
            self._add_to_conversation_history(message, complete_response)
            self._cleanup_conversation_history()
            
        except Exception as e:
            self._logger.error(f"Claude streaming failed: {e}")
            raise AIServiceError(f"Claude streaming failed: {e}")
    
    def is_ready(self) -> bool:
        """Check if Claude client is ready for requests."""
        return self._is_initialized and self._client is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'context_length': self.max_context_length,
            'context_window_seconds': self.context_window_seconds
        }
    
    async def test_connection(self) -> bool:
        """Test connection to Claude API."""
        try:
            response = await self.send_message("Hello", "test")
            return response.get('text') is not None
        except Exception as e:
            self._logger.error(f"Connection test failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'conversation_history_length': len(self._conversation_history),
            'last_request_time': self._last_request_time,
            'is_initialized': self._is_initialized
        }