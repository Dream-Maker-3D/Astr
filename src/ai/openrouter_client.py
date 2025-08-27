import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import time
import os

from openai import AsyncOpenAI

from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import AIServiceError, NetworkError, ConfigurationError


class OpenRouterClient:
    """
    OpenRouter API client wrapper with conversation management and error handling.
    Implements natural conversation patterns for voice assistant use.
    """
    
    def __init__(self):
        self.config = ConfigurationManager()
        self._client: Optional[AsyncOpenAI] = None
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        
        # Conversation context
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt = ""
        
        # Configuration
        ai_config = self.config.get_ai_config()
        self.model = ai_config.get('model', 'anthropic/claude-3.5-sonnet')
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
        """Initialize OpenRouter client with API key and system prompt."""
        try:
            ai_config = self.config.get_ai_config()
            api_key_env = ai_config.get('api_key_env', 'OPENROUTER_API_KEY')
            api_key = self.config.get_env(api_key_env)
            
            if not api_key:
                raise ConfigurationError(f"API key not found in environment variable: {api_key_env}")
            
            # Initialize OpenRouter client (OpenAI-compatible)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional
                    "X-Title": "Voice Assistant",  # Optional
                }
            )
            
            # Set up system prompt for natural conversation
            self._system_prompt = ai_config.get('system_prompt', self._get_default_system_prompt())
            
            self._is_initialized = True
            self._logger.info(f"OpenRouter client initialized with model: {self.model}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise AIServiceError(f"Failed to initialize OpenRouter client: {e}")
    
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
        Send a message to OpenRouter and get response.
        
        Args:
            message: User message text
            conversation_id: Optional conversation identifier
            
        Returns:
            Dictionary with response data
        """
        if not self._is_initialized or not self._client:
            raise AIServiceError("OpenRouter client not initialized")
        
        if not message.strip():
            raise AIServiceError("Empty message provided")
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Prepare conversation context
            messages = self._prepare_conversation_context(message)
            
            start_time = time.time()
            
            # Send request to OpenRouter
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            # Extract response text
            response_text = ""
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content or ""
            
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
                    'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'output_tokens': response.usage.completion_tokens if response.usage else 0
                }
            }
            
            self._logger.debug(f"OpenRouter response generated in {response_time:.2f}s")
            return result
            
        except Exception as e:
            self._logger.error(f"OpenRouter API request failed: {e}")
            raise AIServiceError(f"OpenRouter API request failed: {e}")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _prepare_conversation_context(self, current_message: str) -> List[Dict[str, str]]:
        """Prepare conversation context for OpenRouter API."""
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self._system_prompt
        })
        
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
        Stream response from OpenRouter for real-time conversation.
        
        Args:
            message: User message text
            conversation_id: Optional conversation identifier
            
        Yields:
            Chunks of response text as they arrive
        """
        if not self._is_initialized or not self._client:
            raise AIServiceError("OpenRouter client not initialized")
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Prepare conversation context
            messages = self._prepare_conversation_context(message)
            
            # Start streaming request
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                timeout=self.timeout
            )
            
            complete_response = ""
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        text_chunk = delta.content
                        complete_response += text_chunk
                        yield text_chunk
            
            # Add to conversation history
            self._add_to_conversation_history(message, complete_response)
            self._cleanup_conversation_history()
            
        except Exception as e:
            self._logger.error(f"OpenRouter streaming failed: {e}")
            raise AIServiceError(f"OpenRouter streaming failed: {e}")
    
    def is_ready(self) -> bool:
        """Check if OpenRouter client is ready for requests."""
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
        """Test connection to OpenRouter API."""
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
