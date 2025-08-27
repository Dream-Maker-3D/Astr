import asyncio
import logging
from typing import Dict, Any, Optional
import time
import uuid

from .openrouter_client import OpenRouterClient
from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import AIServiceError


class AIConversationService:
    """
    AI conversation service implementing Command pattern for request handling.
    Manages natural conversation flow with Claude AI integration.
    """
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self.config = ConfigurationManager()
        self._ai_client = OpenRouterClient()
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        
        # Conversation state
        self._current_conversation_id: Optional[str] = None
        self._is_processing = False
        self._last_user_input = ""
        self._conversation_start_time = 0
        
        # Subscribe to speech recognition events
        self.event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._handle_speech_recognized)
        self.event_bus.subscribe(EventTypes.NATURAL_SPEECH_DETECTED, self._handle_natural_speech)
        self.event_bus.subscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
    
    async def initialize(self) -> bool:
        """Initialize the AI conversation service."""
        try:
            # Initialize OpenRouter client
            success = await self._ai_client.initialize()
            if success:
                self._is_initialized = True
                
                await self.event_bus.publish_async(
                    EventTypes.SYSTEM_STATUS_CHANGED,
                    {'service': 'ai_conversation', 'status': 'initialized'}
                )
                
                self._logger.info("AI conversation service initialized")
                return True
            else:
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to initialize AI conversation service: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'ai_conversation', 'error': str(e)}
            )
            return False
    
    async def _handle_speech_recognized(self, event_data: Dict[str, Any]) -> None:
        """Handle speech recognition results."""
        try:
            text = event_data['data'].get('text', '').strip()
            confidence = event_data['data'].get('confidence', 0.0)
            
            if text and confidence > 0.6:  # Only process high-confidence transcriptions
                await self._process_user_input(text)
                
        except Exception as e:
            self._logger.error(f"Error handling speech recognition: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'ai_conversation', 'error': str(e)}
            )
    
    async def _handle_natural_speech(self, event_data: Dict[str, Any]) -> None:
        """Handle natural speech detection for conversational flow."""
        try:
            text = event_data['data'].get('text', '').strip()
            
            # For natural conversation, process any detected speech
            if text:
                await self._process_user_input(text)
                
        except Exception as e:
            self._logger.error(f"Error handling natural speech: {e}")
    
    async def _handle_interruption(self, event_data: Dict[str, Any]) -> None:
        """Handle interruption during AI processing or response."""
        if self._is_processing:
            self._logger.debug("AI processing interrupted by user speech")
            # In a more advanced implementation, we could cancel the current request
            # For now, we just log the interruption
    
    async def _process_user_input(self, user_input: str) -> None:
        """Process user input and generate AI response."""
        if not self._is_initialized:
            raise AIServiceError("AI conversation service not initialized")
        
        if not user_input.strip():
            return
        
        try:
            self._is_processing = True
            self._last_user_input = user_input
            
            # Start or continue conversation
            if not self._current_conversation_id:
                self._current_conversation_id = str(uuid.uuid4())
                self._conversation_start_time = time.time()
                
                await self.event_bus.publish_async(
                    EventTypes.CONVERSATION_STARTED,
                    {'conversation_id': self._current_conversation_id}
                )
            
            # Publish AI request sent event
            await self.event_bus.publish_async(
                EventTypes.AI_REQUEST_SENT,
                {
                    'text': user_input,
                    'conversation_id': self._current_conversation_id
                }
            )
            
            # Send message to AI
            response = await self._ai_client.send_message(
                user_input,
                self._current_conversation_id
            )
            
            if response and response.get('text'):
                # Publish AI response received event
                await self.event_bus.publish_async(
                    EventTypes.AI_RESPONSE_RECEIVED,
                    {
                        'text': response['text'],
                        'conversation_id': self._current_conversation_id,
                        'user_input': user_input,
                        'response_time': response.get('response_time', 0),
                        'usage': response.get('usage', {})
                    }
                )
                
                # Update conversation context
                await self.event_bus.publish_async(
                    EventTypes.CONTEXT_UPDATED,
                    {
                        'conversation_id': self._current_conversation_id,
                        'user_input': user_input,
                        'ai_response': response['text']
                    }
                )
            
        except Exception as e:
            self._logger.error(f"Failed to process user input: {e}")
            await self.event_bus.publish_async(
                EventTypes.AI_ERROR,
                {
                    'error': str(e),
                    'user_input': user_input,
                    'conversation_id': self._current_conversation_id
                }
            )
        finally:
            self._is_processing = False
    
    async def send_message_direct(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message directly to Claude (bypassing event system).
        Useful for testing or direct API access.
        """
        if not self._is_initialized:
            raise AIServiceError("AI conversation service not initialized")
        
        try:
            response = await self._ai_client.send_message(message, conversation_id)
            return response
        except Exception as e:
            self._logger.error(f"Direct message failed: {e}")
            raise AIServiceError(f"Direct message failed: {e}")
    
    async def start_new_conversation(self) -> str:
        """Start a new conversation session."""
        self._current_conversation_id = str(uuid.uuid4())
        self._conversation_start_time = time.time()
        self._ai_client.clear_conversation_history()
        
        await self.event_bus.publish_async(
            EventTypes.CONVERSATION_STARTED,
            {'conversation_id': self._current_conversation_id}
        )
        
        self._logger.info(f"Started new conversation: {self._current_conversation_id}")
        return self._current_conversation_id
    
    async def end_conversation(self) -> None:
        """End the current conversation."""
        if self._current_conversation_id:
            await self.event_bus.publish_async(
                EventTypes.CONVERSATION_ENDED,
                {'conversation_id': self._current_conversation_id}
            )
            
            conversation_duration = time.time() - self._conversation_start_time
            self._logger.info(f"Ended conversation {self._current_conversation_id} "
                            f"(duration: {conversation_duration:.1f}s)")
            
            self._current_conversation_id = None
            self._conversation_start_time = 0
    
    def get_current_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._current_conversation_id
    
    def get_conversation_history(self) -> list:
        """Get conversation history from Claude client."""
        return self._ai_client.get_conversation_history()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self._ai_client.clear_conversation_history()
        self._logger.debug("Conversation history cleared")
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for conversation behavior."""
        self._ai_client.set_system_prompt(prompt)
        self._logger.info("System prompt updated")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt."""
        return self._ai_client.get_system_prompt()
    
    def is_processing(self) -> bool:
        """Check if currently processing a user input."""
        return self._is_processing
    
    def is_ready(self) -> bool:
        """Check if the service is ready for conversation."""
        return self._is_initialized and self._ai_client.is_ready()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model configuration."""
        return self._ai_client.get_model_info()
    
    async def test_connection(self) -> bool:
        """Test connection to Claude API."""
        try:
            return await self._ai_client.test_connection()
        except Exception as e:
            self._logger.error(f"Connection test failed: {e}")
            return False
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        history_length = len(self._ai_client.get_conversation_history())
        conversation_duration = 0
        
        if self._conversation_start_time > 0:
            conversation_duration = time.time() - self._conversation_start_time
        
        return {
            'current_conversation_id': self._current_conversation_id,
            'conversation_duration': conversation_duration,
            'history_length': history_length,
            'is_processing': self._is_processing,
            'last_user_input': self._last_user_input
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # End current conversation if active
        if self._current_conversation_id:
            await self.end_conversation()
        
        self._is_initialized = False
        self._is_processing = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(EventTypes.SPEECH_RECOGNIZED, self._handle_speech_recognized)
        self.event_bus.unsubscribe(EventTypes.NATURAL_SPEECH_DETECTED, self._handle_natural_speech)
        self.event_bus.unsubscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
        
        await self.event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'ai_conversation', 'status': 'cleanup_complete'}
        )
        
        self._logger.info("AI conversation service cleanup completed")