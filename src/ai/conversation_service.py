"""
AI Conversation Service for natural AI-powered conversations.

This module provides the main orchestrator for AI conversations, integrating
the OpenRouter client, conversation management, context handling, and Event Bus
communication for seamless voice assistant interactions.
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional, Iterator
from queue import Queue, Empty
from datetime import datetime, timedelta
import uuid

from src.core.event_bus import EventBusService
from .types import (
    AIConfig, ConversationRequest, ConversationResponse, ResponseChunk,
    ConversationTurn, ConversationContext, Message, AIStatistics,
    Speaker, MessageRole, ConversationState, Priority, AIEventTypes,
    AIConversationError, OpenRouterAPIError, RateLimitExceededError,
    ResponseMetadata, TurnMetadata
)
from .openrouter_client import OpenRouterClient
from .conversational_intelligence import ConversationalIntelligence, ConversationalConfig

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history, context, and conversation flow.
    
    Handles conversation turns, context window management, and conversation
    state tracking for natural AI interactions.
    """
    
    def __init__(self, config: AIConfig):
        """Initialize conversation manager."""
        self.config = config
        self.conversation_history: List[ConversationTurn] = []
        self.conversation_context = ConversationContext()
        self.conversation_state = ConversationState.IDLE
        self.current_conversation_id = str(uuid.uuid4())
        self.max_context_turns = 20  # Configurable context window
        self.lock = threading.Lock()
        
        logger.info(f"Conversation manager initialized with context window: {self.max_context_turns}")
    
    def add_user_turn(self, message: str, confidence: float = 1.0) -> ConversationTurn:
        """Add a user turn to conversation history."""
        with self.lock:
            turn = ConversationTurn(
                speaker=Speaker.USER,
                message=message,
                metadata=TurnMetadata(
                    confidence=confidence,
                    processing_time=0.0,
                    model_used="",
                    token_count=len(message) // 4  # Rough estimate
                )
            )
            
            self.conversation_history.append(turn)
            self._manage_context_window()
            
            logger.debug(f"Added user turn: '{message[:50]}...' (confidence: {confidence})")
            return turn
    
    def add_assistant_turn(self, message: str, model_used: str, processing_time: float, 
                          token_count: int) -> ConversationTurn:
        """Add an assistant turn to conversation history."""
        with self.lock:
            turn = ConversationTurn(
                speaker=Speaker.ASSISTANT,
                message=message,
                metadata=TurnMetadata(
                    confidence=1.0,
                    processing_time=processing_time,
                    model_used=model_used,
                    token_count=token_count
                )
            )
            
            self.conversation_history.append(turn)
            self._manage_context_window()
            
            logger.debug(f"Added assistant turn: '{message[:50]}...' (model: {model_used})")
            return turn
    
    def get_context_messages(self) -> List[Message]:
        """Get conversation history as messages for AI processing."""
        with self.lock:
            messages = []
            
            # Add system message
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=self.config.system_prompt
            ))
            
            # Add conversation history
            for turn in self.conversation_history:
                messages.append(turn.to_message())
            
            logger.debug(f"Generated {len(messages)} context messages")
            return messages
    
    def _manage_context_window(self) -> None:
        """Manage conversation context window to stay within limits."""
        if len(self.conversation_history) > self.max_context_turns:
            # Remove oldest turns but keep recent context
            turns_to_remove = len(self.conversation_history) - self.max_context_turns
            removed_turns = self.conversation_history[:turns_to_remove]
            self.conversation_history = self.conversation_history[turns_to_remove:]
            
            logger.info(f"Removed {len(removed_turns)} old turns to manage context window")
    
    def update_conversation_state(self, state: ConversationState) -> None:
        """Update the current conversation state."""
        with self.lock:
            old_state = self.conversation_state
            self.conversation_state = state
            logger.debug(f"Conversation state: {old_state.value} → {state.value}")
    
    def clear_conversation(self) -> None:
        """Clear conversation history and reset context."""
        with self.lock:
            self.conversation_history.clear()
            self.conversation_context = ConversationContext()
            self.current_conversation_id = str(uuid.uuid4())
            self.conversation_state = ConversationState.IDLE
            
            logger.info("Conversation history cleared")
    
    def get_recent_turns(self, count: int = 3) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        with self.lock:
            return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        with self.lock:
            total_turns = len(self.conversation_history)
            user_turns = sum(1 for turn in self.conversation_history if turn.speaker == Speaker.USER)
            assistant_turns = sum(1 for turn in self.conversation_history if turn.speaker == Speaker.ASSISTANT)
            
            return {
                'conversation_id': self.current_conversation_id,
                'total_turns': total_turns,
                'user_turns': user_turns,
                'assistant_turns': assistant_turns,
                'state': self.conversation_state.value,
                'context_window_usage': f"{total_turns}/{self.max_context_turns}"
            }


class AIConversationService:
    """
    Main AI Conversation Service for natural AI-powered conversations.
    
    Orchestrates AI conversations by integrating OpenRouter client, conversation
    management, Event Bus communication, and response processing for seamless
    voice assistant interactions.
    """
    
    def __init__(self, event_bus: EventBusService, config: AIConfig):
        """
        Initialize the AI Conversation Service.
        
        Args:
            event_bus (EventBusService): Event bus for communication
            config (AIConfig): AI conversation configuration
        """
        self.event_bus = event_bus
        self.config = config
        self.openrouter_client = OpenRouterClient(config)
        self.conversation_manager = ConversationManager(config)
        self.conversational_intelligence = ConversationalIntelligence(ConversationalConfig())
        self.statistics = AIStatistics()
        
        # Processing queue and worker thread
        self.processing_queue: Queue[ConversationRequest] = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_initialized = False
        
        # Current processing state
        self.current_request_id: Optional[str] = None
        self.processing_lock = threading.Lock()
        
        logger.info("AI Conversation Service created")
    
    def initialize(self) -> bool:
        """
        Initialize the AI Conversation Service.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing AI Conversation Service...")
            
            # Initialize OpenRouter client
            if not self.openrouter_client.initialize():
                logger.error("Failed to initialize OpenRouter client")
                return False
            
            # Subscribe to Event Bus events
            self._subscribe_to_events()
            
            # Start worker thread
            self.is_running = True
            self.worker_thread = threading.Thread(
                target=self._process_conversation_worker,
                name="AIConversationWorker",
                daemon=True
            )
            self.worker_thread.start()
            
            self.is_initialized = True
            
            # Publish initialization event
            self.event_bus.publish(AIEventTypes.AI_SERVICE_INITIALIZED, {
                'service': 'AIConversationService',
                'model': self.config.default_model,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("AI Conversation Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Conversation Service: {e}")
            return False
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant Event Bus events."""
        # Subscribe to transcription events from STT service
        self.event_bus.subscribe("TRANSCRIPTION_COMPLETED", self._handle_transcription_event)
        
        # Subscribe to interruption events
        self.event_bus.subscribe("AI_INTERRUPTION", self._handle_interruption_event)
        
        logger.debug("Subscribed to Event Bus events")
    
    def _handle_transcription_event(self, event_data: Dict[str, Any]) -> None:
        """Handle transcription ready events from STT service."""
        logger.info(f"🎯 Received transcription event: '{event_data.get('text', 'N/A')}' (confidence: {event_data.get('confidence', 0.0):.3f})")
        try:
            text = event_data.get('text', '')
            confidence = event_data.get('confidence', 0.0)
            
            if text and confidence > 0.05:  # Process transcriptions with reasonable confidence
                logger.info(f"Processing transcription: '{text}' (confidence: {confidence})")
                
                # Create conversation request
                request = ConversationRequest(
                    text=text,
                    context=self.conversation_manager.conversation_context,
                    priority=Priority.NORMAL,
                    streaming=self.config.streaming_enabled
                )
                
                # Add to processing queue
                self.processing_queue.put(request)
            else:
                logger.debug(f"Skipping low-confidence transcription: {confidence}")
                
        except Exception as e:
            logger.error(f"Error handling transcription event: {e}")
    
    def _handle_interruption_event(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation interruption events."""
        try:
            with self.processing_lock:
                if self.current_request_id:
                    logger.info(f"Interrupting current conversation: {self.current_request_id}")
                    
                    # Update conversation state
                    self.conversation_manager.update_conversation_state(ConversationState.INTERRUPTED)
                    
                    # Publish interruption event
                    self.event_bus.publish(AIEventTypes.AI_RESPONSE_INTERRUPTED, {
                        'request_id': self.current_request_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Clear current request
                    self.current_request_id = None
                    
        except Exception as e:
            logger.error(f"Error handling interruption event: {e}")
    
    def process_message(self, text: str, context: Optional[ConversationContext] = None) -> ConversationResponse:
        """
        Process a single message and return AI response.
        
        Args:
            text (str): User message text
            context (ConversationContext, optional): Conversation context
            
        Returns:
            ConversationResponse: AI response with metadata
        """
        try:
            if not self.is_initialized:
                raise AIConversationError("Service not initialized")
            
            logger.info(f"Processing message: '{text[:50]}...'")
            start_time = time.time()
            
            # Add user turn to conversation
            user_turn = self.conversation_manager.add_user_turn(text)
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(ConversationState.PROCESSING)
            
            # Publish processing started event
            self.event_bus.publish(AIEventTypes.AI_PROCESSING_STARTED, {
                'user_input': text,
                'model': self.openrouter_client.current_model,
                'request_id': user_turn.turn_id,
                'timestamp': datetime.now().isoformat()
            })
            
            # Get context messages
            context_messages = self.conversation_manager.get_context_messages()
            
            # Generate AI response
            raw_response = self.openrouter_client.generate_response(context_messages, stream=False)
            
            # Apply conversational intelligence optimization
            conversation_context = {
                'recent_turns': [turn.to_dict() for turn in self.conversation_manager.get_recent_turns(3)]
            }
            response_text = self.conversational_intelligence.optimize_response(raw_response, conversation_context)
            
            processing_time = time.time() - start_time
            
            # Add assistant turn to conversation
            assistant_turn = self.conversation_manager.add_assistant_turn(
                response_text,
                self.openrouter_client.current_model,
                processing_time,
                len(response_text) // 4  # Rough token estimate
            )
            
            # Create response object
            response = ConversationResponse(
                text=response_text,
                model_used=self.openrouter_client.current_model,
                processing_time=processing_time,
                token_count=len(response_text) // 4,
                confidence=1.0,
                metadata=ResponseMetadata(
                    model_name=self.openrouter_client.current_model,
                    processing_device="cloud",
                    token_usage={'total_tokens': len(response_text) // 4},
                    quality_metrics={'processing_time': processing_time}
                ),
                request_id=user_turn.turn_id
            )
            
            # Update statistics
            self.statistics.total_requests += 1
            self.statistics.average_response_time = (
                (self.statistics.average_response_time * (self.statistics.total_requests - 1) + processing_time) /
                self.statistics.total_requests
            )
            self.statistics.total_tokens_used += response.token_count
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(ConversationState.IDLE)
            
            # Publish response ready event
            self.event_bus.publish(AIEventTypes.AI_RESPONSE_READY, {
                'text': response.format_for_tts(),
                'model_used': response.model_used,
                'processing_time': response.processing_time,
                'token_count': response.token_count,
                'request_id': response.request_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Message processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.statistics.error_count += 1
            self.conversation_manager.update_conversation_state(ConversationState.ERROR)
            
            # Publish error event
            self.event_bus.publish(AIEventTypes.AI_ERROR, {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'user_input': text,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.error(f"Error processing message: {e}")
            raise AIConversationError(f"Failed to process message: {e}")
    
    def process_message_stream(self, text: str, context: Optional[ConversationContext] = None) -> Iterator[ResponseChunk]:
        """
        Process message with streaming response for real-time conversation.
        
        Args:
            text (str): User message text
            context (ConversationContext, optional): Conversation context
            
        Yields:
            ResponseChunk: Response chunks as they become available
        """
        try:
            if not self.is_initialized:
                raise AIConversationError("Service not initialized")
            
            logger.info(f"Processing streaming message: '{text[:50]}...'")
            start_time = time.time()
            
            # Add user turn to conversation
            user_turn = self.conversation_manager.add_user_turn(text)
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(ConversationState.PROCESSING)
            
            # Publish processing started event
            self.event_bus.publish(AIEventTypes.AI_PROCESSING_STARTED, {
                'user_input': text,
                'model': self.openrouter_client.current_model,
                'request_id': user_turn.turn_id,
                'streaming': True,
                'timestamp': datetime.now().isoformat()
            })
            
            # Get context messages
            context_messages = self.conversation_manager.get_context_messages()
            
            # Generate streaming AI response
            full_response = ""
            chunk_count = 0
            
            for chunk_text in self.openrouter_client.generate_response(context_messages, stream=True):
                chunk_count += 1
                full_response += chunk_text
                
                # Create response chunk
                chunk = ResponseChunk(
                    chunk_text=chunk_text,
                    chunk_id=f"{user_turn.turn_id}-{chunk_count}",
                    is_final=False,
                    timestamp=time.time(),
                    model_used=self.openrouter_client.current_model
                )
                
                # Publish chunk event
                self.event_bus.publish(AIEventTypes.AI_RESPONSE_CHUNK, {
                    'chunk_text': chunk_text,
                    'chunk_id': chunk.chunk_id,
                    'request_id': user_turn.turn_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                yield chunk
            
            # Create final chunk
            final_chunk = ResponseChunk(
                chunk_text="",
                chunk_id=f"{user_turn.turn_id}-final",
                is_final=True,
                timestamp=time.time(),
                model_used=self.openrouter_client.current_model
            )
            
            processing_time = time.time() - start_time
            
            # Add assistant turn to conversation
            self.conversation_manager.add_assistant_turn(
                full_response,
                self.openrouter_client.current_model,
                processing_time,
                len(full_response) // 4
            )
            
            # Update statistics
            self.statistics.total_requests += 1
            self.statistics.average_response_time = (
                (self.statistics.average_response_time * (self.statistics.total_requests - 1) + processing_time) /
                self.statistics.total_requests
            )
            self.statistics.total_tokens_used += len(full_response) // 4
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(ConversationState.IDLE)
            
            # Apply conversational intelligence optimization to final response
            conversation_context = {
                'recent_turns': [turn.to_dict() for turn in self.conversation_manager.get_recent_turns(3)]
            }
            optimized_response = self.conversational_intelligence.optimize_response(full_response, conversation_context)
            
            # Publish final response event
            self.event_bus.publish(AIEventTypes.AI_RESPONSE_READY, {
                'text': optimized_response,
                'model_used': self.openrouter_client.current_model,
                'processing_time': processing_time,
                'token_count': len(full_response) // 4,
                'request_id': user_turn.turn_id,
                'streaming': True,
                'chunk_count': chunk_count,
                'timestamp': datetime.now().isoformat()
            })
            
            yield final_chunk
            
            logger.info(f"Streaming message processed: {chunk_count} chunks in {processing_time:.2f}s")
            
        except Exception as e:
            self.statistics.error_count += 1
            self.conversation_manager.update_conversation_state(ConversationState.ERROR)
            
            # Publish error event
            self.event_bus.publish(AIEventTypes.AI_ERROR, {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'user_input': text,
                'streaming': True,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.error(f"Error processing streaming message: {e}")
            raise AIConversationError(f"Failed to process streaming message: {e}")
    
    def _process_conversation_worker(self) -> None:
        """Worker thread for processing conversation requests."""
        logger.info("AI Conversation worker thread started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                request = self.processing_queue.get(timeout=1.0)
                
                with self.processing_lock:
                    self.current_request_id = request.request_id
                
                # Process the request
                if request.streaming:
                    # Process streaming request
                    chunks = list(self.process_message_stream(request.text, request.context))
                    logger.debug(f"Processed streaming request: {len(chunks)} chunks")
                else:
                    # Process regular request
                    response = self.process_message(request.text, request.context)
                    logger.debug(f"Processed regular request: {len(response.text)} chars")
                
                with self.processing_lock:
                    self.current_request_id = None
                
                self.processing_queue.task_done()
                
            except Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                logger.error(f"Error in conversation worker: {e}")
                with self.processing_lock:
                    self.current_request_id = None
        
        logger.info("AI Conversation worker thread stopped")
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different AI model.
        
        Args:
            model_name (str): Name of the model to switch to
            
        Returns:
            bool: True if model switch successful
        """
        try:
            old_model = self.openrouter_client.current_model
            self.openrouter_client.set_model(model_name)
            
            # Publish model change event
            self.event_bus.publish(AIEventTypes.AI_MODEL_CHANGED, {
                'old_model': old_model,
                'new_model': model_name,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Switched AI model: {old_model} → {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model to {model_name}: {e}")
            return False
    
    def get_conversation_history(self) -> List[ConversationTurn]:
        """
        Get current conversation history.
        
        Returns:
            List[ConversationTurn]: Conversation history
        """
        return self.conversation_manager.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear conversation history and reset context."""
        self.conversation_manager.clear_conversation()
        logger.info("Conversation cleared")
    
    def get_statistics(self) -> AIStatistics:
        """
        Get AI service statistics.
        
        Returns:
            AIStatistics: Service performance statistics
        """
        # Combine our statistics with OpenRouter client statistics
        client_stats = self.openrouter_client.get_statistics()
        
        combined_stats = AIStatistics(
            total_requests=max(self.statistics.total_requests, client_stats.total_requests),
            average_response_time=client_stats.average_response_time or self.statistics.average_response_time,
            total_tokens_used=max(self.statistics.total_tokens_used, client_stats.total_tokens_used),
            model_usage_distribution=client_stats.model_usage_distribution or self.statistics.model_usage_distribution,
            error_count=max(self.statistics.error_count, client_stats.error_count),
            uptime=max(self.statistics.uptime, client_stats.uptime),
            rate_limit_hits=client_stats.rate_limit_hits
        )
        
        return combined_stats
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        return self.conversation_manager.get_conversation_summary()
    
    def shutdown(self) -> None:
        """Shutdown the AI Conversation Service."""
        logger.info("Shutting down AI Conversation Service...")
        
        # Stop worker thread
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Shutdown OpenRouter client
        self.openrouter_client.shutdown()
        
        # Publish shutdown event
        self.event_bus.publish(AIEventTypes.AI_SERVICE_SHUTDOWN, {
            'service': 'AIConversationService',
            'timestamp': datetime.now().isoformat()
        })
        
        self.is_initialized = False
        logger.info("AI Conversation Service shutdown complete")
