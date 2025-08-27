"""
Conversation State Management System.

This module implements a comprehensive conversation state machine for managing
natural conversation flow, turn-taking, interruption handling, and context
preservation in real-time voice interactions.
"""

import time
import uuid
import logging
import threading
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .event_bus import EventBusService

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation states for natural flow management."""
    IDLE = auto()           # Not in conversation
    LISTENING = auto()      # Actively listening for user input
    PROCESSING = auto()     # Processing user input (STT + AI)
    RESPONDING = auto()     # AI is speaking response
    WAITING = auto()        # Brief pause waiting for user continuation
    INTERRUPTED = auto()    # User interrupted AI response
    ERROR = auto()          # Error state requiring recovery


class TurnType(Enum):
    """Types of conversation turns."""
    USER_SPEECH = auto()    # User speaking
    AI_RESPONSE = auto()    # AI responding
    INTERRUPTION = auto()   # User interruption
    CLARIFICATION = auto()  # AI requesting clarification
    CONTINUATION = auto()   # Continuing previous topic


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_type: TurnType = TurnType.USER_SPEECH
    speaker: str = "user"  # "user" or "assistant"
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    context_references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            'turn_id': self.turn_id,
            'turn_type': self.turn_type.name,
            'speaker': self.speaker,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'context_references': self.context_references,
            'metadata': self.metadata
        }


@dataclass
class ConversationContext:
    """Maintains conversation context and history."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[ConversationTurn] = field(default_factory=list)
    current_topic: Optional[str] = None
    context_keywords: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(turn)
        self.last_activity = datetime.now()
        
        # Update context keywords from turn content
        if turn.content:
            # Simple keyword extraction (could be enhanced with NLP)
            words = turn.content.lower().split()
            keywords = [w for w in words if len(w) > 3 and w.isalpha()]
            self.context_keywords.extend(keywords[-5:])  # Keep recent keywords
            
            # Limit context keywords to prevent memory bloat
            if len(self.context_keywords) > 50:
                self.context_keywords = self.context_keywords[-30:]
    
    def get_recent_context(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns for context."""
        return self.turns[-max_turns:] if self.turns else []
    
    def get_context_summary(self) -> str:
        """Generate a summary of current conversation context."""
        if not self.turns:
            return "No conversation history."
        
        recent_turns = self.get_recent_context(3)
        summary_parts = []
        
        for turn in recent_turns:
            speaker = "User" if turn.speaker == "user" else "Assistant"
            content = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            summary_parts.append(f"{speaker}: {content}")
        
        return " | ".join(summary_parts)


class ConversationStateManager:
    """
    Manages conversation state transitions and natural flow.
    
    Handles state machine logic, turn-taking, interruption detection,
    and context preservation for seamless voice interactions.
    """
    
    def __init__(self, event_bus: EventBusService):
        """
        Initialize conversation state manager.
        
        Args:
            event_bus: Event bus service for state notifications
        """
        self._event_bus = event_bus
        self._current_state = ConversationState.IDLE
        self._context = ConversationContext()
        self._state_lock = threading.RLock()
        
        # State transition callbacks
        self._state_callbacks: Dict[ConversationState, List[Callable]] = {
            state: [] for state in ConversationState
        }
        
        # Timing configuration
        self._silence_timeout = 3.0  # Seconds of silence before timeout
        self._response_timeout = 30.0  # Max response time
        self._interruption_threshold = 0.5  # Seconds to detect interruption
        
        # State timing
        self._state_start_time = time.time()
        self._last_user_input = None
        self._last_ai_response = None
        
        # Turn-taking management
        self._turn_taking_enabled = True
        self._auto_listen_after_response = True
        self._conversation_active = False
        
        logger.info("Conversation State Manager initialized")
    
    @property
    def current_state(self) -> ConversationState:
        """Get current conversation state."""
        with self._state_lock:
            return self._current_state
    
    @property
    def context(self) -> ConversationContext:
        """Get current conversation context."""
        return self._context
    
    @property
    def is_conversation_active(self) -> bool:
        """Check if conversation is currently active."""
        return self._conversation_active
    
    def start_conversation(self) -> bool:
        """
        Start a new conversation session.
        
        Returns:
            bool: True if conversation started successfully
        """
        try:
            with self._state_lock:
                if self._current_state != ConversationState.IDLE:
                    logger.warning(f"Cannot start conversation from state: {self._current_state}")
                    return False
                
                # Reset context for new conversation
                self._context = ConversationContext()
                self._conversation_active = True
                
                # Transition to listening state
                self._transition_to_state(ConversationState.LISTENING)
                
                # Publish conversation started event
                self._event_bus.publish('conversation.started', {
                    'conversation_id': self._context.conversation_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Conversation started: {self._context.conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return False
    
    def end_conversation(self) -> bool:
        """
        End the current conversation session.
        
        Returns:
            bool: True if conversation ended successfully
        """
        try:
            with self._state_lock:
                if not self._conversation_active:
                    logger.warning("No active conversation to end")
                    return False
                
                # Transition to idle state
                self._transition_to_state(ConversationState.IDLE)
                self._conversation_active = False
                
                # Publish conversation ended event
                self._event_bus.publish('conversation.ended', {
                    'conversation_id': self._context.conversation_id,
                    'duration': (datetime.now() - self._context.started_at).total_seconds(),
                    'turns_count': len(self._context.turns),
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Conversation ended: {self._context.conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return False
    
    def handle_user_speech_start(self) -> bool:
        """
        Handle user starting to speak.
        
        Returns:
            bool: True if transition handled successfully
        """
        try:
            with self._state_lock:
                current_state = self._current_state
                
                if current_state == ConversationState.LISTENING:
                    # Normal speech start during listening
                    self._transition_to_state(ConversationState.PROCESSING)
                    return True
                
                elif current_state == ConversationState.RESPONDING:
                    # User interruption during AI response
                    self._transition_to_state(ConversationState.INTERRUPTED)
                    
                    # Publish interruption event
                    self._event_bus.publish('conversation.interrupted', {
                        'conversation_id': self._context.conversation_id,
                        'interrupted_at': datetime.now().isoformat()
                    })
                    
                    return True
                
                elif current_state == ConversationState.WAITING:
                    # User continuing conversation
                    self._transition_to_state(ConversationState.PROCESSING)
                    return True
                
                else:
                    logger.warning(f"Unexpected user speech in state: {current_state}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to handle user speech start: {e}")
            return False
    
    def handle_user_speech_complete(self, transcription: str, confidence: float = 1.0) -> bool:
        """
        Handle completed user speech transcription.
        
        Args:
            transcription: The transcribed text
            confidence: Transcription confidence score
            
        Returns:
            bool: True if handled successfully
        """
        try:
            with self._state_lock:
                # Create conversation turn
                turn = ConversationTurn(
                    turn_type=TurnType.USER_SPEECH,
                    speaker="user",
                    content=transcription,
                    confidence=confidence,
                    metadata={'state_when_received': self._current_state.name}
                )
                
                # Add to context
                self._context.add_turn(turn)
                self._last_user_input = time.time()
                
                # Publish user turn event
                self._event_bus.publish('conversation.user_turn', {
                    'turn': turn.to_dict(),
                    'context_summary': self._context.get_context_summary()
                })
                
                logger.info(f"User speech complete: '{transcription[:50]}...'")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle user speech complete: {e}")
            return False
    
    def handle_ai_response_start(self) -> bool:
        """
        Handle AI starting to generate/speak response.
        
        Returns:
            bool: True if transition handled successfully
        """
        try:
            with self._state_lock:
                if self._current_state in [ConversationState.PROCESSING, ConversationState.INTERRUPTED]:
                    self._transition_to_state(ConversationState.RESPONDING)
                    return True
                else:
                    logger.warning(f"Unexpected AI response start in state: {self._current_state}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to handle AI response start: {e}")
            return False
    
    def handle_ai_response_complete(self, response: str) -> bool:
        """
        Handle completed AI response.
        
        Args:
            response: The AI response text
            
        Returns:
            bool: True if handled successfully
        """
        try:
            with self._state_lock:
                # Create conversation turn
                turn = ConversationTurn(
                    turn_type=TurnType.AI_RESPONSE,
                    speaker="assistant",
                    content=response,
                    metadata={'state_when_completed': self._current_state.name}
                )
                
                # Add to context
                self._context.add_turn(turn)
                self._last_ai_response = time.time()
                
                # Publish AI turn event
                self._event_bus.publish('conversation.ai_turn', {
                    'turn': turn.to_dict(),
                    'context_summary': self._context.get_context_summary()
                })
                
                # Transition based on turn-taking settings
                if self._auto_listen_after_response:
                    self._transition_to_state(ConversationState.WAITING)
                else:
                    self._transition_to_state(ConversationState.IDLE)
                
                logger.info(f"AI response complete: '{response[:50]}...'")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle AI response complete: {e}")
            return False
    
    def handle_silence_detected(self, duration: float) -> bool:
        """
        Handle silence detection.
        
        Args:
            duration: Duration of silence in seconds
            
        Returns:
            bool: True if handled successfully
        """
        try:
            with self._state_lock:
                if self._current_state == ConversationState.WAITING and duration >= self._silence_timeout:
                    # Timeout waiting for user - return to listening
                    self._transition_to_state(ConversationState.LISTENING)
                    
                    self._event_bus.publish('conversation.silence_timeout', {
                        'duration': duration,
                        'conversation_id': self._context.conversation_id
                    })
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to handle silence detection: {e}")
            return False
    
    def handle_error(self, error: str, recoverable: bool = True) -> bool:
        """
        Handle conversation errors.
        
        Args:
            error: Error description
            recoverable: Whether the error is recoverable
            
        Returns:
            bool: True if handled successfully
        """
        try:
            with self._state_lock:
                if recoverable:
                    # Try to recover to listening state
                    self._transition_to_state(ConversationState.LISTENING)
                else:
                    # Unrecoverable error - transition to error state
                    self._transition_to_state(ConversationState.ERROR)
                
                self._event_bus.publish('conversation.error', {
                    'error': error,
                    'recoverable': recoverable,
                    'conversation_id': self._context.conversation_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.error(f"Conversation error: {error} (recoverable: {recoverable})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle conversation error: {e}")
            return False
    
    def _transition_to_state(self, new_state: ConversationState) -> None:
        """
        Transition to a new conversation state.
        
        Args:
            new_state: The state to transition to
        """
        old_state = self._current_state
        self._current_state = new_state
        self._state_start_time = time.time()
        
        # Execute state callbacks
        for callback in self._state_callbacks.get(new_state, []):
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
        
        # Publish state change event
        self._event_bus.publish('conversation.state_changed', {
            'old_state': old_state.name,
            'new_state': new_state.name,
            'conversation_id': self._context.conversation_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"State transition: {old_state.name} → {new_state.name}")
    
    def register_state_callback(self, state: ConversationState, callback: Callable) -> None:
        """
        Register a callback for state transitions.
        
        Args:
            state: The state to register callback for
            callback: Callback function (old_state, new_state) -> None
        """
        self._state_callbacks[state].append(callback)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dict containing conversation statistics
        """
        with self._state_lock:
            duration = (datetime.now() - self._context.started_at).total_seconds()
            
            return {
                'conversation_id': self._context.conversation_id,
                'current_state': self._current_state.name,
                'duration_seconds': duration,
                'turns_count': len(self._context.turns),
                'user_turns': len([t for t in self._context.turns if t.speaker == "user"]),
                'ai_turns': len([t for t in self._context.turns if t.speaker == "assistant"]),
                'context_keywords': self._context.context_keywords[-10:],  # Recent keywords
                'last_activity': self._context.last_activity.isoformat(),
                'conversation_active': self._conversation_active
            }
