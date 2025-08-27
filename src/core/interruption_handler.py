"""
Real-time Interruption and Correction Handler.

This module implements comprehensive interruption detection, processing, and recovery
for natural conversation flow, including user corrections and clarification requests.
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
from .conversation_state import ConversationStateManager, ConversationState, TurnType

logger = logging.getLogger(__name__)


class InterruptionType(Enum):
    """Types of interruptions that can occur."""
    USER_SPEECH = auto()        # User starts speaking during AI response
    CORRECTION = auto()         # User corrects previous statement
    CLARIFICATION = auto()      # User requests clarification
    TOPIC_CHANGE = auto()       # User changes topic mid-conversation
    EMERGENCY_STOP = auto()     # Emergency stop command
    SYSTEM_ERROR = auto()       # System-initiated interruption


class InterruptionPriority(Enum):
    """Priority levels for interruption handling."""
    LOW = 1                     # Background noise, false positives
    NORMAL = 2                  # Regular user speech
    HIGH = 3                    # Clear interruption intent
    CRITICAL = 4                # Emergency stop, system errors


@dataclass
class InterruptionEvent:
    """Represents an interruption event."""
    interruption_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interruption_type: InterruptionType = InterruptionType.USER_SPEECH
    priority: InterruptionPriority = InterruptionPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source_service: str = ""
    confidence: float = 1.0
    audio_data: Optional[bytes] = None
    transcription: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interruption event to dictionary."""
        return {
            'interruption_id': self.interruption_id,
            'interruption_type': self.interruption_type.name,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'source_service': self.source_service,
            'confidence': self.confidence,
            'transcription': self.transcription,
            'context': self.context
        }


@dataclass
class CorrectionEvent:
    """Represents a user correction event."""
    correction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = ""
    corrected_text: str = ""
    correction_type: str = "replacement"  # replacement, addition, deletion
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    context_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert correction event to dictionary."""
        return {
            'correction_id': self.correction_id,
            'original_text': self.original_text,
            'corrected_text': self.corrected_text,
            'correction_type': self.correction_type,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'context_reference': self.context_reference
        }


class InterruptionHandler:
    """
    Handles real-time interruptions and corrections in voice conversations.
    
    Coordinates between audio services, conversation state management, and AI processing
    to provide seamless interruption recovery and natural correction handling.
    """
    
    def __init__(self, event_bus: EventBusService, conversation_manager: ConversationStateManager):
        """
        Initialize interruption handler.
        
        Args:
            event_bus: Event bus service for coordination
            conversation_manager: Conversation state manager
        """
        self._event_bus = event_bus
        self._conversation_manager = conversation_manager
        self._interruption_lock = threading.RLock()
        
        # Interruption detection settings
        self._interruption_threshold = 0.3  # Seconds to detect interruption
        self._voice_activity_threshold = 0.5  # Voice activity confidence threshold
        self._correction_keywords = [
            "no", "wait", "actually", "i mean", "sorry", "correction",
            "that's wrong", "not quite", "let me rephrase", "what i meant"
        ]
        self._clarification_keywords = [
            "what", "how", "why", "when", "where", "can you explain",
            "i don't understand", "clarify", "repeat", "say that again"
        ]
        
        # Active interruption tracking
        self._active_interruptions: Dict[str, InterruptionEvent] = {}
        self._interruption_history: List[InterruptionEvent] = []
        self._correction_history: List[CorrectionEvent] = []
        
        # Service references (will be set by facade)
        self._audio_capture = None
        self._audio_player = None
        self._speech_recognition = None
        self._speech_synthesis = None
        self._ai_conversation = None
        
        # Statistics
        self._interruption_count = 0
        self._correction_count = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Interruption Handler initialized")
    
    def set_services(self, audio_capture=None, audio_player=None, speech_recognition=None, 
                    speech_synthesis=None, ai_conversation=None):
        """Set service references for interruption coordination."""
        self._audio_capture = audio_capture
        self._audio_player = audio_player
        self._speech_recognition = speech_recognition
        self._speech_synthesis = speech_synthesis
        self._ai_conversation = ai_conversation
        
        logger.debug("Service references set for interruption handler")
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for interruption detection."""
        # Voice activity detection
        self._event_bus.subscribe("VOICE_ACTIVITY_DETECTED", self._on_voice_activity_detected)
        
        # Audio interruption events
        self._event_bus.subscribe("INTERRUPTION_DETECTED", self._on_interruption_detected)
        
        # Speech recognition events
        self._event_bus.subscribe("STT_PARTIAL_RESULT", self._on_partial_transcription)
        self._event_bus.subscribe("STT_FINAL_RESULT", self._on_final_transcription)
        
        # AI conversation events
        self._event_bus.subscribe("AI_RESPONSE_STARTED", self._on_ai_response_started)
        self._event_bus.subscribe("AI_RESPONSE_CHUNK", self._on_ai_response_chunk)
        
        # TTS events
        self._event_bus.subscribe("TTS_PLAYBACK_STARTED", self._on_tts_playback_started)
        self._event_bus.subscribe("TTS_PLAYBACK_PROGRESS", self._on_tts_playback_progress)
        
        logger.debug("Event subscriptions setup for interruption handler")
    
    def _on_voice_activity_detected(self, event_data: Dict[str, Any]):
        """Handle voice activity detection during AI response."""
        try:
            with self._interruption_lock:
                current_state = self._conversation_manager.current_state
                
                # Only process if AI is currently responding
                if current_state == ConversationState.RESPONDING:
                    confidence = event_data.get('confidence', 0.0)
                    
                    if confidence >= self._voice_activity_threshold:
                        # Create interruption event
                        interruption = InterruptionEvent(
                            interruption_type=InterruptionType.USER_SPEECH,
                            priority=InterruptionPriority.HIGH,
                            source_service="audio_capture",
                            confidence=confidence,
                            context={'conversation_state': current_state.name}
                        )
                        
                        # Process interruption immediately
                        self._process_interruption(interruption)
                        
        except Exception as e:
            logger.error(f"Error handling voice activity detection: {e}")
    
    def _on_interruption_detected(self, event_data: Dict[str, Any]):
        """Handle explicit interruption detection from audio services."""
        try:
            with self._interruption_lock:
                # Create interruption event from audio service data
                interruption = InterruptionEvent(
                    interruption_type=InterruptionType.USER_SPEECH,
                    priority=InterruptionPriority.HIGH,
                    source_service=event_data.get('source', 'audio_capture'),
                    confidence=event_data.get('confidence', 1.0),
                    audio_data=event_data.get('audio_data'),
                    context=event_data.get('context', {})
                )
                
                # Process interruption
                self._process_interruption(interruption)
                
        except Exception as e:
            logger.error(f"Error handling interruption detection: {e}")
    
    def _process_interruption(self, interruption: InterruptionEvent):
        """Process an interruption event with immediate response."""
        try:
            logger.info(f"Processing interruption: {interruption.interruption_type.name}")
            
            # Add to active interruptions
            self._active_interruptions[interruption.interruption_id] = interruption
            self._interruption_count += 1
            
            # Immediate actions based on priority
            if interruption.priority in [InterruptionPriority.HIGH, InterruptionPriority.CRITICAL]:
                # Stop current AI response generation
                if self._ai_conversation and hasattr(self._ai_conversation, 'stop_current_response'):
                    self._ai_conversation.stop_current_response()
                
                # Stop TTS playback immediately
                if self._audio_player and hasattr(self._audio_player, 'stop_playback'):
                    self._audio_player.stop_playback()
                
                # Update conversation state
                self._conversation_manager.handle_user_speech_start()
                
                # Publish interruption event
                self._event_bus.publish("INTERRUPTION_PROCESSED", {
                    'interruption': interruption.to_dict(),
                    'actions_taken': ['ai_response_stopped', 'tts_stopped', 'state_updated'],
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Interruption processed successfully: {interruption.interruption_id}")
                
            # Add to history
            self._interruption_history.append(interruption)
            
            # Limit history size
            if len(self._interruption_history) > 100:
                self._interruption_history = self._interruption_history[-50:]
                
        except Exception as e:
            logger.error(f"Error processing interruption: {e}")
            self._failed_recoveries += 1
    
    def _on_final_transcription(self, event_data: Dict[str, Any]):
        """Handle final transcription for correction and clarification detection."""
        try:
            final_text = event_data.get('text', '').strip()
            confidence = event_data.get('confidence', 1.0)
            
            if final_text:
                # Check for corrections
                if any(keyword in final_text.lower() for keyword in self._correction_keywords):
                    self._handle_correction(final_text, confidence)
                
                # Check for clarification requests
                elif any(keyword in final_text.lower() for keyword in self._clarification_keywords):
                    self._handle_clarification_request(final_text, confidence)
                
                # Regular interruption - continue conversation
                else:
                    self._handle_regular_interruption(final_text, confidence)
                
        except Exception as e:
            logger.error(f"Error handling final transcription: {e}")
    
    def _handle_correction(self, correction_text: str, confidence: float):
        """Handle user correction with context analysis."""
        try:
            logger.info(f"Processing correction: '{correction_text[:50]}...'")
            
            # Get recent conversation context
            recent_turns = self._conversation_manager.context.get_recent_context(3)
            
            # Find what might be corrected (last AI response)
            original_text = ""
            for turn in reversed(recent_turns):
                if turn.speaker == "assistant":
                    original_text = turn.content
                    break
            
            # Create correction event
            correction = CorrectionEvent(
                original_text=original_text,
                corrected_text=correction_text,
                confidence=confidence,
                context_reference=recent_turns[-1].turn_id if recent_turns else None
            )
            
            # Add to correction history
            self._correction_history.append(correction)
            self._correction_count += 1
            
            # Update conversation state with correction
            self._conversation_manager.handle_user_speech_complete(
                f"[CORRECTION] {correction_text}", confidence
            )
            
            # Publish correction event
            self._event_bus.publish("USER_CORRECTION_DETECTED", {
                'correction': correction.to_dict(),
                'conversation_context': self._conversation_manager.context.get_context_summary(),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Correction processed successfully")
            self._successful_recoveries += 1
            
        except Exception as e:
            logger.error(f"Error handling correction: {e}")
            self._failed_recoveries += 1
    
    def _handle_clarification_request(self, clarification_text: str, confidence: float):
        """Handle user clarification request."""
        try:
            logger.info(f"Processing clarification request: '{clarification_text[:50]}...'")
            
            # Update conversation state with clarification request
            self._conversation_manager.handle_user_speech_complete(
                f"[CLARIFICATION] {clarification_text}", confidence
            )
            
            # Publish clarification event
            self._event_bus.publish("CLARIFICATION_REQUEST", {
                'request_text': clarification_text,
                'confidence': confidence,
                'conversation_context': self._conversation_manager.context.get_context_summary(),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Clarification request processed successfully")
            self._successful_recoveries += 1
            
        except Exception as e:
            logger.error(f"Error handling clarification request: {e}")
            self._failed_recoveries += 1
    
    def _handle_regular_interruption(self, interruption_text: str, confidence: float):
        """Handle regular user interruption (new topic/continuation)."""
        try:
            logger.info(f"Processing regular interruption: '{interruption_text[:50]}...'")
            
            # Update conversation state with new user input
            self._conversation_manager.handle_user_speech_complete(interruption_text, confidence)
            
            # Publish regular interruption event
            self._event_bus.publish("USER_INTERRUPTION_COMPLETE", {
                'text': interruption_text,
                'confidence': confidence,
                'interruption_type': 'regular',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Regular interruption processed successfully")
            self._successful_recoveries += 1
            
        except Exception as e:
            logger.error(f"Error handling regular interruption: {e}")
            self._failed_recoveries += 1
    
    def force_interruption(self, reason: str = "Manual interruption"):
        """Force an interruption (for emergency stops or manual control)."""
        try:
            interruption = InterruptionEvent(
                interruption_type=InterruptionType.EMERGENCY_STOP,
                priority=InterruptionPriority.CRITICAL,
                source_service="manual",
                confidence=1.0,
                context={'reason': reason}
            )
            
            self._process_interruption(interruption)
            logger.info(f"Forced interruption executed: {reason}")
            
        except Exception as e:
            logger.error(f"Error forcing interruption: {e}")
    
    def get_interruption_statistics(self) -> Dict[str, Any]:
        """Get interruption handling statistics."""
        with self._interruption_lock:
            total_attempts = self._successful_recoveries + self._failed_recoveries
            success_rate = (self._successful_recoveries / total_attempts * 100) if total_attempts > 0 else 0
            
            return {
                'total_interruptions': self._interruption_count,
                'total_corrections': self._correction_count,
                'successful_recoveries': self._successful_recoveries,
                'failed_recoveries': self._failed_recoveries,
                'success_rate_percent': round(success_rate, 2),
                'active_interruptions': len(self._active_interruptions),
                'recent_interruptions': len([i for i in self._interruption_history 
                                           if (datetime.now() - i.timestamp).seconds < 300])  # Last 5 minutes
            }
