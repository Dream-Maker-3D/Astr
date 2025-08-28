"""
Data types and structures for AI conversation functionality.

This module defines all data classes, enums, and configuration objects
used by the AI Conversation Service and OpenRouter integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class Speaker(Enum):
    """Speaker in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageRole(Enum):
    """Message role for AI processing."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationState(Enum):
    """State of a conversation."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class Priority(Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ConversationContext:
    """Context information for a conversation."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"
    session_start: datetime = field(default_factory=datetime.now)
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def update_context(self, data: dict) -> None:
        """Update context data."""
        self.context_data.update(data)
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        return f"Conversation {self.conversation_id} started at {self.session_start}"


@dataclass
class TurnMetadata:
    """Metadata for a conversation turn."""
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    token_count: int = 0


@dataclass
class ResponseMetadata:
    """Metadata about AI response generation."""
    model_name: str
    processing_device: str = "cloud"
    token_usage: Dict[str, int] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Message:
    """Message in OpenAI/OpenRouter format."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI API format."""
        return {
            'role': self.role.value,
            'content': self.content
        }


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker: Speaker
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: TurnMetadata = field(default_factory=TurnMetadata)
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_message(self) -> Message:
        """Convert to Message format for AI processing."""
        role_mapping = {
            Speaker.USER: MessageRole.USER,
            Speaker.ASSISTANT: MessageRole.ASSISTANT,
            Speaker.SYSTEM: MessageRole.SYSTEM
        }
        return Message(
            role=role_mapping[self.speaker],
            content=self.message,
            timestamp=self.timestamp
        )
    
    def get_token_count(self) -> int:
        """Estimate token count for this turn."""
        # Rough estimation: ~4 characters per token
        return len(self.message) // 4
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'turn_id': self.turn_id,
            'speaker': self.speaker.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': {
                'confidence': self.metadata.confidence,
                'processing_time': self.metadata.processing_time,
                'model_used': self.metadata.model_used,
                'token_count': self.metadata.token_count
            }
        }


@dataclass
class ConversationRequest:
    """Request for AI conversation processing."""
    text: str
    context: Optional[ConversationContext] = None
    priority: Priority = Priority.NORMAL
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    streaming: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'context': self.context.get_context_summary() if self.context else None,
            'priority': self.priority.value,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'streaming': self.streaming
        }


@dataclass
class ConversationResponse:
    """Response from AI conversation processing."""
    text: str
    model_used: str
    processing_time: float
    token_count: int
    confidence: float
    metadata: ResponseMetadata
    request_id: str
    
    def format_for_tts(self) -> str:
        """Format response text for speech synthesis."""
        try:
            # Remove markdown formatting
            formatted_text = self.text.replace('**', '').replace('*', '')
            formatted_text = formatted_text.replace('`', '')
            
            # Add natural pauses for better speech synthesis
            formatted_text = formatted_text.replace('. ', '... ')
            formatted_text = formatted_text.replace('? ', '?... ')
            formatted_text = formatted_text.replace('! ', '!... ')
            
            # Remove excessive whitespace
            formatted_text = ' '.join(formatted_text.split())
            
            logger.debug(f"Formatted text for TTS: '{formatted_text[:100]}...'")
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting text for TTS: {e}")
            return self.text  # Return original text as fallback


@dataclass
class ResponseChunk:
    """Individual chunk of streaming AI response."""
    chunk_text: str
    chunk_id: str
    is_final: bool
    timestamp: float
    model_used: str
    
    def merge_with(self, other: 'ResponseChunk') -> 'ResponseChunk':
        """Merge with another response chunk."""
        return ResponseChunk(
            chunk_text=self.chunk_text + other.chunk_text,
            chunk_id=f"{self.chunk_id}+{other.chunk_id}",
            is_final=other.is_final,
            timestamp=min(self.timestamp, other.timestamp),
            model_used=self.model_used
        )


@dataclass
class AIStatistics:
    """AI service performance statistics."""
    total_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_used: int = 0
    model_usage_distribution: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    uptime: float = 0.0
    rate_limit_hits: int = 0
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics summary."""
        return {
            'total_requests': self.total_requests,
            'average_response_time': self.average_response_time,
            'total_tokens_used': self.total_tokens_used,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'uptime': self.uptime,
            'rate_limit_hits': self.rate_limit_hits,
            'most_used_model': max(self.model_usage_distribution.items(), 
                                 key=lambda x: x[1], default=("none", 0))[0]
        }
    
    def get_cost_estimate(self) -> float:
        """Estimate total cost based on token usage."""
        # Rough estimation: $0.01 per 1000 tokens
        return self.total_tokens_used * 0.00001


@dataclass
class AIConfig:
    """Configuration for AI conversation service."""
    
    # OpenRouter Configuration
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3-5-sonnet"
    
    # Model Parameters
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window_limit: int = 200000
    
    # Streaming Configuration
    streaming_enabled: bool = True
    
    # Rate Limiting
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_rph: int = 1000  # Requests per hour
    
    # Retry Configuration
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # System Prompt
    system_prompt: str = """You are a helpful AI assistant engaged in natural conversation. 
Keep responses concise and conversational. Use natural speech patterns with contractions. 
Avoid verbose explanations unless specifically asked. Be ready to be interrupted naturally."""
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.api_key and
                0.0 <= self.temperature <= 2.0 and
                self.max_tokens > 0 and
                self.timeout_seconds > 0 and
                self.retry_attempts >= 0)
    
    def get_headers(self) -> dict:
        """Get HTTP headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Dream-Maker-3D/Astr",
            "X-Title": "Astir Voice Assistant"
        }


class AIEventTypes:
    """Event types published by AI Conversation Service."""
    
    # Core processing events
    AI_PROCESSING_STARTED = "ai.processing.started"
    AI_RESPONSE_CHUNK = "ai.response.chunk"
    AI_RESPONSE_READY = "ai.response.ready"
    AI_RESPONSE_INTERRUPTED = "ai.response.interrupted"
    
    # Model management events
    AI_MODEL_CHANGED = "ai.model.changed"
    AI_ERROR = "ai.error"
    AI_RATE_LIMIT_HIT = "ai.rate_limit.hit"
    
    # Service lifecycle events
    AI_SERVICE_INITIALIZED = "ai.service.initialized"
    AI_SERVICE_SHUTDOWN = "ai.service.shutdown"


# AI-specific exceptions
class AIConversationError(Exception):
    """Base exception for AI conversation errors."""
    pass


class OpenRouterAPIError(AIConversationError):
    """Raised when OpenRouter API calls fail."""
    pass


class ModelNotAvailableError(AIConversationError):
    """Raised when requested model is not available."""
    pass


class RateLimitExceededError(AIConversationError):
    """Raised when API rate limits are exceeded."""
    pass


class InvalidAPIKeyError(AIConversationError):
    """Raised when API key is invalid or expired."""
    pass


class ContextWindowExceededError(AIConversationError):
    """Raised when context window limit is exceeded."""
    pass
