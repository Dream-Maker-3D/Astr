# AI Conversation Service API Specification

## Overview
This document defines the API specification for the AI Conversation Service, implementing OpenRouter integration for natural AI-powered conversations with support for multiple models (Claude, GPT, Llama).

## Core Interface

### AIConversationService (Main Service)

```python
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

class AIConversationService:
    """
    Main AI Conversation Service for natural AI-powered conversations.
    
    Integrates with OpenRouter API to support multiple AI models and provides
    natural conversation management with context preservation and streaming.
    """
    
    def __init__(self, event_bus: EventBusService, config: AIConfig):
        """Initialize the AI Conversation Service."""
        pass
    
    def initialize(self) -> bool:
        """Initialize the AI Conversation Service."""
        pass
    
    def process_message(self, text: str, context: Optional[ConversationContext] = None) -> ConversationResponse:
        """Process a single message and return AI response."""
        pass
    
    def process_message_stream(self, text: str, context: Optional[ConversationContext] = None) -> Iterator[ResponseChunk]:
        """Process message with streaming response for real-time conversation."""
        pass
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different AI model."""
        pass
    
    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get current conversation history."""
        pass
    
    def clear_conversation(self) -> None:
        """Clear conversation history and reset context."""
        pass
    
    def get_statistics(self) -> AIStatistics:
        """Get AI service statistics."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the AI Conversation Service."""
        pass
```

### OpenRouterClient (API Client)

```python
class OpenRouterClient:
    """
    OpenRouter API client for multi-model AI integration.
    
    Provides unified interface for Claude, GPT, and Llama models
    through the OpenRouter service.
    """
    
    def initialize(self) -> bool:
        """Initialize the OpenRouter client and validate credentials."""
        pass
    
    def set_model(self, model_name: str) -> None:
        """Set the current AI model."""
        pass
    
    def generate_response(self, messages: List[Message], stream: bool = False) -> Union[str, Iterator[str]]:
        """Generate AI response from messages."""
        pass
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available AI models."""
        pass
    
    def validate_api_key(self) -> bool:
        """Validate OpenRouter API key."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the OpenRouter client."""
        pass
```

## Data Structures

### Core Data Classes

```python
@dataclass
class ConversationRequest:
    """Request for AI conversation processing."""
    text: str
    context: Optional[ConversationContext]
    priority: Priority
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    streaming: bool = False

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
        formatted_text = self.text.replace('**', '').replace('*', '')
        formatted_text = formatted_text.replace('. ', '... ')
        return formatted_text

@dataclass
class ResponseChunk:
    """Individual chunk of streaming AI response."""
    chunk_text: str
    chunk_id: str
    is_final: bool
    timestamp: float
    model_used: str

@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    speaker: Speaker
    message: str
    timestamp: datetime
    metadata: TurnMetadata
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
class AIStatistics:
    """AI service performance statistics."""
    total_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_used: int = 0
    model_usage_distribution: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    uptime: float = 0.0
    rate_limit_hits: int = 0
```

### Enumerations

```python
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
```

## Configuration Schema

### AIConfig

```python
@dataclass
class AIConfig:
    """Configuration for AI conversation service."""
    
    # OpenRouter Configuration
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3-5-sonnet"
    
    # Model Parameters
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window_limit: int = 200000
    
    # Streaming Configuration
    streaming_enabled: bool = True
    
    # Rate Limiting
    rate_limit_rpm: int = 60
    rate_limit_rph: int = 1000
    
    # Retry Configuration
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # System Prompt
    system_prompt: str = """You are a helpful AI assistant engaged in natural conversation. 
Keep responses concise and conversational. Use natural speech patterns with contractions. 
Avoid verbose explanations unless specifically asked."""
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.api_key and
                0.0 <= self.temperature <= 2.0 and
                self.max_tokens > 0 and
                self.timeout_seconds > 0)
```

## Event Types

### AIEventTypes

```python
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
```

## Usage Examples

### Basic Conversation

```python
# Initialize AI service
ai_service = AIConversationService(event_bus, config)
ai_service.initialize()

# Process a message
response = ai_service.process_message("What's the weather like?")
print(f"AI: {response.text}")

# Format for TTS
tts_text = response.format_for_tts()
```

### Streaming Conversation

```python
# Stream a response
for chunk in ai_service.process_message_stream("Tell me about AI"):
    print(f"Chunk: {chunk.chunk_text}")
    if chunk.is_final:
        print("Response complete")
```

### Event Handling

```python
# Subscribe to AI events
event_bus.subscribe(AIEventTypes.AI_RESPONSE_READY, handle_ai_response)

def handle_ai_response(event_data):
    text = event_data['text']
    # Send to TTS service
    tts_service.synthesize_text(text)
```

## Error Handling

### Exception Types

```python
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
```

## Performance Requirements

### Latency Targets
- **First Response Chunk**: < 500ms
- **Complete Response**: < 2000ms for typical queries
- **Context Processing**: < 100ms
- **Model Switching**: < 200ms

### Quality Metrics
- **Response Relevance**: > 90% contextually appropriate
- **Error Rate**: < 1% for valid inputs
- **Uptime**: > 99.5% availability
