# Event Bus Service - API Specification

## Overview
The Event Bus Service implements the Observer pattern to provide decoupled, event-driven communication between system components. This specification defines the complete API interface, data structures, and usage patterns.

## Core Interface

### EventBusService Class

```python
class EventBusService:
    """
    Event-driven communication service implementing the Observer pattern.
    Provides synchronous and asynchronous event publishing with type-safe handlers.
    """
    
    def __init__(self, max_queue_size: int = 1000, worker_threads: int = 4):
        """
        Initialize the Event Bus Service.
        
        Args:
            max_queue_size: Maximum number of queued async events
            worker_threads: Number of worker threads for async processing
        """
    
    def initialize(self) -> bool:
        """
        Initialize the event bus and start worker threads.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            EventBusError: If initialization fails
        """
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the event bus.
        Processes all pending events before stopping.
        """
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: Callable that processes the event data
            
        Returns:
            str: Subscription ID for later unsubscription
            
        Raises:
            ValueError: If event_type is invalid or handler is not callable
        """
    
    def unsubscribe(self, event_type: str, subscription_id: str) -> bool:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: The event type to unsubscribe from
            subscription_id: The subscription ID returned by subscribe()
            
        Returns:
            bool: True if unsubscribed successfully, False if not found
        """
    
    def publish(self, event_type: str, data: Dict[str, Any], 
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an event synchronously to all subscribers.
        
        Args:
            event_type: The type of event being published
            data: Event payload data
            metadata: Optional metadata (timestamp, source, etc.)
            
        Raises:
            EventValidationError: If event data is invalid
            EventBusError: If publishing fails
        """
    
    async def publish_async(self, event_type: str, data: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an event asynchronously to all subscribers.
        
        Args:
            event_type: The type of event being published
            data: Event payload data
            metadata: Optional metadata
            
        Raises:
            EventValidationError: If event data is invalid
            EventBusError: If publishing fails
        """
    
    def get_subscribers(self, event_type: str) -> List[str]:
        """
        Get list of subscription IDs for an event type.
        
        Args:
            event_type: The event type to query
            
        Returns:
            List[str]: List of subscription IDs
        """
    
    def get_statistics(self) -> EventBusStatistics:
        """
        Get event bus performance and usage statistics.
        
        Returns:
            EventBusStatistics: Current statistics
        """
```

## Event Types

### Core System Events
```python
class EventTypes:
    """Standard event types used throughout the system."""
    
    # System lifecycle events
    SYSTEM_READY = "SYSTEM_READY"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    
    # Audio events
    AUDIO_DATA_RECEIVED = "AUDIO_DATA_RECEIVED"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    SPEECH_ENDED = "SPEECH_ENDED"
    INTERRUPTION_DETECTED = "INTERRUPTION_DETECTED"
    AUDIO_DEVICE_ERROR = "AUDIO_DEVICE_ERROR"
    
    # Speech recognition events
    SPEECH_RECOGNIZED = "SPEECH_RECOGNIZED"
    LOW_CONFIDENCE_RECOGNITION = "LOW_CONFIDENCE_RECOGNITION"
    RECOGNITION_ERROR = "RECOGNITION_ERROR"
    
    # AI conversation events
    AI_RESPONSE_RECEIVED = "AI_RESPONSE_RECEIVED"
    AI_ERROR = "AI_ERROR"
    CONVERSATION_STARTED = "CONVERSATION_STARTED"
    CONVERSATION_ENDED = "CONVERSATION_ENDED"
    
    # TTS events
    TTS_AUDIO_GENERATED = "TTS_AUDIO_GENERATED"
    TTS_ERROR = "TTS_ERROR"
    
    # Playback events
    PLAYBACK_STARTED = "PLAYBACK_STARTED"
    PLAYBACK_FINISHED = "PLAYBACK_FINISHED"
    PLAYBACK_INTERRUPTED = "PLAYBACK_INTERRUPTED"
    PLAYBACK_ERROR = "PLAYBACK_ERROR"
    
    # Configuration events
    CONFIG_CHANGED = "CONFIG_CHANGED"
    CONFIG_ERROR = "CONFIG_ERROR"
```

## Data Structures

### Event Data Models
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class EventMetadata:
    """Standard metadata included with all events."""
    timestamp: datetime
    source_service: str
    event_id: str
    correlation_id: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=critical

@dataclass
class Event:
    """Complete event structure."""
    type: str
    data: Dict[str, Any]
    metadata: EventMetadata

# Specific event data structures
@dataclass
class AudioDataEvent:
    """Audio data received event payload."""
    audio_data: bytes
    sample_rate: int
    channels: int
    duration_ms: int
    format: str = "wav"

@dataclass
class SpeechDetectedEvent:
    """Speech detection event payload."""
    confidence: float
    start_time: datetime
    audio_level: float
    is_interruption: bool = False

@dataclass
class SpeechRecognizedEvent:
    """Speech recognition result payload."""
    text: str
    confidence: float
    language: str
    alternatives: List[Dict[str, Any]]
    processing_time_ms: int

@dataclass
class AIResponseEvent:
    """AI response event payload."""
    text: str
    model_used: str
    response_time_ms: int
    token_count: int
    confidence: Optional[float] = None

@dataclass
class TTSAudioEvent:
    """TTS audio generation event payload."""
    audio_data: bytes
    text: str
    voice_id: str
    duration_ms: int
    format: str = "wav"

@dataclass
class SystemErrorEvent:
    """System error event payload."""
    error_type: str
    error_message: str
    component: str
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
```

### Statistics and Monitoring
```python
@dataclass
class EventBusStatistics:
    """Event bus performance statistics."""
    total_events_published: int
    total_events_delivered: int
    events_per_second: float
    average_delivery_time_ms: float
    active_subscriptions: int
    queue_size: int
    error_count: int
    uptime_seconds: int
    memory_usage_mb: float
    
    # Per-event-type statistics
    event_type_stats: Dict[str, Dict[str, Any]]
```

## Usage Examples

### Basic Subscription and Publishing
```python
# Initialize event bus
event_bus = EventBusService()
event_bus.initialize()

# Subscribe to events
def handle_speech_recognized(event_data):
    print(f"Recognized: {event_data['text']}")
    print(f"Confidence: {event_data['confidence']}")

subscription_id = event_bus.subscribe(
    EventTypes.SPEECH_RECOGNIZED, 
    handle_speech_recognized
)

# Publish an event
event_bus.publish(
    EventTypes.SPEECH_RECOGNIZED,
    {
        "text": "Hello, how are you?",
        "confidence": 0.95,
        "language": "en",
        "alternatives": [],
        "processing_time_ms": 150
    }
)

# Unsubscribe when done
event_bus.unsubscribe(EventTypes.SPEECH_RECOGNIZED, subscription_id)
```

### Integration with Services
```python
class AudioCaptureService:
    """Example service integration."""
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Subscribe to relevant events."""
        self.event_bus.subscribe(
            EventTypes.SYSTEM_READY,
            self._handle_system_ready
        )
        self.event_bus.subscribe(
            EventTypes.INTERRUPTION_DETECTED,
            self._handle_interruption
        )
    
    def _handle_system_ready(self, event_data):
        """Start audio capture when system is ready."""
        self.start_capture()
    
    def _handle_interruption(self, event_data):
        """Handle speech interruption."""
        self.stop_current_playback()
    
    def on_audio_captured(self, audio_data):
        """Publish audio data event."""
        self.event_bus.publish(
            EventTypes.AUDIO_DATA_RECEIVED,
            {
                "audio_data": audio_data,
                "sample_rate": 16000,
                "channels": 1,
                "duration_ms": len(audio_data) // 32,  # 16kHz, 16-bit
                "format": "wav"
            }
        )
```

## Error Handling

### Exception Types
```python
class EventBusError(Exception):
    """Base exception for event bus errors."""
    pass

class EventValidationError(EventBusError):
    """Raised when event data validation fails."""
    pass

class SubscriptionError(EventBusError):
    """Raised when subscription operations fail."""
    pass

class PublishingError(EventBusError):
    """Raised when event publishing fails."""
    pass
```

## Testing Support

### Mock Event Bus
```python
class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.published_events = []
        self.subscribers = {}
    
    def publish(self, event_type: str, data: Dict[str, Any]):
        self.published_events.append((event_type, data))
    
    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        return f"mock_sub_{len(self.subscribers[event_type])}"
    
    def get_published_events(self, event_type: str = None):
        if event_type:
            return [data for etype, data in self.published_events if etype == event_type]
        return self.published_events
```

This API specification provides a complete interface definition for the Event Bus Service, enabling precise implementation and comprehensive testing following the BDD methodology.
