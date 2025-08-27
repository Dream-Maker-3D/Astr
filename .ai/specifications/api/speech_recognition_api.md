# Speech Recognition Service API Specification

## Overview

The Speech Recognition Service provides real-time speech-to-text transcription capabilities using the Strategy pattern to support multiple STT implementations (Whisper, FastWhisper, Cloud services). It integrates with the Event Bus for loose coupling and supports streaming transcription for natural conversation flow.

## Core Interfaces

### ISpeechRecognition (Strategy Interface)

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from dataclasses import dataclass

class ISpeechRecognition(ABC):
    """Strategy interface for speech-to-text implementations."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the STT strategy with configured parameters."""
        pass
    
    @abstractmethod
    def transcribe_audio(self, audio_data: 'AudioData') -> 'TranscriptionResult':
        """Transcribe a complete audio segment to text."""
        pass
    
    @abstractmethod
    def transcribe_stream(self, audio_stream: Iterator['AudioData']) -> Iterator['PartialResult']:
        """Transcribe streaming audio with partial results."""
        pass
    
    @abstractmethod
    def detect_language(self, audio_data: 'AudioData') -> 'LanguageResult':
        """Detect the language of the audio segment."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> 'STTCapabilities':
        """Get the capabilities of this STT implementation."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and shutdown the STT strategy."""
        pass
```

### SpeechRecognitionService (Main Service)

```python
class SpeechRecognitionService:
    """Main speech recognition service using Strategy pattern."""
    
    def __init__(self, event_bus: EventBusService, config: SpeechConfig):
        """Initialize with event bus and configuration."""
        pass
    
    def initialize(self) -> bool:
        """Initialize the service and load the configured STT strategy."""
        pass
    
    def set_strategy(self, strategy: ISpeechRecognition) -> None:
        """Set the STT strategy implementation."""
        pass
    
    def process_audio(self, audio_data: AudioData) -> TranscriptionResult:
        """Process a single audio segment for transcription."""
        pass
    
    def start_streaming(self, audio_stream: Iterator[AudioData]) -> None:
        """Start streaming transcription mode."""
        pass
    
    def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        pass
    
    def get_statistics(self) -> STTStatistics:
        """Get service performance statistics."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the service and clean up resources."""
        pass
```

## Data Structures

### TranscriptionResult

```python
@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    text: str                           # Final transcribed text
    confidence: float                   # Overall confidence score (0.0-1.0)
    language: str                       # Detected/configured language code
    segments: List[TranscriptionSegment] # Word/phrase segments with timestamps
    processing_time: float              # Time taken to process (seconds)
    metadata: dict                      # Additional processing metadata
    audio_id: str                       # Unique identifier for the audio
    
    def is_high_confidence(self) -> bool:
        """Check if transcription meets confidence threshold."""
        return self.confidence >= 0.8
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        pass
```

### PartialResult

```python
@dataclass
class PartialResult:
    """Partial transcription result for streaming."""
    partial_text: str        # Current partial transcription
    confidence: float        # Confidence for this partial result
    is_final: bool          # Whether this is the final result for this segment
    segment_id: str         # Unique identifier for this segment
    timestamp: float        # Timestamp when this partial was generated
```

### STTCapabilities

```python
@dataclass
class STTCapabilities:
    """STT implementation capabilities."""
    supports_streaming: bool              # Supports real-time streaming
    supports_language_detection: bool     # Can detect languages automatically
    supported_languages: List[str]        # List of supported language codes
    max_audio_length: int                 # Maximum audio length in seconds
    supported_formats: List[AudioFormat]  # Supported audio formats
    real_time_factor: float              # Processing speed vs real-time (1.0 = real-time)
```

## Event Types

### Published Events

```python
class STTEventTypes:
    """STT-specific event types."""
    
    TRANSCRIPTION_STARTED = "TRANSCRIPTION_STARTED"
    TRANSCRIPTION_PARTIAL = "TRANSCRIPTION_PARTIAL"
    TRANSCRIPTION_COMPLETED = "TRANSCRIPTION_COMPLETED"
    TRANSCRIPTION_ERROR = "TRANSCRIPTION_ERROR"
    LANGUAGE_DETECTED = "LANGUAGE_DETECTED"
    STT_PERFORMANCE_WARNING = "STT_PERFORMANCE_WARNING"
```

## Configuration Schema

```python
@dataclass
class SpeechConfig:
    """Speech recognition configuration."""
    
    # Model configuration
    model_size: str = "base"              # Whisper model size
    device: str = "cpu"                   # Processing device
    language: str = "auto"                # Target language
    
    # Processing settings
    enable_streaming: bool = True         # Enable streaming transcription
    confidence_threshold: float = 0.7     # Minimum confidence for results
    max_segment_length: int = 30          # Maximum audio segment length
    
    # Enhancement settings
    enable_punctuation: bool = True       # Add punctuation to results
    enable_context_awareness: bool = True  # Use conversation context
```
