"""
Base interface for speech recognition strategies.

This module defines the Strategy pattern interface that all STT implementations
must follow, ensuring consistent behavior across different providers.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ...audio import AudioData


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class TranscriptionSegment:
    """Individual segment within a transcription."""
    text: str
    start_time: float
    end_time: float
    confidence: float


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
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'segments': [
                {
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence
                }
                for seg in self.segments
            ],
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'audio_id': self.audio_id
        }


@dataclass
class PartialResult:
    """Partial transcription result for streaming."""
    partial_text: str        # Current partial transcription
    confidence: float        # Confidence for this partial result
    is_final: bool          # Whether this is the final result for this segment
    segment_id: str         # Unique identifier for this segment
    timestamp: float        # Timestamp when this partial was generated


@dataclass
class LanguageAlternative:
    """Alternative language detection result."""
    language: str
    confidence: float


@dataclass
class LanguageResult:
    """Language detection result."""
    language: str                           # Primary detected language code
    confidence: float                       # Confidence in language detection
    alternatives: List[LanguageAlternative] # Alternative language candidates
    detection_method: str                   # Method used for detection
    
    def is_confident(self) -> bool:
        """Check if language detection is confident."""
        return self.confidence >= 0.7


@dataclass
class STTCapabilities:
    """STT implementation capabilities."""
    supports_streaming: bool              # Supports real-time streaming
    supports_language_detection: bool     # Can detect languages automatically
    supported_languages: List[str]        # List of supported language codes
    max_audio_length: int                 # Maximum audio length in seconds
    supported_formats: List[AudioFormat]  # Supported audio formats
    real_time_factor: float              # Processing speed vs real-time (1.0 = real-time)
    
    def is_compatible_with(self, requirements: dict) -> bool:
        """Check compatibility with requirements."""
        if 'max_length' in requirements:
            if requirements['max_length'] > self.max_audio_length:
                return False
        
        if 'language' in requirements:
            if requirements['language'] not in self.supported_languages:
                return False
        
        if 'format' in requirements:
            if requirements['format'] not in self.supported_formats:
                return False
        
        return True


@dataclass
class STTStatistics:
    """Service performance statistics."""
    total_transcriptions: int      # Total number of transcriptions processed
    average_processing_time: float # Average processing time in seconds
    accuracy_score: float          # Estimated accuracy score
    language_distribution: dict    # Distribution of detected languages
    error_count: int              # Number of processing errors
    uptime: float                 # Service uptime in seconds
    
    def get_performance_metrics(self) -> dict:
        """Get formatted performance metrics."""
        return {
            'total_transcriptions': self.total_transcriptions,
            'average_processing_time_ms': self.average_processing_time * 1000,
            'accuracy_percentage': self.accuracy_score * 100,
            'error_rate_percentage': (self.error_count / max(self.total_transcriptions, 1)) * 100,
            'uptime_hours': self.uptime / 3600,
            'language_distribution': self.language_distribution
        }


class ISpeechRecognition(ABC):
    """Strategy interface for speech-to-text implementations."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the STT strategy with configured parameters.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            InitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def transcribe_audio(self, audio_data: AudioData) -> TranscriptionResult:
        """
        Transcribe a complete audio segment to text.
        
        Args:
            audio_data: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Complete transcription with metadata
            
        Raises:
            TranscriptionError: If transcription fails
        """
        pass
    
    @abstractmethod
    def transcribe_stream(self, audio_stream: Iterator[AudioData]) -> Iterator[PartialResult]:
        """
        Transcribe streaming audio with partial results.
        
        Args:
            audio_stream: Iterator of audio chunks
            
        Yields:
            PartialResult: Partial transcription results
            
        Raises:
            TranscriptionError: If streaming transcription fails
        """
        pass
    
    @abstractmethod
    def detect_language(self, audio_data: AudioData) -> LanguageResult:
        """
        Detect the language of the audio segment.
        
        Args:
            audio_data: Audio data for language detection
            
        Returns:
            LanguageResult: Detected language with confidence
            
        Raises:
            LanguageDetectionError: If language detection fails
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> STTCapabilities:
        """
        Get the capabilities of this STT implementation.
        
        Returns:
            STTCapabilities: Implementation capabilities and limitations
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean up resources and shutdown the STT strategy.
        
        This method should properly dispose of models, close connections,
        and free any allocated resources.
        """
        pass
