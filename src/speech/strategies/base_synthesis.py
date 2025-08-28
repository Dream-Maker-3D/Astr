"""
Base interface and data structures for speech synthesis.

This module defines the Strategy pattern interface for TTS providers and all
related data classes for speech synthesis functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Synthesis priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class VoiceParameters:
    """Voice synthesis parameters."""
    speaking_rate: float = 1.0  # 0.5 to 2.0
    pitch: float = 0.0  # -1.0 to 1.0
    volume: float = 1.0  # 0.0 to 1.0
    naturalness: str = "high"  # "low", "medium", "high"
    emotion: str = "neutral"  # "neutral", "happy", "sad", "excited"
    emphasis: List[str] = field(default_factory=list)  # Words to emphasize
    
    def validate(self) -> bool:
        """Validate parameter ranges."""
        return (0.5 <= self.speaking_rate <= 2.0 and
                -1.0 <= self.pitch <= 1.0 and
                0.0 <= self.volume <= 1.0 and
                self.naturalness in ["low", "medium", "high"] and
                self.emotion in ["neutral", "happy", "sad", "excited"])
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'speaking_rate': self.speaking_rate,
            'pitch': self.pitch,
            'volume': self.volume,
            'naturalness': self.naturalness,
            'emotion': self.emotion,
            'emphasis': self.emphasis
        }


@dataclass
class SynthesisMetadata:
    """Metadata about synthesis process."""
    model_name: str
    voice_characteristics: Dict[str, Any]
    processing_device: str  # "cpu", "gpu", "cuda:0"
    quality_metrics: Dict[str, float]
    generation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesisRequest:
    """Request for text-to-speech synthesis."""
    text: str
    voice_id: str
    priority: Priority
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Optional[VoiceParameters] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'voice_id': self.voice_id,
            'priority': self.priority.value,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters.to_dict() if self.parameters else None
        }


@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis."""
    audio_data: bytes
    sample_rate: int
    duration: float
    voice_id: str
    synthesis_time: float
    metadata: SynthesisMetadata
    request_id: str
    
    def save_to_file(self, path: str) -> None:
        """Save audio data to file."""
        try:
            import soundfile as sf
            import numpy as np
            
            # Convert bytes to numpy array based on sample rate
            audio_array = np.frombuffer(self.audio_data, dtype=np.float32)
            sf.write(path, audio_array, self.sample_rate)
            logger.info(f"Audio saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save audio to {path}: {e}")
            raise
    
    def to_audio_clip(self) -> 'AudioClip':
        """Convert to AudioClip for playback."""
        try:
            from src.audio.player_service import AudioClip, PlaybackPriority
            import numpy as np
            import uuid
            import time
            
            return AudioClip(
                clip_id=str(uuid.uuid4()),
                data=self.audio_data,  # Pass raw bytes directly
                sample_rate=self.sample_rate,
                channels=1,  # Assuming mono audio
                priority=PlaybackPriority.NORMAL,
                duration_seconds=self.duration,
                timestamp=time.time(),
                metadata={'voice_id': self.voice_id, 'synthesis_time': self.synthesis_time}
            )
        except ImportError:
            # Fallback if AudioClip not available
            logger.warning("AudioClip not available, returning raw data")
            return {
                'data': self.audio_data,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'metadata': {'voice_id': self.voice_id, 'synthesis_time': self.synthesis_time}
            }


@dataclass
class AudioChunk:
    """Individual audio chunk for streaming synthesis."""
    chunk_data: bytes
    chunk_id: str
    is_final: bool
    timestamp: float
    duration: float
    
    def merge_with(self, other: 'AudioChunk') -> 'AudioChunk':
        """Merge with another audio chunk."""
        return AudioChunk(
            chunk_data=self.chunk_data + other.chunk_data,
            chunk_id=f"{self.chunk_id}+{other.chunk_id}",
            is_final=other.is_final,
            timestamp=min(self.timestamp, other.timestamp),
            duration=self.duration + other.duration
        )


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    voice_id: str
    name: str
    gender: str
    age_group: str
    language: str
    style: str
    sample_rate: int
    is_cloned: bool = False
    
    def is_compatible_with(self, requirements: dict) -> bool:
        """Check if voice meets requirements."""
        for key, value in requirements.items():
            if hasattr(self, key) and getattr(self, key) != value:
                return False
        return True


@dataclass
class TTSCapabilities:
    """TTS provider capabilities."""
    supports_streaming: bool
    supports_voice_cloning: bool
    supported_languages: List[str]
    max_text_length: int
    supported_formats: List[AudioFormat]
    real_time_factor: float  # Speed multiplier (1.0 = real-time)
    voice_count: int
    
    def is_compatible_with(self, requirements: Dict[str, Any]) -> bool:
        """Check if capabilities meet requirements."""
        if requirements.get('streaming_required', False) and not self.supports_streaming:
            return False
        if requirements.get('voice_cloning_required', False) and not self.supports_voice_cloning:
            return False
        if 'language' in requirements and requirements['language'] not in self.supported_languages:
            return False
        if 'max_text_length' in requirements and requirements['max_text_length'] > self.max_text_length:
            return False
        return True


@dataclass
class TTSStatistics:
    """TTS service performance statistics."""
    total_syntheses: int = 0
    average_synthesis_time: float = 0.0
    total_audio_generated: float = 0.0  # Total seconds of audio
    voice_usage_distribution: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    uptime: float = 0.0  # Service uptime in seconds
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics summary."""
        return {
            'total_syntheses': self.total_syntheses,
            'average_synthesis_time': self.average_synthesis_time,
            'total_audio_generated': self.total_audio_generated,
            'error_rate': self.error_count / max(self.total_syntheses, 1),
            'uptime': self.uptime,
            'most_used_voice': max(self.voice_usage_distribution.items(), 
                                 key=lambda x: x[1], default=("none", 0))[0]
        }


@dataclass
class SpeechConfig:
    """Configuration for speech synthesis."""
    
    # TTS Model Configuration
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_path: Optional[str] = None
    device: str = "cpu"  # "cpu", "gpu", "cuda:0"
    
    # Voice Configuration
    default_voice_id: str = "female_young"
    voice_samples_path: str = "./voices/"
    enable_voice_cloning: bool = True
    
    # Audio Configuration
    sample_rate: int = 22050
    audio_format: str = "wav"
    bit_depth: int = 16
    
    # Performance Configuration
    max_text_length: int = 1000
    synthesis_timeout: float = 30.0
    enable_streaming: bool = True
    chunk_size: int = 1024
    
    # Quality Configuration
    quality_level: str = "high"  # "low", "medium", "high"
    enable_quality_validation: bool = True
    min_quality_score: float = 0.7
    
    # Voice Parameters (defaults)
    default_speaking_rate: float = 1.0
    default_pitch: float = 0.0
    default_volume: float = 0.8
    default_naturalness: str = "high"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.sample_rate > 0 and
                self.synthesis_timeout > 0 and
                self.chunk_size > 0 and
                0.0 <= self.min_quality_score <= 1.0 and
                self.quality_level in ["low", "medium", "high"])


class TTSEventTypes:
    """Event types published by Speech Synthesis Service."""
    
    # Core synthesis events
    SYNTHESIS_STARTED = "tts.synthesis.started"
    SYNTHESIS_COMPLETED = "tts.synthesis.completed"
    SYNTHESIS_FAILED = "tts.synthesis.failed"
    SYNTHESIS_INTERRUPTED = "tts.synthesis.interrupted"
    
    # Streaming events
    SYNTHESIS_CHUNK_READY = "tts.synthesis.chunk_ready"
    SYNTHESIS_STREAM_STARTED = "tts.synthesis.stream_started"
    SYNTHESIS_STREAM_ENDED = "tts.synthesis.stream_ended"
    
    # Voice management events
    VOICE_CHANGED = "tts.voice.changed"
    VOICE_CLONED = "tts.voice.cloned"
    VOICE_PARAMETERS_CHANGED = "tts.voice.parameters_changed"
    
    # Performance events
    TTS_PERFORMANCE_WARNING = "tts.performance.warning"
    TTS_QUALITY_ALERT = "tts.quality.alert"
    
    # Service lifecycle events
    TTS_SERVICE_INITIALIZED = "tts.service.initialized"
    TTS_SERVICE_SHUTDOWN = "tts.service.shutdown"
    TTS_MODEL_LOADED = "tts.model.loaded"
    TTS_MODEL_UNLOADED = "tts.model.unloaded"


class ISpeechSynthesis(ABC):
    """
    Abstract base class defining the Strategy interface for speech synthesis providers.
    Allows swapping between different TTS implementations (Coqui, FastSpeech, Cloud providers).
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the TTS provider and load required models.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def synthesize_text(self, text: str, voice_id: str) -> SynthesisResult:
        """
        Synthesize text to speech using the specified voice.
        
        Args:
            text (str): Text to synthesize
            voice_id (str): Voice identifier to use for synthesis
            
        Returns:
            SynthesisResult: Complete synthesis result with audio data
        """
        pass
    
    @abstractmethod
    def synthesize_stream(self, text_stream: Iterator[str], voice_id: str) -> Iterator[AudioChunk]:
        """
        Synthesize streaming text to audio chunks for real-time playback.
        
        Args:
            text_stream (Iterator[str]): Stream of text chunks to synthesize
            voice_id (str): Voice identifier to use for synthesis
            
        Yields:
            AudioChunk: Individual audio chunks as they become available
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[VoiceInfo]:
        """
        Get list of available voices for synthesis.
        
        Returns:
            List[VoiceInfo]: Available voice information
        """
        pass
    
    @abstractmethod
    def set_voice_parameters(self, params: VoiceParameters) -> None:
        """
        Configure voice synthesis parameters.
        
        Args:
            params (VoiceParameters): Voice parameter configuration
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> TTSCapabilities:
        """
        Get TTS provider capabilities and limitations.
        
        Returns:
            TTSCapabilities: Provider capability information
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean shutdown of TTS provider, releasing resources.
        """
        pass


# TTS-specific exceptions
class SynthesisError(Exception):
    """Base exception for synthesis errors."""
    pass


class ModelLoadError(SynthesisError):
    """Raised when TTS model fails to load."""
    pass


class VoiceNotFoundError(SynthesisError):
    """Raised when requested voice is not available."""
    pass


class SynthesisTimeoutError(SynthesisError):
    """Raised when synthesis takes too long."""
    pass


class AudioGenerationError(SynthesisError):
    """Raised when audio generation fails."""
    pass


class VoiceCloningError(SynthesisError):
    """Raised when voice cloning fails."""
    pass
