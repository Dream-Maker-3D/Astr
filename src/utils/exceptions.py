"""
Custom exception hierarchy for the Astir Voice Assistant.

This module defines all custom exceptions used throughout the system,
following a hierarchical structure for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class AstirError(Exception):
    """Base exception for all Astir Voice Assistant errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Event Bus Exceptions
class EventBusError(AstirError):
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


# Configuration Exceptions
class ConfigurationError(AstirError):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class ConfigLoadError(ConfigurationError):
    """Raised when configuration loading fails."""
    pass


# Audio Exceptions
class AudioError(AstirError):
    """Base exception for audio-related errors."""
    pass


class AudioDeviceError(AudioError):
    """Raised when audio device operations fail."""
    pass


class AudioCaptureError(AudioError):
    """Raised when audio capture fails."""
    pass


class AudioPlaybackError(AudioError):
    """Raised when audio playback fails."""
    pass


# Speech Recognition Exceptions
class SpeechRecognitionError(AstirError):
    """Base exception for speech recognition errors."""
    pass


class TranscriptionError(SpeechRecognitionError):
    """Raised when speech transcription fails."""
    pass


class ModelLoadError(SpeechRecognitionError):
    """Raised when STT model loading fails."""
    pass


# Speech Synthesis Exceptions
class SpeechSynthesisError(AstirError):
    """Base exception for speech synthesis errors."""
    pass


class TTSError(SpeechSynthesisError):
    """Raised when text-to-speech conversion fails."""
    pass


class VoiceLoadError(SpeechSynthesisError):
    """Raised when TTS voice loading fails."""
    pass


# AI Service Exceptions
class AIServiceError(AstirError):
    """Base exception for AI service errors."""
    pass


class NetworkError(AIServiceError):
    """Raised when network operations fail."""
    pass


class APIError(AIServiceError):
    """Raised when API calls fail."""
    pass


class RateLimitError(AIServiceError):
    """Raised when API rate limits are exceeded."""
    pass


# System Exceptions
class SystemError(AstirError):
    """Base exception for system-level errors."""
    pass


class InitializationError(SystemError):
    """Raised when system initialization fails."""
    pass


class ShutdownError(SystemError):
    """Raised when system shutdown fails."""
    pass
