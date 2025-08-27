"""
Utility modules for the Astir Voice Assistant.

This package contains utility functions, exception classes, and helper modules
used throughout the voice assistant system.
"""

from .exceptions import *

__all__ = [
    # Base exceptions
    'AstirError',
    
    # Event Bus exceptions
    'EventBusError',
    'EventValidationError', 
    'SubscriptionError',
    'PublishingError',
    
    # Configuration exceptions
    'ConfigurationError',
    'ConfigValidationError',
    'ConfigLoadError',
    
    # Audio exceptions
    'AudioError',
    'AudioDeviceError',
    'AudioCaptureError',
    'AudioPlaybackError',
    
    # Speech recognition exceptions
    'SpeechRecognitionError',
    'TranscriptionError',
    'ModelLoadError',
    
    # Speech synthesis exceptions
    'SpeechSynthesisError',
    'TTSError',
    'VoiceLoadError',
    
    # AI service exceptions
    'AIServiceError',
    'NetworkError',
    'APIError',
    'RateLimitError',
    
    # System exceptions
    'SystemError',
    'InitializationError',
    'ShutdownError'
]
