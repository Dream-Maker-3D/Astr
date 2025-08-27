"""
Speech processing services for the Astir Voice Assistant.

This package contains speech-to-text (STT) and text-to-speech (TTS) services
that provide natural language processing capabilities for voice interactions.
"""

from .recognition_service import (
    SpeechRecognitionService,
    TranscriptionResult,
    PartialResult,
    LanguageResult,
    STTCapabilities,
    STTStatistics
)

from .strategies import (
    ISpeechRecognition,
    WhisperSTTStrategy
)

__all__ = [
    'SpeechRecognitionService',
    'TranscriptionResult',
    'PartialResult', 
    'LanguageResult',
    'STTCapabilities',
    'STTStatistics',
    'ISpeechRecognition',
    'WhisperSTTStrategy'
]
