"""
Speech recognition strategy implementations.

This package contains the Strategy pattern implementations for different
speech-to-text providers (Whisper, FastWhisper, Cloud services).
"""

from .base import ISpeechRecognition
from .whisper_stt import WhisperSTTStrategy

__all__ = [
    'ISpeechRecognition',
    'WhisperSTTStrategy'
]
