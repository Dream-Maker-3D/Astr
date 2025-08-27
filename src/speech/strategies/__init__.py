"""
Speech processing strategy implementations.

This package contains the Strategy pattern implementations for different
speech-to-text and text-to-speech providers.
"""

# Speech Recognition Strategies
from .base import ISpeechRecognition
from .whisper_stt import WhisperSTTStrategy

# Speech Synthesis Strategies
from .base_synthesis import ISpeechSynthesis
from .coqui_tts import CoquiTTSStrategy

__all__ = [
    # STT Strategies
    'ISpeechRecognition',
    'WhisperSTTStrategy',
    
    # TTS Strategies
    'ISpeechSynthesis',
    'CoquiTTSStrategy'
]
