"""
Speech processing services for the Astir Voice Assistant.

This package contains speech-to-text (STT) and text-to-speech (TTS) services
that provide natural language processing capabilities for voice interactions.
"""

# Speech Recognition (STT)
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

# Speech Synthesis (TTS)
from .synthesis_service import SpeechSynthesisService
from .strategies.base_synthesis import (
    ISpeechSynthesis, SynthesisResult, SynthesisRequest, AudioChunk,
    VoiceInfo, VoiceParameters, TTSCapabilities, TTSStatistics,
    Priority, TTSEventTypes, SpeechConfig, AudioFormat,
    SynthesisMetadata
)
from .strategies.coqui_tts import CoquiTTSStrategy

__all__ = [
    # Speech Recognition
    'SpeechRecognitionService',
    'TranscriptionResult',
    'PartialResult', 
    'LanguageResult',
    'STTCapabilities',
    'STTStatistics',
    'ISpeechRecognition',
    'WhisperSTTStrategy',
    
    # Speech Synthesis
    'SpeechSynthesisService',
    'ISpeechSynthesis',
    'SynthesisResult',
    'SynthesisRequest',
    'AudioChunk',
    'VoiceInfo',
    'VoiceParameters',
    'TTSCapabilities',
    'TTSStatistics',
    'Priority',
    'TTSEventTypes',
    'SpeechConfig',
    'AudioFormat',
    'SynthesisMetadata',
    'CoquiTTSStrategy'
]
