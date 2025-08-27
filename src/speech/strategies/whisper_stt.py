"""
Whisper STT Strategy - OpenAI Whisper implementation.

This module implements the WhisperSTTStrategy using OpenAI's Whisper models
for speech-to-text transcription, following the Strategy pattern.
"""

import logging
import time
import uuid
import numpy as np
from typing import Iterator, List, Optional, Dict, Any
from datetime import datetime

from ...audio import AudioData
from ...core import SpeechConfig
from ...utils.exceptions import (
    InitializationError,
    TranscriptionError,
    LanguageDetectionError
)

from .base import (
    ISpeechRecognition,
    TranscriptionResult,
    PartialResult,
    LanguageResult,
    LanguageAlternative,
    STTCapabilities,
    TranscriptionSegment,
    AudioFormat
)

# Try to import Whisper - graceful fallback if not available
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None


class WhisperSTTStrategy(ISpeechRecognition):
    """
    Whisper STT Strategy implementation using OpenAI Whisper models.
    
    Provides high-quality speech-to-text transcription with support for
    multiple languages, streaming, and confidence scoring.
    """
    
    def __init__(self, config: SpeechConfig):
        """Initialize the Whisper STT strategy."""
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Model configuration
        self._model_size = getattr(config, 'model_size', 'base')
        self._device = getattr(config, 'device', 'cpu')
        self._language = getattr(config, 'recognition_language', 'en')
        
        # Processing settings
        self._confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        self._enable_streaming = getattr(config, 'enable_streaming', True)
        self._max_segment_length = getattr(config, 'max_segment_length', 30)
        
        # Model and processor
        self._model = None
        self._model_loaded = False
        self._is_initialized = False
        
        # Performance tracking
        self._processing_times = []
        self._transcription_count = 0
        
        # Supported languages (Whisper supports 99 languages)
        self._supported_languages = [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl',
            'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro'
        ]
    
    def initialize(self) -> bool:
        """Initialize the Whisper STT strategy."""
        if self._is_initialized:
            self._logger.warning("Whisper STT strategy already initialized")
            return True
        
        try:
            if WHISPER_AVAILABLE:
                # Load actual Whisper model
                self._logger.info(f"Loading Whisper {self._model_size} model...")
                self._model = whisper.load_model(self._model_size)
                self._logger.info(f"Whisper {self._model_size} model loaded successfully")
                
                # Validate model capabilities
                if hasattr(self._model, 'dims'):
                    self._logger.info(f"Model dimensions: {self._model.dims}")
                
                self._is_initialized = True
                self._logger.info(f"Whisper STT strategy initialized with {self._model_size} model")
            else:
                # Fallback to mock mode if Whisper not available
                self._logger.warning("Whisper not available, using mock implementation")
                self._model = None
                self._is_initialized = True
                self._logger.info(f"Whisper STT strategy initialized (mock mode) with {self._model_size} model")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Whisper STT strategy: {e}")
            raise InitializationError(f"Whisper STT initialization failed: {e}")
    
    def transcribe_audio(self, audio_data: AudioData) -> TranscriptionResult:
        """Transcribe a complete audio segment to text."""
        if not self._is_initialized:
            raise TranscriptionError("Whisper STT strategy not initialized")
        
        start_time = time.time()
        audio_id = str(uuid.uuid4())
        
        try:
            if WHISPER_AVAILABLE and self._model is not None:
                # Real Whisper transcription
                self._logger.debug(f"Transcribing audio with Whisper {self._model_size}")
                
                # Convert audio data to format expected by Whisper
                if hasattr(audio_data, 'data'):
                    # AudioData object with data attribute
                    audio_array = np.array(audio_data.data, dtype=np.float32)
                elif isinstance(audio_data, bytes):
                    # Convert bytes to numpy array (assuming 16-bit PCM)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    # Assume it's already a numpy array or list
                    audio_array = np.array(audio_data, dtype=np.float32)
                
                # Transcribe with Whisper
                result = self._model.transcribe(
                    audio_array,
                    language=self._language if self._language != 'auto' else None,
                    task='transcribe',
                    verbose=False
                )
                
                # Extract transcription results
                text = result['text'].strip()
                language = result.get('language', 'en')
                
                # Create segments from Whisper segments
                segments = []
                for seg in result.get('segments', []):
                    segments.append(TranscriptionSegment(
                        text=seg['text'].strip(),
                        start_time=seg['start'],
                        end_time=seg['end'],
                        confidence=max(0.0, min(1.0, (seg.get('avg_logprob', -1.0) + 1.0) / 2.0))
                    ))
                
                # Calculate overall confidence from segments
                if segments:
                    confidence = sum(seg.confidence for seg in segments) / len(segments)
                else:
                    confidence = 0.8  # Default confidence for successful transcription
                
                self._logger.debug(f"Whisper transcription: '{text}' (confidence: {confidence:.3f})")
                
            else:
                # Fallback to mock transcription
                text = "Mock transcription result"
                confidence = 0.95
                language = self._language if self._language != 'auto' else 'en'
                segments = []
            
            processing_time = time.time() - start_time
            
            # Update performance tracking
            self._processing_times.append(processing_time)
            self._transcription_count += 1
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=language,
                segments=segments,
                processing_time=processing_time,
                metadata={'model': self._model_size, 'real_whisper': WHISPER_AVAILABLE and self._model is not None},
                audio_id=audio_id
            )
            
        except Exception as e:
            self._logger.error(f"Transcription failed for audio {audio_id}: {e}")
            raise TranscriptionError(f"Whisper transcription failed: {e}")
    
    def transcribe_stream(self, audio_stream: Iterator[AudioData]) -> Iterator[PartialResult]:
        """Transcribe streaming audio with partial results."""
        if not self._is_initialized:
            raise TranscriptionError("Whisper STT strategy not initialized")
        
        if not self._enable_streaming:
            raise TranscriptionError("Streaming not enabled in configuration")
        
        try:
            segment_id = str(uuid.uuid4())
            chunk_count = 0
            
            for audio_chunk in audio_stream:
                chunk_count += 1
                
                # Mock partial results for now
                # TODO: Implement actual streaming transcription
                if chunk_count % 3 == 0:
                    yield PartialResult(
                        partial_text=f"Mock partial {chunk_count//3}",
                        confidence=0.8,
                        is_final=False,
                        segment_id=segment_id,
                        timestamp=time.time()
                    )
            
            # Final result
            yield PartialResult(
                partial_text="Mock final transcription",
                confidence=0.95,
                is_final=True,
                segment_id=segment_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            self._logger.error(f"Streaming transcription failed: {e}")
            raise TranscriptionError(f"Streaming transcription failed: {e}")
    
    def detect_language(self, audio_data: AudioData) -> LanguageResult:
        """Detect the language of the audio segment."""
        if not self._is_initialized:
            raise LanguageDetectionError("Whisper STT strategy not initialized")
        
        try:
            if WHISPER_AVAILABLE and self._model is not None:
                # Real Whisper language detection
                self._logger.debug("Detecting language with Whisper")
                
                # Convert audio data to format expected by Whisper
                if hasattr(audio_data, 'data'):
                    audio_array = np.array(audio_data.data, dtype=np.float32)
                elif isinstance(audio_data, bytes):
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_array = np.array(audio_data, dtype=np.float32)
                
                # Use Whisper's language detection
                # Load audio and pad/trim it to fit 30 seconds
                audio_array = whisper.pad_or_trim(audio_array)
                
                # Make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio_array).to(self._model.device)
                
                # Detect the spoken language
                _, probs = self._model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                confidence = probs[detected_language]
                
                # Create alternatives from top languages
                alternatives = []
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for lang, prob in sorted_probs[1:4]:  # Top 3 alternatives (excluding the detected one)
                    alternatives.append(LanguageAlternative(language=lang, confidence=prob))
                
                self._logger.debug(f"Detected language: {detected_language} (confidence: {confidence:.3f})")
                
                return LanguageResult(
                    language=detected_language,
                    confidence=confidence,
                    alternatives=alternatives,
                    detection_method="whisper_real"
                )
            else:
                # Fallback to mock language detection
                detected_language = self._language if self._language != 'auto' else 'en'
                confidence = 0.9
                
                alternatives = [
                    LanguageAlternative(language='es', confidence=0.1),
                    LanguageAlternative(language='fr', confidence=0.05)
                ]
                
                return LanguageResult(
                    language=detected_language,
                    confidence=confidence,
                    alternatives=alternatives,
                    detection_method="whisper_mock"
                )
            
        except Exception as e:
            self._logger.error(f"Language detection failed: {e}")
            raise LanguageDetectionError(f"Language detection failed: {e}")
    
    def get_capabilities(self) -> STTCapabilities:
        """Get the capabilities of this STT implementation."""
        return STTCapabilities(
            supports_streaming=self._enable_streaming,
            supports_language_detection=True,
            supported_languages=self._supported_languages,
            max_audio_length=self._max_segment_length,
            supported_formats=[AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC],
            real_time_factor=0.3 if self._model_size == 'base' else 0.5
        )
    
    def shutdown(self) -> None:
        """Clean up resources and shutdown the STT strategy."""
        if not self._is_initialized:
            return
        
        self._logger.info("Shutting down Whisper STT strategy...")
        
        # Clear model from memory
        if self._model is not None:
            del self._model
            self._model = None
        
        self._model_loaded = False
        self._is_initialized = False
        
        self._logger.info("Whisper STT strategy shutdown complete")
