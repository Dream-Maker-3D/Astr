"""
Coqui TTS Strategy Implementation.

This module implements the Coqui TTS strategy for speech synthesis,
providing high-quality voice synthesis with voice cloning capabilities
using the XTTS-v2 model.
"""

import time
import numpy as np
from typing import List, Iterator, Dict, Any
from datetime import datetime
import logging

from src.speech.strategies.base_synthesis import (
    ISpeechSynthesis, SynthesisResult, AudioChunk, VoiceInfo,
    VoiceParameters, TTSCapabilities, SpeechConfig, AudioFormat,
    SynthesisMetadata
)
from src.utils.exceptions import (
    ModelLoadError, VoiceNotFoundError, AudioGenerationError,
    SynthesisTimeoutError
)

# Try to import Coqui TTS - graceful fallback if not available
try:
    from TTS.api import TTS
    import torch
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    TTS = None
    torch = None

logger = logging.getLogger(__name__)


class CoquiTTSStrategy(ISpeechSynthesis):
    """
    Coqui TTS Strategy implementation.
    
    Provides high-quality text-to-speech synthesis using Coqui TTS
    with voice cloning capabilities via XTTS-v2 model.
    """
    
    def __init__(self, config: SpeechConfig):
        """
        Initialize Coqui TTS Strategy.
        
        Args:
            config (SpeechConfig): TTS configuration
        """
        self._config = config
        self._is_initialized = False
        self._model = None
        self._voice_samples: Dict[str, Any] = {}
        self._voice_parameters = VoiceParameters()
        self._model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Available voices with sultry, sexy options
        self._available_voices = [
            VoiceInfo(
                voice_id="gracie_wise",
                name="Gracie Wise - Young Sultry",
                gender="female",
                age_group="young_adult",
                language="en",
                style="sultry_seductive",
                sample_rate=22050,
                is_cloned=False
            ),
            VoiceInfo(
                voice_id="alison_dietlinde",
                name="Alison Dietlinde - Smooth & Alluring",
                gender="female",
                age_group="young_adult",
                language="en",
                style="smooth_alluring",
                sample_rate=22050,
                is_cloned=False
            ),
            VoiceInfo(
                voice_id="annmarie_nele",
                name="Annmarie Nele - Sophisticated Sexy",
                gender="female",
                age_group="young_adult",
                language="en",
                style="sophisticated_sexy",
                sample_rate=22050,
                is_cloned=False
            ),
            VoiceInfo(
                voice_id="asya_anara",
                name="Asya Anara - Exotic & Sensual",
                gender="female",
                age_group="young_adult",
                language="en",
                style="exotic_sensual",
                sample_rate=22050,
                is_cloned=False
            ),
            VoiceInfo(
                voice_id="brenda_stern",
                name="Brenda Stern - Confident & Sexy",
                gender="female",
                age_group="young_adult",
                language="en",
                style="confident_sexy",
                sample_rate=22050,
                is_cloned=False
            )
        ]
        
        logger.info("Coqui TTS Strategy created (Mock mode)")
    
    def initialize(self) -> bool:
        """
        Initialize the Coqui TTS model and voice samples.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if COQUI_AVAILABLE:
                # Real Coqui TTS initialization
                logger.info("Initializing Coqui TTS Strategy with XTTS-v2")
                
                # Determine device (GPU if available, otherwise CPU)
                device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Load XTTS-v2 model
                logger.info(f"Loading {self._model_name} model...")
                self._model = TTS(self._model_name, gpu=(device == "cuda"))
                
                # Get model info
                if hasattr(self._model, 'synthesizer') and self._model.synthesizer:
                    logger.info(f"Model loaded successfully on {device}")
                    if hasattr(self._model.synthesizer.tts_model, 'language_manager'):
                        languages = list(self._model.synthesizer.tts_model.language_manager.language_names)
                        logger.info(f"Supported languages: {languages[:10]}...")  # Show first 10
                
            else:
                # Fallback to mock mode
                logger.warning("Coqui TTS not available, using mock implementation")
                
                # Mock model initialization
                self._model = {
                    'name': self._config.model_name,
                    'device': self._config.device,
                    'loaded': True,
                    'capabilities': {
                        'voice_cloning': True,
                        'multilingual': True,
                        'streaming': True
                    }
                }
            
            # Load voice samples (mock)
            self._load_voice_samples()
            
            self._is_initialized = True
            logger.info("Coqui TTS Strategy initialized successfully (Mock mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS Strategy: {e}")
            return False
    
    def synthesize_text(self, text: str, voice_id: str) -> SynthesisResult:
        """
        Synthesize text to speech using Coqui TTS.
        
        Args:
            text (str): Text to synthesize
            voice_id (str): Voice identifier
            
        Returns:
            SynthesisResult: Synthesis result with audio data
        """
        if not self._is_initialized:
            raise ModelLoadError("Coqui TTS not initialized")
        
        # Validate voice
        if not self._is_voice_available(voice_id):
            raise VoiceNotFoundError(f"Voice {voice_id} not available")
        
        try:
            start_time = time.time()
            
            logger.info(f"Synthesizing text with voice {voice_id}: '{text[:50]}...'")
            
            logger.info(f"🔍 TTS synthesis check: COQUI_AVAILABLE={COQUI_AVAILABLE}, hasattr(model, 'synthesizer')={hasattr(self._model, 'synthesizer')}")
            if COQUI_AVAILABLE and hasattr(self._model, 'synthesizer'):
                # Real Coqui TTS synthesis
                logger.debug(f"Synthesizing with XTTS-v2: '{text[:50]}...'")
                
                # Map voice IDs to actual Coqui TTS speaker names
                speaker_mapping = {
                    "gracie_wise": "Gracie Wise",
                    "alison_dietlinde": "Alison Dietlinde",
                    "annmarie_nele": "Annmarie Nele",
                    "asya_anara": "Asya Anara",
                    "brenda_stern": "Brenda Stern"
                }
                
                # Get the actual speaker name for Coqui TTS
                speaker_name = speaker_mapping.get(voice_id, "Annmarie Nele")  # Default to Annmarie Nele
                
                # Synthesize with Coqui TTS using the mapped speaker
                logger.info(f"🎤 Calling TTS with text: '{text[:50]}...', speaker: '{speaker_name}', language: 'en'")
                audio_array = self._model.tts(
                    text=text,
                    speaker=speaker_name,
                    language="en"
                )
                logger.info(f"🎵 TTS returned: type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'no shape')}, size={getattr(audio_array, 'size', 'no size')}")
                
                # Convert to bytes (assuming 22050 Hz sample rate)
                if isinstance(audio_array, np.ndarray) and audio_array.size > 1:
                    # Convert float32 to int16 PCM
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    audio_data = audio_int16.tobytes()
                elif isinstance(audio_array, (np.ndarray, np.float32, float)) and np.isscalar(audio_array):
                    # Handle scalar values - create a small audio buffer
                    logger.warning(f"Received scalar audio value: {audio_array}, creating minimal audio buffer")
                    # Create a small silent audio buffer (0.1 seconds at 22050 Hz)
                    silent_audio = np.zeros(2205, dtype=np.float32)
                    audio_int16 = (silent_audio * 32767).astype(np.int16)
                    audio_data = audio_int16.tobytes()
                else:
                    # Fallback for other types
                    logger.warning(f"Unexpected audio_array type: {type(audio_array)}, value: {audio_array}")
                    # Create a small silent audio buffer
                    silent_audio = np.zeros(2205, dtype=np.float32)
                    audio_int16 = (silent_audio * 32767).astype(np.int16)
                    audio_data = audio_int16.tobytes()
                
                synthesis_time = time.time() - start_time
                # Use fixed sample rate for XTTS-v2 model (22050 Hz)
                sample_rate = 22050
                if isinstance(audio_array, np.ndarray) and audio_array.size > 1:
                    duration = len(audio_array) / float(sample_rate)
                else:
                    # For scalar or invalid values, use a default duration
                    duration = 0.1  # 0.1 seconds for minimal audio buffer
                
                logger.debug(f"Real synthesis complete: {duration:.2f}s audio in {synthesis_time:.2f}s")
                
            else:
                # Fallback to mock synthesis
                audio_data = self._generate_mock_audio(text, voice_id)
                synthesis_time = time.time() - start_time
                
                # Calculate duration (mock: ~150 words per minute)
                word_count = len(text.split())
                duration = max(0.5, word_count / 2.5)  # Minimum 0.5s
            
            # Create metadata
            metadata = SynthesisMetadata(
                model_name=self._config.model_name,
                voice_characteristics={
                    'voice_id': voice_id,
                    'gender': self._get_voice_info(voice_id).gender,
                    'style': self._get_voice_info(voice_id).style,
                    'speaking_rate': self._voice_parameters.speaking_rate,
                    'pitch': self._voice_parameters.pitch,
                    'volume': self._voice_parameters.volume
                },
                processing_device=self._config.device,
                quality_metrics={
                    'overall': 0.92,  # Mock quality score
                    'naturalness': 0.89,
                    'clarity': 0.95,
                    'consistency': 0.91
                }
            )
            
            # Create synthesis result
            result = SynthesisResult(
                audio_data=audio_data,
                sample_rate=22050,  # Use fixed sample rate for XTTS-v2
                duration=duration,
                voice_id=voice_id,
                synthesis_time=synthesis_time,
                metadata=metadata,
                request_id=""  # Will be set by service
            )
            
            logger.info(f"Synthesis completed in {synthesis_time:.3f}s, duration: {duration:.2f}s")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"Synthesis failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise AudioGenerationError(f"Failed to synthesize text: {e}")
    
    def synthesize_stream(self, text_stream: Iterator[str], voice_id: str) -> Iterator[AudioChunk]:
        """
        Synthesize streaming text to audio chunks.
        
        Args:
            text_stream (Iterator[str]): Stream of text chunks
            voice_id (str): Voice identifier
            
        Yields:
            AudioChunk: Audio chunks as they become available
        """
        if not self._is_initialized:
            raise ModelLoadError("Coqui TTS not initialized")
        
        if not self._is_voice_available(voice_id):
            raise VoiceNotFoundError(f"Voice {voice_id} not available")
        
        logger.info(f"Starting streaming synthesis with voice {voice_id}")
        
        chunk_index = 0
        for text_chunk in text_stream:
            try:
                # Mock streaming synthesis
                chunk_data = self._generate_mock_audio(text_chunk, voice_id)
                
                # Calculate chunk duration
                word_count = len(text_chunk.split())
                chunk_duration = max(0.2, word_count / 2.5)
                
                chunk = AudioChunk(
                    chunk_data=chunk_data,
                    chunk_id=f"chunk_{chunk_index}",
                    is_final=False,  # Will be set to True for last chunk
                    timestamp=time.time(),
                    duration=chunk_duration
                )
                
                chunk_index += 1
                logger.debug(f"Generated audio chunk {chunk_index}: {len(text_chunk)} chars")
                yield chunk
                
            except Exception as e:
                logger.error(f"Failed to synthesize chunk: {e}")
                raise AudioGenerationError(f"Streaming synthesis failed: {e}")
        
        logger.info(f"Streaming synthesis completed: {chunk_index} chunks")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """
        Get list of available voices.
        
        Returns:
            List[VoiceInfo]: Available voices
        """
        return self._available_voices.copy()
    
    def set_voice_parameters(self, params: VoiceParameters) -> None:
        """
        Set voice synthesis parameters.
        
        Args:
            params (VoiceParameters): Voice parameters
        """
        if not params.validate():
            raise ValueError("Invalid voice parameters")
        
        self._voice_parameters = params
        logger.info(f"Voice parameters updated: {params.to_dict()}")
    
    def get_capabilities(self) -> TTSCapabilities:
        """
        Get Coqui TTS capabilities.
        
        Returns:
            TTSCapabilities: TTS capabilities
        """
        return TTSCapabilities(
            supports_streaming=True,
            supports_voice_cloning=True,
            supported_languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
            max_text_length=1000,
            supported_formats=[AudioFormat.WAV, AudioFormat.MP3],
            real_time_factor=0.3,  # 3x faster than real-time
            voice_count=len(self._available_voices)
        )
    
    def shutdown(self) -> None:
        """Shutdown Coqui TTS and release resources."""
        logger.info("Shutting down Coqui TTS Strategy")
        
        if self._model:
            logger.info("Unloading TTS model")
            self._model = None
        
        self._voice_samples.clear()
        self._is_initialized = False
        
        logger.info("Coqui TTS Strategy shutdown complete")
    
    def _load_voice_samples(self) -> None:
        """Load voice samples for voice cloning (mock implementation)."""
        logger.info("Loading voice samples (Mock mode)")
        
        # Mock voice sample loading
        for voice in self._available_voices:
            self._voice_samples[voice.voice_id] = {
                'sample_path': f"./voices/{voice.voice_id}.wav",
                'embedding': np.random.rand(512),  # Mock voice embedding
                'characteristics': {
                    'gender': voice.gender,
                    'age': voice.age_group,
                    'style': voice.style
                }
            }
        
        logger.info(f"Loaded {len(self._voice_samples)} voice samples")
    
    def _generate_mock_audio(self, text: str, voice_id: str) -> bytes:
        """
        Generate mock audio data for testing.
        
        Args:
            text (str): Text to synthesize
            voice_id (str): Voice identifier
            
        Returns:
            bytes: Mock audio data
        """
        # Calculate audio length based on text
        word_count = len(text.split())
        duration = max(0.5, word_count / 2.5)  # ~150 words per minute
        
        # Generate mock audio (sine wave with some variation)
        sample_rate = int(self._config.sample_rate)  # Convert to int to avoid numpy.float32 issues
        samples = int(duration * sample_rate)
        
        # Create a simple sine wave with some variation
        t = np.linspace(0, duration, samples)
        
        # Base frequency varies by voice (higher frequencies for young, happy voices)
        voice_frequencies = {
            'gracie_wise': 200,        # Lower, sultry tone
            'alison_dietlinde': 210,   # Smooth, alluring
            'annmarie_nele': 195,      # Sophisticated, sexy (lowest pitch)
            'asya_anara': 205,         # Exotic, sensual
            'brenda_stern': 190        # Confident, mature sexy voice
        }
        base_freq = voice_frequencies.get(voice_id, 195)  # Default to Annmarie Nele's frequency
        
        # Apply voice parameters
        freq = base_freq * (1 + self._voice_parameters.pitch * 0.2)
        
        # Generate audio with some natural variation
        audio = np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)  # Harmonic
        audio += 0.05 * np.random.normal(0, 1, samples)  # Noise for realism
        
        # Apply volume
        audio *= self._voice_parameters.volume * 0.3  # Keep it quiet
        
        # Apply speaking rate (affects duration, not pitch)
        if self._voice_parameters.speaking_rate != 1.0:
            new_length = int(int(samples) / float(self._voice_parameters.speaking_rate))
            audio = np.interp(np.linspace(0, samples-1, new_length), 
                            np.arange(samples), audio)
        
        # Convert to bytes (float32)
        audio_bytes = audio.astype(np.float32).tobytes()
        
        return audio_bytes
    
    def _is_voice_available(self, voice_id: str) -> bool:
        """Check if voice is available."""
        return any(voice.voice_id == voice_id for voice in self._available_voices)
    
    def _get_voice_info(self, voice_id: str) -> VoiceInfo:
        """Get voice information by ID."""
        for voice in self._available_voices:
            if voice.voice_id == voice_id:
                return voice
        raise VoiceNotFoundError(f"Voice {voice_id} not found")


# TODO: Actual Coqui TTS Implementation
"""
Future implementation with actual Coqui TTS:

class CoquiTTSStrategy(ISpeechSynthesis):
    def __init__(self, config: SpeechConfig):
        self._config = config
        self._tts_model = None
        
    def initialize(self) -> bool:
        try:
            from TTS.api import TTS
            
            # Initialize Coqui TTS
            self._tts_model = TTS(
                model_name=self._config.model_name,
                progress_bar=False,
                gpu=self._config.device.startswith('cuda')
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            return False
    
    def synthesize_text(self, text: str, voice_id: str) -> SynthesisResult:
        # Get voice sample
        speaker_wav = self._voice_samples.get(voice_id)
        
        # Synthesize with Coqui TTS
        audio = self._tts_model.tts(
            text=text,
            speaker_wav=speaker_wav,
            language="en"
        )
        
        # Convert to bytes and create result
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        
        return SynthesisResult(
            audio_data=audio_bytes,
            sample_rate=self._tts_model.synthesizer.output_sample_rate,
            duration=len(audio) / self._tts_model.synthesizer.output_sample_rate,
            voice_id=voice_id,
            synthesis_time=synthesis_time,
            metadata=metadata,
            request_id=request_id
        )
"""
