import numpy as np
import whisper
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import concurrent.futures

from ...utils.exceptions import SpeechRecognitionError, ModelLoadError


class STTStrategy(ABC):
    """Abstract base class for Speech-to-Text strategies following Strategy pattern."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the STT model and resources."""
        pass
    
    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Returns:
            Dict with keys: 'text', 'confidence', 'language', 'segments'
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the STT engine is ready for transcription."""
        pass


class WhisperSTTStrategy(STTStrategy):
    """Whisper-based STT implementation using Strategy pattern."""
    
    def __init__(self, model_name: str = "base", language: str = "en"):
        self.model_name = model_name
        self.language = language
        self.model: Optional[whisper.Whisper] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._logger = logging.getLogger(__name__)
        self._ready = False
    
    async def initialize(self) -> bool:
        """Initialize Whisper model asynchronously."""
        try:
            self._logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self._executor, 
                whisper.load_model, 
                self.model_name
            )
            
            self._ready = True
            self._logger.info(f"Whisper model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load Whisper model: {e}")
            raise ModelLoadError(f"Failed to load Whisper model: {e}")
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        if not self._ready or self.model is None:
            raise SpeechRecognitionError("Whisper model not initialized")
        
        try:
            # Ensure audio is float32 and properly shaped
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if needed
            if audio_data.max() > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Whisper expects 16kHz audio
            if sample_rate != 16000:
                # Simple resampling - in production, use librosa.resample
                from scipy.signal import resample
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = resample(audio_data, target_length).astype(np.float32)
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                audio_data
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")
            raise SpeechRecognitionError(f"Transcription failed: {e}")
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription method for thread execution."""
        # Use Whisper's transcribe method
        result = self.model.transcribe(
            audio_data,
            language=self.language,
            fp16=False,  # Use fp32 for better compatibility
            verbose=False
        )
        
        # Extract confidence from segments if available
        segments = result.get("segments", [])
        avg_confidence = 0.0
        
        if segments:
            confidences = []
            for segment in segments:
                # Whisper doesn't provide confidence, estimate from probability
                if "avg_logprob" in segment:
                    # Convert log probability to rough confidence estimate
                    confidence = max(0.0, min(1.0, np.exp(segment["avg_logprob"])))
                    confidences.append(confidence)
            
            if confidences:
                avg_confidence = np.mean(confidences)
        
        return {
            'text': result.get("text", "").strip(),
            'confidence': avg_confidence,
            'language': result.get("language", self.language),
            'segments': segments
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self._ready = False
        if self._executor:
            self._executor.shutdown(wait=True)
        self.model = None
        self._logger.info("Whisper STT cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if Whisper is ready for transcription."""
        return self._ready and self.model is not None


class FasterWhisperSTTStrategy(STTStrategy):
    """Faster-Whisper implementation for production use."""
    
    def __init__(self, model_name: str = "base", language: str = "en", device: str = "cpu"):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.model = None
        self._logger = logging.getLogger(__name__)
        self._ready = False
    
    async def initialize(self) -> bool:
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            
            self._logger.info(f"Loading faster-whisper model: {self.model_name}")
            
            # Load model in thread pool
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type="float32"
                )
            )
            
            self._ready = True
            self._logger.info(f"Faster-Whisper model {self.model_name} loaded successfully")
            return True
            
        except ImportError:
            raise ModelLoadError("faster-whisper not installed. Install with: pip install faster-whisper")
        except Exception as e:
            self._logger.error(f"Failed to load faster-whisper model: {e}")
            raise ModelLoadError(f"Failed to load faster-whisper model: {e}")
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Transcribe audio using faster-whisper."""
        if not self._ready or self.model is None:
            raise SpeechRecognitionError("Faster-Whisper model not initialized")
        
        try:
            # Prepare audio data
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if sample_rate != 16000:
                from scipy.signal import resample
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = resample(audio_data, target_length).astype(np.float32)
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Faster-Whisper transcription failed: {e}")
            raise SpeechRecognitionError(f"Faster-Whisper transcription failed: {e}")
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription for faster-whisper."""
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            vad_filter=True,  # Enable voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        text_parts = []
        segment_list = []
        confidences = []
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'avg_logprob': getattr(segment, 'avg_logprob', -1.0)
            })
            
            # Convert log probability to confidence estimate
            if hasattr(segment, 'avg_logprob'):
                confidence = max(0.0, min(1.0, np.exp(segment.avg_logprob)))
                confidences.append(confidence)
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'language': info.language,
            'segments': segment_list
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self._ready = False
        self.model = None
        self._logger.info("Faster-Whisper STT cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if faster-whisper is ready."""
        return self._ready and self.model is not None