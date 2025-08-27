import numpy as np
import logging
import asyncio
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import concurrent.futures

from ...utils.exceptions import SpeechSynthesisError, ModelLoadError


class TTSStrategy(ABC):
    """Abstract base class for Text-to-Speech strategies following Strategy pattern."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the TTS model and resources."""
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Returns:
            numpy array of audio data (float32, normalized to [-1, 1])
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the TTS engine is ready for synthesis."""
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        pass


class CoquiTTSStrategy(TTSStrategy):
    """Coqui TTS implementation using Strategy pattern with XTTS-v2 model."""
    
    def __init__(
        self, 
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
        language: str = "en"
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.tts = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._logger = logging.getLogger(__name__)
        self._ready = False
        self._sample_rate = 22050  # Default Coqui TTS sample rate
    
    async def initialize(self) -> bool:
        """Initialize Coqui TTS model asynchronously."""
        try:
            from TTS.api import TTS
            
            self._logger.info(f"Loading Coqui TTS model: {self.model_name}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.tts = await loop.run_in_executor(
                self._executor,
                self._load_model
            )
            
            self._ready = True
            self._logger.info(f"Coqui TTS model {self.model_name} loaded successfully")
            return True
            
        except ImportError:
            raise ModelLoadError("Coqui TTS not installed. Install with: pip install coqui-tts")
        except Exception as e:
            self._logger.error(f"Failed to load Coqui TTS model: {e}")
            raise ModelLoadError(f"Failed to load Coqui TTS model: {e}")
    
    def _load_model(self):
        """Load TTS model synchronously."""
        from TTS.api import TTS
        
        # Determine GPU usage
        use_gpu = self.device == "cuda" or (self.device == "auto" and self._is_gpu_available())
        
        return TTS(
            model_name=self.model_name,
            gpu=use_gpu
        )
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for TTS."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def synthesize(
        self, 
        text: str, 
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize text to speech using Coqui TTS.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio for voice cloning
            **kwargs: Additional TTS parameters
        """
        if not self._ready or self.tts is None:
            raise SpeechSynthesisError("Coqui TTS model not initialized")
        
        if not text.strip():
            raise SpeechSynthesisError("Empty text provided for synthesis")
        
        try:
            # Run synthesis in thread pool
            loop = asyncio.get_event_loop()
            wav_data = await loop.run_in_executor(
                self._executor,
                self._synthesize_sync,
                text,
                speaker_wav,
                kwargs
            )
            
            return wav_data
            
        except Exception as e:
            self._logger.error(f"TTS synthesis failed: {e}")
            raise SpeechSynthesisError(f"TTS synthesis failed: {e}")
    
    def _synthesize_sync(self, text: str, speaker_wav: Optional[str], kwargs: Dict[str, Any]) -> np.ndarray:
        """Synchronous synthesis method for thread execution."""
        # Prepare TTS arguments
        tts_kwargs = {
            'text': text,
            'language': self.language,
        }
        
        # Add speaker reference if provided (for voice cloning)
        if speaker_wav and os.path.exists(speaker_wav):
            tts_kwargs['speaker_wav'] = speaker_wav
        elif 'speaker_id' in kwargs:
            # Use speaker_id from kwargs
            tts_kwargs['speaker'] = kwargs.pop('speaker_id')
        
        # Add any additional parameters
        tts_kwargs.update(kwargs)
        
        # Generate audio
        wav = self.tts.tts(**tts_kwargs)
        
        # Convert to numpy array and ensure proper format
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        
        # Ensure audio is normalized to [-1, 1]
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / np.max(np.abs(wav))
        
        return wav
    
    async def synthesize_to_file(
        self, 
        text: str, 
        output_path: str,
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synthesize text and save to file."""
        if not self._ready or self.tts is None:
            raise SpeechSynthesisError("Coqui TTS model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._synthesize_to_file_sync,
                text,
                output_path,
                speaker_wav,
                kwargs
            )
            
            return output_path
            
        except Exception as e:
            self._logger.error(f"TTS file synthesis failed: {e}")
            raise SpeechSynthesisError(f"TTS file synthesis failed: {e}")
    
    def _synthesize_to_file_sync(
        self, 
        text: str, 
        output_path: str, 
        speaker_wav: Optional[str], 
        kwargs: Dict[str, Any]
    ) -> None:
        """Synchronous file synthesis."""
        tts_kwargs = {
            'text': text,
            'file_path': output_path,
            'language': self.language,
        }
        
        if speaker_wav and os.path.exists(speaker_wav):
            tts_kwargs['speaker_wav'] = speaker_wav
        
        tts_kwargs.update(kwargs)
        
        self.tts.tts_to_file(**tts_kwargs)
    
    async def create_temp_audio_file(self, text: str, **kwargs) -> str:
        """Create temporary audio file for immediate playback."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        await self.synthesize_to_file(text, temp_path, **kwargs)
        return temp_path
    
    def get_available_speakers(self) -> list:
        """Get list of available speakers if supported by the model."""
        if self.tts and hasattr(self.tts, 'speakers'):
            return self.tts.speakers
        return []
    
    def get_available_languages(self) -> list:
        """Get list of available languages."""
        if self.tts and hasattr(self.tts, 'languages'):
            return self.tts.languages
        return [self.language]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self._ready = False
        if self._executor:
            self._executor.shutdown(wait=True)
        self.tts = None
        self._logger.info("Coqui TTS cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if Coqui TTS is ready for synthesis."""
        return self._ready and self.tts is not None
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        return self._sample_rate


class EdgeTTSStrategy(TTSStrategy):
    """Alternative TTS using Microsoft Edge TTS (for comparison/fallback)."""
    
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self._logger = logging.getLogger(__name__)
        self._ready = False
        self._sample_rate = 24000  # Edge TTS sample rate
    
    async def initialize(self) -> bool:
        """Initialize Edge TTS."""
        try:
            import edge_tts
            self._ready = True
            self._logger.info("Edge TTS initialized successfully")
            return True
        except ImportError:
            raise ModelLoadError("edge-tts not installed. Install with: pip install edge-tts")
        except Exception as e:
            self._logger.error(f"Failed to initialize Edge TTS: {e}")
            raise ModelLoadError(f"Failed to initialize Edge TTS: {e}")
    
    async def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Synthesize using Edge TTS."""
        if not self._ready:
            raise SpeechSynthesisError("Edge TTS not initialized")
        
        try:
            import edge_tts
            import io
            
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            
            # Generate audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # Convert to numpy array (this would need proper audio decoding)
            # This is a simplified implementation - in practice, you'd need to decode the audio
            # For now, return empty array as placeholder
            return np.array([], dtype=np.float32)
            
        except Exception as e:
            self._logger.error(f"Edge TTS synthesis failed: {e}")
            raise SpeechSynthesisError(f"Edge TTS synthesis failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up Edge TTS resources."""
        self._ready = False
        self._logger.info("Edge TTS cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if Edge TTS is ready."""
        return self._ready
    
    def get_sample_rate(self) -> int:
        """Get Edge TTS sample rate."""
        return self._sample_rate