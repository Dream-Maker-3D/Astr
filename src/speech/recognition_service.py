import numpy as np
import logging
import asyncio
from typing import Dict, Any, Optional

from .strategies.whisper_stt import STTStrategy, WhisperSTTStrategy, FasterWhisperSTTStrategy
from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import SpeechRecognitionError


class SpeechRecognitionService:
    """
    Speech recognition service using Strategy pattern for different STT implementations.
    Integrates with EventBusService for decoupled communication.
    """
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self.config = ConfigurationManager()
        self._strategy: Optional[STTStrategy] = None
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        
        # Subscribe to audio events
        self.event_bus.subscribe(EventTypes.AUDIO_DATA_RECEIVED, self._handle_audio_data)
    
    async def initialize(self) -> bool:
        """Initialize the speech recognition service with configured strategy."""
        try:
            speech_config = self.config.get_speech_config()
            recognition_config = speech_config.get('recognition', {})
            
            provider = recognition_config.get('provider', 'whisper')
            model = recognition_config.get('model', 'base')
            language = recognition_config.get('language', 'en')
            
            # Create strategy based on configuration
            if provider == 'whisper':
                self._strategy = WhisperSTTStrategy(model, language)
            elif provider == 'faster-whisper':
                self._strategy = FasterWhisperSTTStrategy(model, language)
            else:
                raise SpeechRecognitionError(f"Unknown STT provider: {provider}")
            
            # Initialize the strategy
            success = await self._strategy.initialize()
            if success:
                self._is_initialized = True
                await self.event_bus.publish_async(
                    EventTypes.SYSTEM_STATUS_CHANGED,
                    {'service': 'speech_recognition', 'status': 'initialized'}
                )
                self._logger.info(f"Speech recognition service initialized with {provider}")
                return True
            else:
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to initialize speech recognition: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'speech_recognition', 'error': str(e)}
            )
            return False
    
    async def _handle_audio_data(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming audio data from the event bus."""
        try:
            audio_data = event_data['data'].get('audio_data')
            sample_rate = event_data['data'].get('sample_rate', 16000)
            
            if audio_data is not None:
                # Process the audio data
                result = await self.transcribe_audio(audio_data, sample_rate)
                
                if result and result.get('text'):
                    # Publish speech recognition result
                    await self.event_bus.publish_async(
                        EventTypes.SPEECH_RECOGNIZED,
                        {
                            'text': result['text'],
                            'confidence': result.get('confidence', 0.0),
                            'language': result.get('language', 'en'),
                            'segments': result.get('segments', [])
                        }
                    )
                    
                    # Also publish natural speech detection for conversational flow
                    await self.event_bus.publish_async(
                        EventTypes.NATURAL_SPEECH_DETECTED,
                        {
                            'text': result['text'],
                            'confidence': result.get('confidence', 0.0)
                        }
                    )
                    
        except Exception as e:
            self._logger.error(f"Error handling audio data: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'speech_recognition', 'error': str(e)}
            )
    
    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with transcription results or None if failed
        """
        if not self._is_initialized or not self._strategy:
            raise SpeechRecognitionError("Speech recognition service not initialized")
        
        try:
            # Get confidence threshold from config
            speech_config = self.config.get_speech_config()
            confidence_threshold = speech_config.get('recognition', {}).get('confidence_threshold', 0.6)
            
            # Transcribe using the strategy
            result = await self._strategy.transcribe(audio_data, sample_rate)
            
            # Filter by confidence threshold
            if result.get('confidence', 0.0) >= confidence_threshold:
                return result
            else:
                self._logger.debug(f"Transcription confidence too low: {result.get('confidence', 0.0)}")
                return None
                
        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")
            raise SpeechRecognitionError(f"Transcription failed: {e}")
    
    async def transcribe_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio from file."""
        try:
            import soundfile as sf
            
            # Load audio file
            audio_data, sample_rate = sf.read(file_path)
            
            # Ensure audio is mono and float32
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            audio_data = audio_data.astype(np.float32)
            
            return await self.transcribe_audio(audio_data, sample_rate)
            
        except Exception as e:
            self._logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise SpeechRecognitionError(f"Failed to transcribe file: {e}")
    
    def set_strategy(self, strategy: STTStrategy) -> None:
        """Change the STT strategy at runtime."""
        self._strategy = strategy
        self._is_initialized = False
        self._logger.info(f"STT strategy changed to {type(strategy).__name__}")
    
    def get_current_strategy(self) -> Optional[STTStrategy]:
        """Get the current STT strategy."""
        return self._strategy
    
    def is_ready(self) -> bool:
        """Check if the service is ready for transcription."""
        return (self._is_initialized and 
                self._strategy is not None and 
                self._strategy.is_ready())
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._strategy:
            await self._strategy.cleanup()
        
        self._is_initialized = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(EventTypes.AUDIO_DATA_RECEIVED, self._handle_audio_data)
        
        await self.event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'speech_recognition', 'status': 'cleanup_complete'}
        )
        
        self._logger.info("Speech recognition service cleanup completed")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        # This would depend on the specific strategy
        return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
    
    def get_available_models(self) -> list:
        """Get list of available models for current strategy."""
        return ['tiny', 'base', 'small', 'medium', 'large']