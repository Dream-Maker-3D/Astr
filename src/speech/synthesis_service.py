import numpy as np
import logging
import asyncio
import tempfile
import os
from typing import Optional, Dict, Any

from .strategies.coqui_tts import TTSStrategy, CoquiTTSStrategy
from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import SpeechSynthesisError


class SpeechSynthesisService:
    """
    Speech synthesis service using Strategy pattern for different TTS implementations.
    Integrates with EventBusService for decoupled communication.
    """
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self.config = ConfigurationManager()
        self._strategy: Optional[TTSStrategy] = None
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._reference_voice_path: Optional[str] = None
        
        # Subscribe to AI response events
        self.event_bus.subscribe(EventTypes.AI_RESPONSE_RECEIVED, self._handle_ai_response)
    
    async def initialize(self) -> bool:
        """Initialize the speech synthesis service with configured strategy."""
        try:
            speech_config = self.config.get_speech_config()
            synthesis_config = speech_config.get('synthesis', {})
            
            provider = synthesis_config.get('provider', 'coqui')
            model = synthesis_config.get('model', 'tts_models/multilingual/multi-dataset/xtts_v2')
            voice_path = synthesis_config.get('voice')
            
            # Create strategy based on configuration
            if provider == 'coqui':
                # Determine device based on GPU availability
                device = "auto"  # Let Coqui decide
                self._strategy = CoquiTTSStrategy(model, device)
            else:
                raise SpeechSynthesisError(f"Unknown TTS provider: {provider}")
            
            # Store reference voice path if provided
            if voice_path and os.path.exists(voice_path):
                self._reference_voice_path = voice_path
                self._logger.info(f"Reference voice loaded: {voice_path}")
            
            # Initialize the strategy
            success = await self._strategy.initialize()
            if success:
                self._is_initialized = True
                await self.event_bus.publish_async(
                    EventTypes.SYSTEM_STATUS_CHANGED,
                    {'service': 'speech_synthesis', 'status': 'initialized'}
                )
                self._logger.info(f"Speech synthesis service initialized with {provider}")
                return True
            else:
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to initialize speech synthesis: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'speech_synthesis', 'error': str(e)}
            )
            return False
    
    async def _handle_ai_response(self, event_data: Dict[str, Any]) -> None:
        """Handle AI response by converting to speech."""
        try:
            response_text = event_data['data'].get('text')
            conversation_id = event_data['data'].get('conversation_id')
            
            if response_text:
                # Publish synthesis start event
                await self.event_bus.publish_async(
                    EventTypes.SPEECH_SYNTHESIS_START,
                    {
                        'text': response_text,
                        'conversation_id': conversation_id
                    }
                )
                
                # Also publish streaming start for conversational flow
                await self.event_bus.publish_async(
                    EventTypes.RESPONSE_STREAMING_START,
                    {
                        'text': response_text,
                        'conversation_id': conversation_id
                    }
                )
                
                # Synthesize speech
                audio_data = await self.synthesize_text(response_text)
                
                if audio_data is not None:
                    # Publish synthesis complete event
                    await self.event_bus.publish_async(
                        EventTypes.SPEECH_SYNTHESIS_COMPLETE,
                        {
                            'audio_data': audio_data,
                            'sample_rate': self._strategy.get_sample_rate(),
                            'text': response_text,
                            'conversation_id': conversation_id
                        }
                    )
                    
        except Exception as e:
            self._logger.error(f"Error handling AI response for TTS: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'speech_synthesis', 'error': str(e)}
            )
    
    async def synthesize_text(
        self, 
        text: str,
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize
            speaker_wav: Optional path to reference speaker audio
            **kwargs: Additional synthesis parameters
            
        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized or not self._strategy:
            raise SpeechSynthesisError("Speech synthesis service not initialized")
        
        if not text.strip():
            return None
        
        try:
            # Use provided speaker_wav or default reference voice
            voice_reference = speaker_wav or self._reference_voice_path
            
            # Get synthesis parameters from config
            speech_config = self.config.get_speech_config()
            synthesis_config = speech_config.get('synthesis', {})
            
            # Add config parameters to kwargs
            if 'speaking_rate' in synthesis_config:
                kwargs.setdefault('speed', synthesis_config['speaking_rate'])
            
            # Add speaker_id if configured and no speaker_wav provided
            if not voice_reference and 'speaker_id' in synthesis_config:
                kwargs['speaker_id'] = synthesis_config['speaker_id']
            
            # Synthesize using the strategy
            if voice_reference:
                audio_data = await self._strategy.synthesize(
                    text, 
                    speaker_wav=voice_reference,
                    **kwargs
                )
            else:
                audio_data = await self._strategy.synthesize(
                    text,
                    **kwargs
                )
            
            return audio_data
            
        except Exception as e:
            self._logger.error(f"Text synthesis failed: {e}")
            raise SpeechSynthesisError(f"Text synthesis failed: {e}")
    
    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synthesize text and save to audio file."""
        if not self._is_initialized or not self._strategy:
            raise SpeechSynthesisError("Speech synthesis service not initialized")
        
        try:
            voice_reference = speaker_wav or self._reference_voice_path
            
            if hasattr(self._strategy, 'synthesize_to_file'):
                await self._strategy.synthesize_to_file(
                    text, 
                    output_path, 
                    speaker_wav=voice_reference,
                    **kwargs
                )
            else:
                # Fallback: synthesize to memory then save
                audio_data = await self.synthesize_text(text, speaker_wav, **kwargs)
                if audio_data is not None:
                    import soundfile as sf
                    sf.write(output_path, audio_data, self._strategy.get_sample_rate())
            
            return output_path
            
        except Exception as e:
            self._logger.error(f"File synthesis failed: {e}")
            raise SpeechSynthesisError(f"File synthesis failed: {e}")
    
    async def create_temp_audio(self, text: str, **kwargs) -> str:
        """Create temporary audio file for immediate playback."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        await self.synthesize_to_file(text, temp_path, **kwargs)
        return temp_path
    
    def set_reference_voice(self, voice_path: str) -> bool:
        """Set reference voice for voice cloning."""
        if os.path.exists(voice_path):
            self._reference_voice_path = voice_path
            self._logger.info(f"Reference voice set to: {voice_path}")
            return True
        else:
            self._logger.warning(f"Reference voice file not found: {voice_path}")
            return False
    
    def get_reference_voice(self) -> Optional[str]:
        """Get current reference voice path."""
        return self._reference_voice_path
    
    def set_strategy(self, strategy: TTSStrategy) -> None:
        """Change the TTS strategy at runtime."""
        self._strategy = strategy
        self._is_initialized = False
        self._logger.info(f"TTS strategy changed to {type(strategy).__name__}")
    
    def get_current_strategy(self) -> Optional[TTSStrategy]:
        """Get the current TTS strategy."""
        return self._strategy
    
    def is_ready(self) -> bool:
        """Check if the service is ready for synthesis."""
        return (self._is_initialized and 
                self._strategy is not None and 
                self._strategy.is_ready())
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        if self._strategy:
            return self._strategy.get_sample_rate()
        return 22050  # Default
    
    def get_available_speakers(self) -> list:
        """Get available speakers if supported by current strategy."""
        if self._strategy and hasattr(self._strategy, 'get_available_speakers'):
            return self._strategy.get_available_speakers()
        return []
    
    def get_available_languages(self) -> list:
        """Get available languages for current strategy."""
        if self._strategy and hasattr(self._strategy, 'get_available_languages'):
            return self._strategy.get_available_languages()
        return ['en']
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._strategy:
            await self._strategy.cleanup()
        
        self._is_initialized = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(EventTypes.AI_RESPONSE_RECEIVED, self._handle_ai_response)
        
        await self.event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'speech_synthesis', 'status': 'cleanup_complete'}
        )
        
        self._logger.info("Speech synthesis service cleanup completed")