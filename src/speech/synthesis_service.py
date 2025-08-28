"""
Speech Synthesis Service - Main TTS orchestrator.

This service manages text-to-speech synthesis using the Strategy pattern,
integrates with the Event Bus for communication, and coordinates with
the Audio Player Service for playback.
"""

import time
import threading
from queue import Queue, PriorityQueue, Empty
from typing import Optional, List, Iterator, Dict, Any
from datetime import datetime
import logging

from src.core.event_bus import EventBusService
from src.speech.strategies.base_synthesis import (
    ISpeechSynthesis, SynthesisRequest, SynthesisResult, AudioChunk,
    VoiceParameters, VoiceInfo, TTSCapabilities, TTSStatistics,
    Priority, TTSEventTypes, SpeechConfig
)
from src.utils.exceptions import (
    SpeechSynthesisError, ModelLoadError, VoiceNotFoundError,
    SynthesisTimeoutError, AudioGenerationError
)

logger = logging.getLogger(__name__)


class SpeechSynthesisService:
    """
    Main Speech Synthesis Service implementing TTS orchestration.
    
    Uses the Strategy pattern for TTS providers and integrates with
    the Event Bus for system-wide communication.
    """
    
    def __init__(self, event_bus: EventBusService, config: SpeechConfig):
        """
        Initialize the Speech Synthesis Service.
        
        Args:
            event_bus (EventBusService): Event bus for communication
            config (SpeechConfig): TTS configuration
        """
        self._event_bus = event_bus
        self._config = config
        self._tts_strategy: Optional[ISpeechSynthesis] = None
        self._audio_player = None  # Will be injected later
        
        # Service state
        self._is_initialized = False
        self._is_running = False
        self._current_voice = config.default_voice_id
        self._voice_parameters = VoiceParameters(
            speaking_rate=config.default_speaking_rate,
            pitch=config.default_pitch,
            volume=config.default_volume,
            naturalness=config.default_naturalness
        )
        
        # Processing queue and threading
        self._synthesis_queue = PriorityQueue()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics and monitoring
        self._statistics = TTSStatistics()
        self._start_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Speech Synthesis Service initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the Speech Synthesis Service.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            with self._lock:
                if self._is_initialized:
                    logger.warning("Speech Synthesis Service already initialized")
                    return True
                
                # Validate configuration
                if not self._config.validate():
                    raise ValueError("Invalid speech synthesis configuration")
                
                # Initialize TTS strategy
                from .strategies.coqui_tts import CoquiTTSStrategy
                tts_strategy = CoquiTTSStrategy(self._config)
                self.set_strategy(tts_strategy)
                
                # Subscribe to events
                self._event_bus.subscribe("ai.response.ready", self._handle_ai_response)
                self._event_bus.subscribe("SYNTHESIS_REQUEST", self._handle_synthesis_request)
                
                # Start worker thread
                self._is_running = True
                self._worker_thread = threading.Thread(
                    target=self._process_synthesis_worker,
                    name="SynthesisWorker",
                    daemon=True
                )
                self._worker_thread.start()
                
                self._is_initialized = True
                
                # Publish initialization event
                self._publish_event(TTSEventTypes.TTS_SERVICE_INITIALIZED, {
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'model_name': self._config.model_name,
                        'default_voice': self._config.default_voice_id,
                        'device': self._config.device
                    }
                })
                
                logger.info("Speech Synthesis Service initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Speech Synthesis Service: {e}")
            return False
    
    def set_strategy(self, strategy: ISpeechSynthesis) -> None:
        """
        Set the TTS strategy to use.
        
        Args:
            strategy (ISpeechSynthesis): TTS strategy implementation
        """
        with self._lock:
            if self._tts_strategy:
                logger.info("Shutting down previous TTS strategy")
                self._tts_strategy.shutdown()
            
            self._tts_strategy = strategy
            
            # Initialize the strategy
            if not self._tts_strategy.initialize():
                raise ModelLoadError("Failed to initialize TTS strategy")
            
            logger.info(f"TTS strategy set to {type(strategy).__name__}")
            
            # Publish model loaded event
            self._publish_event(TTSEventTypes.TTS_MODEL_LOADED, {
                'strategy': type(strategy).__name__,
                'capabilities': self._tts_strategy.get_capabilities().__dict__,
                'timestamp': datetime.now().isoformat()
            })
    
    def synthesize_text(self, text: str, priority: Priority = Priority.NORMAL) -> str:
        """
        Queue text for synthesis and return request ID.
        
        Args:
            text (str): Text to synthesize
            priority (Priority): Synthesis priority
            
        Returns:
            str: Request ID for tracking
        """
        if not self._is_initialized or not self._tts_strategy:
            raise SpeechSynthesisError("Service not initialized or no TTS strategy set")
        
        # Create synthesis request
        request = SynthesisRequest(
            text=text,
            voice_id=self._current_voice,
            priority=priority,
            parameters=self._voice_parameters
        )
        
        # Queue request with priority (lower number = higher priority)
        priority_value = {
            Priority.URGENT: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3
        }[priority]
        
        self._synthesis_queue.put((priority_value, time.time(), request))
        
        logger.info(f"Queued synthesis request: {request.request_id}")
        return request.request_id
    
    def change_voice(self, voice_id: str) -> bool:
        """
        Change the current voice.
        
        Args:
            voice_id (str): New voice identifier
            
        Returns:
            bool: True if voice changed successfully
        """
        if not self._tts_strategy:
            return False
        
        # Check if voice is available
        available_voices = self._tts_strategy.get_available_voices()
        voice_ids = [voice.voice_id for voice in available_voices]
        
        if voice_id not in voice_ids:
            logger.error(f"Voice {voice_id} not available. Available: {voice_ids}")
            return False
        
        old_voice = self._current_voice
        self._current_voice = voice_id
        
        # Publish voice change event
        self._publish_event(TTSEventTypes.VOICE_CHANGED, {
            'old_voice': old_voice,
            'new_voice': voice_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Voice changed from {old_voice} to {voice_id}")
        return True
    
    def set_voice_parameters(self, params: VoiceParameters) -> None:
        """
        Set voice synthesis parameters.
        
        Args:
            params (VoiceParameters): Voice parameters
        """
        if not params.validate():
            raise ValueError("Invalid voice parameters")
        
        with self._lock:
            old_params = self._voice_parameters
            self._voice_parameters = params
            
            # Update strategy parameters
            if self._tts_strategy:
                self._tts_strategy.set_voice_parameters(params)
            
            # Publish parameter change event
            self._publish_event(TTSEventTypes.VOICE_PARAMETERS_CHANGED, {
                'old_parameters': old_params.to_dict(),
                'new_parameters': params.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Voice parameters updated: {params.to_dict()}")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """
        Get list of available voices.
        
        Returns:
            List[VoiceInfo]: Available voices
        """
        if not self._tts_strategy:
            return []
        
        return self._tts_strategy.get_available_voices()
    
    def get_statistics(self) -> TTSStatistics:
        """
        Get TTS service statistics.
        
        Returns:
            TTSStatistics: Service statistics
        """
        with self._lock:
            # Update uptime
            self._statistics.uptime = time.time() - self._start_time
            return self._statistics
    
    def shutdown(self) -> None:
        """Shutdown the Speech Synthesis Service."""
        logger.info("Shutting down Speech Synthesis Service")
        
        with self._lock:
            # Signal shutdown
            self._is_running = False
            self._shutdown_event.set()
            
            # Wait for worker thread
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            
            # Shutdown TTS strategy
            if self._tts_strategy:
                self._tts_strategy.shutdown()
            
            # Unsubscribe from events
            try:
                self._event_bus.unsubscribe("ai.response.ready", self._handle_ai_response)
                self._event_bus.unsubscribe("SYNTHESIS_REQUEST", self._handle_synthesis_request)
            except Exception as e:
                logger.warning(f"Error unsubscribing from events: {e}")
            
            # Publish shutdown event
            self._publish_event(TTSEventTypes.TTS_SERVICE_SHUTDOWN, {
                'timestamp': datetime.now().isoformat(),
                'statistics': self._statistics.get_performance_metrics()
            })
            
            self._is_initialized = False
            logger.info("Speech Synthesis Service shutdown complete")
    
    def _process_synthesis_worker(self) -> None:
        """Worker thread for processing synthesis requests."""
        logger.info("Synthesis worker thread started")
        
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Get request from queue with timeout
                try:
                    priority, timestamp, request = self._synthesis_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process the synthesis request
                self._handle_synthesis_request(request)
                self._synthesis_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in synthesis worker: {e}")
        
        logger.info("Synthesis worker thread stopped")
    
    def _handle_synthesis_request(self, request: SynthesisRequest) -> None:
        """
        Handle a synthesis request.
        
        Args:
            request (SynthesisRequest): Synthesis request to process
        """
        try:
            start_time = time.time()
            
            # Publish synthesis started event
            self._publish_event(TTSEventTypes.SYNTHESIS_STARTED, {
                'request_id': request.request_id,
                'text': request.text[:100] + "..." if len(request.text) > 100 else request.text,
                'voice_id': request.voice_id,
                'priority': request.priority.value,
                'timestamp': datetime.now().isoformat()
            })
            
            # Perform synthesis
            if not self._tts_strategy:
                raise SpeechSynthesisError("No TTS strategy available")
            
            result = self._tts_strategy.synthesize_text(request.text, request.voice_id)
            synthesis_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self._statistics.total_syntheses += 1
                self._statistics.total_audio_generated += result.duration
                
                # Update average synthesis time
                total_time = (self._statistics.average_synthesis_time * 
                            (self._statistics.total_syntheses - 1) + synthesis_time)
                self._statistics.average_synthesis_time = total_time / self._statistics.total_syntheses
                
                # Update voice usage
                if request.voice_id not in self._statistics.voice_usage_distribution:
                    self._statistics.voice_usage_distribution[request.voice_id] = 0
                self._statistics.voice_usage_distribution[request.voice_id] += 1
            
            # Send to audio player if available
            if self._audio_player:
                try:
                    audio_clip = result.to_audio_clip()
                    self._audio_player.play_audio(audio_clip.data, audio_clip.sample_rate)
                except Exception as e:
                    logger.error(f"Failed to queue audio for playback: {e}")
            
            # Publish synthesis completed event
            self._publish_event(TTSEventTypes.SYNTHESIS_COMPLETED, {
                'request_id': request.request_id,
                'synthesis_time': synthesis_time,
                'audio_duration': result.duration,
                'voice_id': result.voice_id,
                'quality_score': result.metadata.quality_metrics.get('overall', 0.0),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Synthesis completed: {request.request_id} in {synthesis_time:.3f}s")
            
        except Exception as e:
            # Update error statistics
            with self._lock:
                self._statistics.error_count += 1
            
            # Publish synthesis error event
            self._publish_event(TTSEventTypes.SYNTHESIS_FAILED, {
                'request_id': request.request_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.error(f"Synthesis failed for {request.request_id}: {e}")
    
    def _handle_ai_response(self, event_data: Dict[str, Any]) -> None:
        """
        Handle AI response ready event.
        
        Args:
            event_data (dict): Event data containing AI response
        """
        try:
            text = event_data.get('text', '')
            priority_str = event_data.get('priority', 'normal')
            
            # Convert priority string to enum
            priority_map = {
                'urgent': Priority.URGENT,
                'high': Priority.HIGH,
                'normal': Priority.NORMAL,
                'low': Priority.LOW
            }
            priority = priority_map.get(priority_str, Priority.NORMAL)
            
            # Queue for synthesis
            request_id = self.synthesize_text(text, priority)
            logger.info(f"AI response queued for synthesis: {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle AI response: {e}")
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish event to the Event Bus.
        
        Args:
            event_type (str): Type of event
            data (dict): Event data
        """
        try:
            self._event_bus.publish(event_type, data)
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    def set_audio_player(self, audio_player) -> None:
        """
        Set the audio player service for playback.
        
        Args:
            audio_player: Audio player service instance
        """
        self._audio_player = audio_player
        logger.info("Audio player service connected to TTS service")
