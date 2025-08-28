"""
Speech Recognition Service - Main STT orchestrator.

This module implements the main speech recognition service that coordinates
STT processing using the Strategy pattern and integrates with the Event Bus.
"""

import logging
import threading
import time
import uuid
from queue import Queue, Empty
from typing import Iterator, Optional, Dict, Any
from datetime import datetime

from ..core import EventBusService, EventTypes, SpeechConfig
from ..audio import AudioData
from ..utils.exceptions import (
    InitializationError,
    SpeechRecognitionError,
    TranscriptionError
)

from .strategies.base import (
    ISpeechRecognition,
    TranscriptionResult,
    PartialResult,
    LanguageResult,
    STTCapabilities,
    STTStatistics,
    TranscriptionSegment
)


class STTEventTypes:
    """STT-specific event types."""
    
    TRANSCRIPTION_STARTED = "TRANSCRIPTION_STARTED"
    TRANSCRIPTION_PARTIAL = "TRANSCRIPTION_PARTIAL"
    TRANSCRIPTION_COMPLETED = "TRANSCRIPTION_COMPLETED"
    TRANSCRIPTION_ERROR = "TRANSCRIPTION_ERROR"
    LANGUAGE_DETECTED = "LANGUAGE_DETECTED"
    STT_PERFORMANCE_WARNING = "STT_PERFORMANCE_WARNING"


class SpeechRecognitionService:
    """
    Main speech recognition service using Strategy pattern.
    
    Coordinates STT processing, manages audio queues, publishes events,
    and provides performance monitoring for speech-to-text operations.
    """
    
    def __init__(self, event_bus: EventBusService, config: SpeechConfig):
        """
        Initialize the Speech Recognition Service.
        
        Args:
            event_bus: Event bus for publishing STT events
            config: Speech recognition configuration
        """
        self._event_bus = event_bus
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Strategy pattern - STT implementation
        self._stt_strategy: Optional[ISpeechRecognition] = None
        
        # Service state
        self._is_initialized = False
        self._is_processing = False
        self._is_streaming = False
        
        # Audio processing queue
        self._processing_queue: Queue[AudioData] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics tracking
        self._statistics = STTStatistics(
            total_transcriptions=0,
            average_processing_time=0.0,
            accuracy_score=0.95,  # Default estimate
            language_distribution={},
            error_count=0,
            uptime=0.0
        )
        self._start_time = time.time()
        self._processing_times = []
        
        # Conversation context for enhanced transcription
        self._conversation_history = []
        self._context_window = 10  # Keep last 10 transcriptions
        
        # Performance monitoring
        self._performance_threshold = {
            'processing_time': 0.5,  # 500ms
            'memory_usage': 512 * 1024 * 1024,  # 512MB
            'error_rate': 0.05  # 5%
        }
    
    def initialize(self) -> bool:
        """
        Initialize the service and load the configured STT strategy.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            InitializationError: If initialization fails
        """
        if self._is_initialized:
            self._logger.warning("Speech Recognition Service already initialized")
            return True
        
        try:
            # Load the configured STT strategy
            self._load_stt_strategy()
            
            # Subscribe to audio events
            self._subscribe_to_events()
            
            # Start worker thread for audio processing
            self._start_worker_thread()
            
            self._is_initialized = True
            self._logger.info("Speech Recognition Service initialized successfully")
            
            # Publish initialization event
            self._publish_event(EventTypes.SYSTEM_READY, {
                'service': 'SpeechRecognitionService',
                'strategy': self._stt_strategy.__class__.__name__,
                'capabilities': self._stt_strategy.get_capabilities().__dict__,
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Speech Recognition Service: {e}")
            raise InitializationError(f"STT service initialization failed: {e}")
    
    def set_strategy(self, strategy: ISpeechRecognition) -> None:
        """
        Set the STT strategy implementation.
        
        Args:
            strategy: STT strategy to use
            
        Raises:
            SpeechRecognitionError: If strategy setting fails
        """
        try:
            # Shutdown current strategy if exists
            if self._stt_strategy:
                self._stt_strategy.shutdown()
            
            # Initialize new strategy
            self._stt_strategy = strategy
            if not self._stt_strategy.initialize():
                raise SpeechRecognitionError("Failed to initialize STT strategy")
            
            self._logger.info(f"STT strategy set to: {strategy.__class__.__name__}")
            
            # Publish strategy change event
            self._publish_event("STT_STRATEGY_CHANGED", {
                'new_strategy': strategy.__class__.__name__,
                'capabilities': strategy.get_capabilities().__dict__,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self._logger.error(f"Failed to set STT strategy: {e}")
            raise SpeechRecognitionError(f"Strategy setting failed: {e}")
    
    def process_audio(self, audio_data: AudioData) -> TranscriptionResult:
        """
        Process a single audio segment for transcription.
        
        Args:
            audio_data: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription with metadata
            
        Raises:
            TranscriptionError: If transcription fails
        """
        if not self._is_initialized or not self._stt_strategy:
            raise SpeechRecognitionError("Service not initialized or no STT strategy set")
        
        start_time = time.time()
        audio_id = str(uuid.uuid4())
        
        try:
            # Publish transcription started event
            self._publish_event(STTEventTypes.TRANSCRIPTION_STARTED, {
                'audio_id': audio_id,
                'audio_duration': audio_data.duration_ms / 1000.0,  # Convert ms to seconds
                'timestamp': datetime.now(),
                'model_info': {
                    'strategy': self._stt_strategy.__class__.__name__,
                    'language': self._config.recognition_language
                }
            })
            
            # Perform transcription
            result = self._stt_strategy.transcribe_audio(audio_data)
            result.audio_id = audio_id
            
            # Debug logging for transcription result
            self._logger.info(f"✅ Transcription completed: '{result.text}' (confidence: {result.confidence:.3f})")
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(result, processing_time)
            
            # Add to conversation context
            self._add_to_context(result.text)
            
            # Publish completion event
            self._publish_event(STTEventTypes.TRANSCRIPTION_COMPLETED, {
                'audio_id': audio_id,
                'text': result.text,
                'confidence': result.confidence,
                'language': result.language,
                'processing_time': processing_time,
                'word_count': len(result.text.split()),
                'timestamp': datetime.now(),
                'metadata': result.metadata
            })
            
            # Check performance and publish warnings if needed
            self._check_performance_thresholds(processing_time)
            
            return result
            
        except Exception as e:
            self._statistics.error_count += 1
            self._logger.error(f"Transcription failed for audio {audio_id}: {e}")
            
            # Publish error event
            self._publish_event(STTEventTypes.TRANSCRIPTION_ERROR, {
                'audio_id': audio_id,
                'error_type': type(e).__name__,
                'message': str(e),
                'retry_possible': isinstance(e, TranscriptionError),
                'timestamp': datetime.now()
            })
            
            raise TranscriptionError(f"Transcription failed: {e}")
    
    def start_streaming(self, audio_stream: Iterator[AudioData]) -> None:
        """
        Start streaming transcription mode.
        
        Args:
            audio_stream: Iterator of audio chunks
        """
        if not self._is_initialized or not self._stt_strategy:
            raise SpeechRecognitionError("Service not initialized")
        
        if self._is_streaming:
            self._logger.warning("Streaming already active")
            return
        
        self._is_streaming = True
        self._logger.info("Starting streaming transcription")
        
        try:
            # Start streaming transcription in separate thread
            streaming_thread = threading.Thread(
                target=self._process_streaming_audio,
                args=(audio_stream,),
                daemon=True
            )
            streaming_thread.start()
            
        except Exception as e:
            self._is_streaming = False
            self._logger.error(f"Failed to start streaming: {e}")
            raise SpeechRecognitionError(f"Streaming start failed: {e}")
    
    def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        if self._is_streaming:
            self._is_streaming = False
            self._logger.info("Stopping streaming transcription")
    
    def get_statistics(self) -> STTStatistics:
        """
        Get service performance statistics.
        
        Returns:
            STTStatistics: Current performance statistics
        """
        # Update uptime
        self._statistics.uptime = time.time() - self._start_time
        
        # Calculate average processing time
        if self._processing_times:
            self._statistics.average_processing_time = sum(self._processing_times) / len(self._processing_times)
        
        return self._statistics
    
    def shutdown(self) -> None:
        """Shutdown the service and clean up resources."""
        if not self._is_initialized:
            return
        
        self._logger.info("Shutting down Speech Recognition Service...")
        
        # Stop streaming
        self.stop_streaming()
        
        # Signal shutdown to worker thread
        self._shutdown_event.set()
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # Shutdown STT strategy
        if self._stt_strategy:
            self._stt_strategy.shutdown()
        
        self._is_initialized = False
        self._logger.info("Speech Recognition Service shutdown complete")
    
    def _load_stt_strategy(self) -> None:
        """Load the configured STT strategy."""
        # Import here to avoid circular imports
        from .strategies.whisper_stt import WhisperSTTStrategy
        
        # For now, default to Whisper strategy
        # In the future, this could be configurable
        strategy = WhisperSTTStrategy(self._config)
        self.set_strategy(strategy)
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events from other services."""
        self._event_bus.subscribe("AUDIO_DATA_RECEIVED", self._handle_audio_data)
        self._event_bus.subscribe("SPEECH_DETECTED", self._handle_speech_start)
        self._event_bus.subscribe("SPEECH_ENDED", self._handle_speech_end)
    
    def _handle_audio_data(self, event_data: Dict[str, Any]) -> None:
        """Handle audio data received from Audio Capture Service."""
        if not self._is_processing:
            return
        
        # Extract audio data from event
        # This would need to be implemented based on the actual event structure
        pass
    
    def _handle_speech_start(self, event_data: Dict[str, Any]) -> None:
        """Handle speech detection start event."""
        self._is_processing = True
        self._logger.debug("Speech detected - starting transcription processing")
    
    def _handle_speech_end(self, event_data: Dict[str, Any]) -> None:
        """Handle speech detection end event."""
        self._is_processing = False
        self._logger.debug("Speech ended - processing audio for transcription")
        
        # Extract audio data from the event
        audio_data = event_data.get('audio_data')
        if audio_data:
            self._logger.info(f"Processing speech audio: {audio_data.duration_ms/1000.0:.2f}s, {len(audio_data.data)} samples")
            # Queue the audio for transcription
            try:
                self._processing_queue.put(audio_data, timeout=1.0)
            except Exception as e:
                self._logger.error(f"Failed to queue audio for transcription: {e}")
        else:
            self._logger.warning("SPEECH_ENDED event received but no audio_data found")
    
    def _start_worker_thread(self) -> None:
        """Start the worker thread for processing audio queue."""
        self._worker_thread = threading.Thread(
            target=self._process_audio_worker,
            daemon=True
        )
        self._worker_thread.start()
    
    def _process_audio_worker(self) -> None:
        """Worker thread for processing queued audio data."""
        while not self._shutdown_event.is_set():
            try:
                # Get audio from queue with timeout
                audio_data = self._processing_queue.get(timeout=1.0)
                
                # Process the audio
                self.process_audio(audio_data)
                
                # Mark task as done
                self._processing_queue.task_done()
                
            except Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                self._logger.error(f"Error in audio processing worker: {e}")
    
    def _process_streaming_audio(self, audio_stream: Iterator[AudioData]) -> None:
        """Process streaming audio in separate thread."""
        try:
            for partial_result in self._stt_strategy.transcribe_stream(audio_stream):
                if not self._is_streaming:
                    break
                
                # Publish partial result
                self._publish_event(STTEventTypes.TRANSCRIPTION_PARTIAL, {
                    'partial_text': partial_result.partial_text,
                    'confidence': partial_result.confidence,
                    'is_final': partial_result.is_final,
                    'segment_id': partial_result.segment_id,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            self._logger.error(f"Streaming transcription error: {e}")
            self._publish_event(STTEventTypes.TRANSCRIPTION_ERROR, {
                'error_type': 'streaming_error',
                'message': str(e),
                'timestamp': datetime.now()
            })
        finally:
            self._is_streaming = False
    
    def _update_statistics(self, result: TranscriptionResult, processing_time: float) -> None:
        """Update service statistics with transcription result."""
        self._statistics.total_transcriptions += 1
        self._processing_times.append(processing_time)
        
        # Keep only recent processing times for average calculation
        if len(self._processing_times) > 100:
            self._processing_times = self._processing_times[-100:]
        
        # Update language distribution
        lang = result.language
        if lang in self._statistics.language_distribution:
            self._statistics.language_distribution[lang] += 1
        else:
            self._statistics.language_distribution[lang] = 1
    
    def _add_to_context(self, text: str) -> None:
        """Add transcription to conversation context."""
        self._conversation_history.append(text)
        
        # Keep only recent context
        if len(self._conversation_history) > self._context_window:
            self._conversation_history = self._conversation_history[-self._context_window:]
    
    def _check_performance_thresholds(self, processing_time: float) -> None:
        """Check performance thresholds and publish warnings if needed."""
        if processing_time > self._performance_threshold['processing_time']:
            self._publish_event(STTEventTypes.STT_PERFORMANCE_WARNING, {
                'metric': 'processing_time',
                'current_value': processing_time,
                'threshold': self._performance_threshold['processing_time'],
                'recommendation': 'consider_model_optimization',
                'timestamp': datetime.now()
            })
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to the Event Bus."""
        try:
            self._event_bus.publish(event_type, data)
        except Exception as e:
            self._logger.error(f"Failed to publish event {event_type}: {e}")


# Re-export data classes for convenience
__all__ = [
    'SpeechRecognitionService',
    'STTEventTypes',
    'TranscriptionResult',
    'PartialResult',
    'LanguageResult',
    'STTCapabilities',
    'STTStatistics'
]
