"""
Audio Capture Service - Continuous Voice Input

This module implements continuous audio capture with Voice Activity Detection (VAD)
for natural conversation without activation words, following the Strategy pattern.
"""

import logging
import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from threading import Lock, Event as ThreadEvent

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import AudioConfig
from ..utils.exceptions import (
    AudioError,
    AudioDeviceError, 
    AudioCaptureError,
    InitializationError
)


@dataclass
class AudioDevice:
    """Audio device information."""
    device_id: int
    name: str
    channels: int
    sample_rate: int
    is_input: bool
    is_default: bool


@dataclass
class AudioData:
    """Audio data container with metadata."""
    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float
    duration_ms: float
    rms_level: float
    is_speech: bool


class VoiceActivityDetector:
    """Simple Voice Activity Detection based on RMS energy."""
    
    def __init__(self, threshold: float = 0.015, window_size: int = 1024):
        """
        Initialize VAD.
        
        Args:
            threshold: RMS threshold for voice detection
            window_size: Audio window size for analysis
        """
        self.threshold = threshold
        self.window_size = window_size
        self.history = deque(maxlen=10)  # Keep last 10 RMS values
        self._logger = logging.getLogger(__name__)
    
    def detect_speech(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """
        Detect speech in audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Tuple of (is_speech, rms_level)
        """
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio_data ** 2)) if len(audio_data) > 0 else 0.0
        
        # Add to history for smoothing
        self.history.append(rms)
        
        # Use moving average for stability
        avg_rms = np.mean(self.history) if self.history else 0.0
        
        # Detect speech if RMS exceeds threshold
        is_speech = avg_rms > self.threshold
        
        return is_speech, rms
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update the VAD threshold."""
        self.threshold = new_threshold
        self._logger.debug(f"VAD threshold updated to {new_threshold}")


class AudioCaptureService:
    """
    Audio Capture Service for continuous voice input.
    
    Provides continuous audio capture with Voice Activity Detection,
    event-driven communication, and configurable audio parameters.
    """
    
    def __init__(self, event_bus: EventBusService, config: AudioConfig):
        """
        Initialize the Audio Capture Service.
        
        Args:
            event_bus: Event bus for publishing audio events
            config: Audio configuration settings
        """
        self._event_bus = event_bus
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Audio system state
        self._is_initialized = False
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = ThreadEvent()
        
        # PyAudio components
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._selected_device: Optional[AudioDevice] = None
        
        # Voice Activity Detection
        self._vad = VoiceActivityDetector(
            threshold=0.015,  # Will be updated from config
            window_size=config.input_chunk_size
        )
        
        # Audio buffering
        self._audio_buffer = deque(maxlen=100)  # Keep last 100 chunks
        self._speech_buffer = deque(maxlen=50)   # Active speech buffer
        self._buffer_lock = Lock()
        
        # Speech state tracking
        self._is_speech_active = False
        self._speech_start_time: Optional[float] = None
        self._silence_counter = 0
        self._silence_threshold = 30  # Frames of silence before speech ends
        
        # Statistics
        self._stats = {
            'total_frames_captured': 0,
            'speech_frames_detected': 0,
            'total_capture_time': 0.0,
            'average_rms_level': 0.0,
            'device_errors': 0
        }
        self._stats_lock = Lock()
    
    def initialize(self) -> bool:
        """
        Initialize the audio capture service.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            AudioDeviceError: If audio system initialization fails
            InitializationError: If service initialization fails
        """
        if self._is_initialized:
            self._logger.warning("Audio Capture Service already initialized")
            return True
        
        if not PYAUDIO_AVAILABLE:
            self._logger.warning("PyAudio not available - audio capture disabled")
            # Create a mock device for testing
            self._selected_device = AudioDevice(
                device_id=0,
                name="Mock Audio Device",
                channels=1,
                sample_rate=16000,
                is_input=True,
                is_default=True
            )
            self._is_initialized = True
            return True
        
        try:
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            # Detect and select audio device
            self._detect_audio_devices()
            self._select_audio_device()
            
            self._is_initialized = True
            self._logger.info("Audio Capture Service initialized successfully")
            
            # Publish initialization event
            self._event_bus.publish(EventTypes.SYSTEM_READY, {
                'service': 'AudioCaptureService',
                'device': self._selected_device.name if self._selected_device else 'None',
                'sample_rate': self._config.input_sample_rate,
                'channels': self._config.input_channels
            })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Audio Capture Service: {e}")
            raise InitializationError(f"Audio capture initialization failed: {e}")
    
    def start_capture(self) -> bool:
        """
        Start continuous audio capture.
        
        Returns:
            bool: True if capture started successfully
            
        Raises:
            AudioCaptureError: If capture cannot be started
        """
        if not self._is_initialized:
            raise AudioCaptureError("Service not initialized")
        
        if self._is_capturing:
            self._logger.warning("Audio capture already running")
            return True
        
        if not PYAUDIO_AVAILABLE:
            self._logger.warning("PyAudio not available - simulating audio capture")
            self._is_capturing = True
            return True
        
        try:
            # Create audio stream
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._config.input_channels,
                rate=self._config.input_sample_rate,
                input=True,
                input_device_index=self._selected_device.device_id if self._selected_device else None,
                frames_per_buffer=self._config.input_chunk_size,
                stream_callback=None  # We'll use blocking read
            )
            
            # Start capture thread
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="AudioCapture",
                daemon=True
            )
            
            self._is_capturing = True
            self._capture_thread.start()
            
            self._logger.info("Audio capture started")
            
            # Publish capture started event
            self._event_bus.publish(EventTypes.AUDIO_DATA_RECEIVED, {
                'status': 'capture_started',
                'device': self._selected_device.name if self._selected_device else 'default',
                'sample_rate': self._config.input_sample_rate
            })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start audio capture: {e}")
            raise AudioCaptureError(f"Cannot start audio capture: {e}")
    
    def stop_capture(self) -> bool:
        """
        Stop audio capture.
        
        Returns:
            bool: True if capture stopped successfully
        """
        if not self._is_capturing:
            return True
        
        try:
            # Signal stop
            self._stop_event.set()
            
            # Wait for capture thread to finish
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            
            # Close audio stream
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            
            self._is_capturing = False
            self._logger.info("Audio capture stopped")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error stopping audio capture: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the audio capture service."""
        if self._is_capturing:
            self.stop_capture()
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        self._is_initialized = False
        self._logger.info("Audio Capture Service shutdown complete")
    
    def get_available_devices(self) -> List[AudioDevice]:
        """
        Get list of available audio input devices.
        
        Returns:
            List of AudioDevice objects
        """
        if not self._pyaudio:
            return []
        
        devices = []
        device_count = self._pyaudio.get_device_count()
        
        for i in range(device_count):
            try:
                info = self._pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:  # Input device
                    device = AudioDevice(
                        device_id=i,
                        name=info['name'],
                        channels=info['maxInputChannels'],
                        sample_rate=int(info['defaultSampleRate']),
                        is_input=True,
                        is_default=(i == self._pyaudio.get_default_input_device_info()['index'])
                    )
                    devices.append(device)
            except Exception as e:
                self._logger.warning(f"Error getting device {i} info: {e}")
        
        return devices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio capture statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            stats['is_capturing'] = self._is_capturing
            stats['is_speech_active'] = self._is_speech_active
            stats['selected_device'] = self._selected_device.name if self._selected_device else None
            stats['buffer_size'] = len(self._audio_buffer)
            return stats
    
    def _detect_audio_devices(self) -> None:
        """Detect available audio input devices."""
        devices = self.get_available_devices()
        
        if not devices:
            raise AudioDeviceError("No audio input devices found")
        
        self._logger.info(f"Found {len(devices)} audio input devices:")
        for device in devices:
            marker = " (default)" if device.is_default else ""
            self._logger.info(f"  {device.device_id}: {device.name}{marker}")
    
    def _select_audio_device(self) -> None:
        """Select the audio input device to use."""
        devices = self.get_available_devices()
        
        if self._config.input_device_id is not None:
            # Use configured device
            device = next((d for d in devices if d.device_id == self._config.input_device_id), None)
            if device:
                self._selected_device = device
                self._logger.info(f"Using configured device: {device.name}")
                return
            else:
                self._logger.warning(f"Configured device {self._config.input_device_id} not found")
        
        # Use default device
        default_device = next((d for d in devices if d.is_default), None)
        if default_device:
            self._selected_device = default_device
            self._logger.info(f"Using default device: {default_device.name}")
        else:
            # Use first available device
            self._selected_device = devices[0]
            self._logger.info(f"Using first available device: {devices[0].name}")
    
    def _capture_loop(self) -> None:
        """Main audio capture loop running in separate thread."""
        self._logger.debug("Audio capture loop started")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Read audio data
                    raw_data = self._stream.read(
                        self._config.input_chunk_size,
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    
                    # Normalize to float32 range [-1, 1]
                    normalized_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Process audio chunk
                    self._process_audio_chunk(normalized_data)
                    
                except Exception as e:
                    self._logger.error(f"Error in capture loop: {e}")
                    with self._stats_lock:
                        self._stats['device_errors'] += 1
                    time.sleep(0.01)  # Brief pause before retry
                    
        except Exception as e:
            self._logger.error(f"Fatal error in capture loop: {e}")
        finally:
            self._logger.debug("Audio capture loop ended")
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        Process a chunk of audio data.
        
        Args:
            audio_data: Normalized audio data [-1, 1]
        """
        timestamp = time.time()
        
        # Perform Voice Activity Detection
        is_speech, rms_level = self._vad.detect_speech(audio_data)
        
        # Create audio data object
        audio_chunk = AudioData(
            data=audio_data,
            sample_rate=self._config.input_sample_rate,
            channels=self._config.input_channels,
            timestamp=timestamp,
            duration_ms=(len(audio_data) / self._config.input_sample_rate) * 1000,
            rms_level=rms_level,
            is_speech=is_speech
        )
        
        # Update statistics
        with self._stats_lock:
            self._stats['total_frames_captured'] += 1
            if is_speech:
                self._stats['speech_frames_detected'] += 1
            self._stats['average_rms_level'] = (
                (self._stats['average_rms_level'] * (self._stats['total_frames_captured'] - 1) + rms_level) 
                / self._stats['total_frames_captured']
            )
        
        # Add to audio buffer
        with self._buffer_lock:
            self._audio_buffer.append(audio_chunk)
        
        # Handle speech state transitions
        self._handle_speech_detection(audio_chunk)
        
        # Publish audio data event
        self._event_bus.publish(EventTypes.AUDIO_DATA_RECEIVED, {
            'timestamp': timestamp,
            'duration_ms': audio_chunk.duration_ms,
            'rms_level': rms_level,
            'is_speech': is_speech,
            'sample_rate': self._config.input_sample_rate
        })
    
    def _handle_speech_detection(self, audio_chunk: AudioData) -> None:
        """
        Handle speech detection state transitions.
        
        Args:
            audio_chunk: Current audio chunk with VAD results
        """
        if audio_chunk.is_speech:
            if not self._is_speech_active:
                # Speech started
                self._is_speech_active = True
                self._speech_start_time = audio_chunk.timestamp
                self._silence_counter = 0
                
                # Clear and start speech buffer
                with self._buffer_lock:
                    self._speech_buffer.clear()
                    self._speech_buffer.append(audio_chunk)
                
                self._logger.debug("Speech detected - starting speech capture")
                
                # Publish speech detection event
                self._event_bus.publish(EventTypes.SPEECH_DETECTED, {
                    'timestamp': audio_chunk.timestamp,
                    'rms_level': audio_chunk.rms_level
                })
            else:
                # Continue speech
                self._silence_counter = 0
                with self._buffer_lock:
                    self._speech_buffer.append(audio_chunk)
        else:
            if self._is_speech_active:
                self._silence_counter += 1
                
                # Add silence to speech buffer (for natural pauses)
                with self._buffer_lock:
                    self._speech_buffer.append(audio_chunk)
                
                # Check if speech has ended
                if self._silence_counter >= self._silence_threshold:
                    self._end_speech_capture()
    
    def _end_speech_capture(self) -> None:
        """End speech capture and publish speech data."""
        if not self._is_speech_active:
            return
        
        self._is_speech_active = False
        speech_duration = time.time() - self._speech_start_time if self._speech_start_time else 0
        
        # Get speech audio data
        with self._buffer_lock:
            speech_chunks = list(self._speech_buffer)
            self._speech_buffer.clear()
        
        if speech_chunks:
            # Combine audio chunks
            combined_audio = np.concatenate([chunk.data for chunk in speech_chunks])
            
            self._logger.debug(f"Speech ended - captured {len(speech_chunks)} chunks, {speech_duration:.2f}s")
            
            # Publish speech ended event
            self._event_bus.publish(EventTypes.SPEECH_ENDED, {
                'timestamp': time.time(),
                'duration_seconds': speech_duration,
                'chunk_count': len(speech_chunks),
                'audio_length': len(combined_audio),
                'sample_rate': self._config.input_sample_rate
            })
        
        self._speech_start_time = None
        self._silence_counter = 0
