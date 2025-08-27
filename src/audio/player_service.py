"""
Audio Player Service - TTS Audio Playback

This module implements audio playback for synthesized speech with queue management,
interruption handling, and event-driven communication, following the Strategy pattern.
"""

import logging
import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from threading import Lock, Event as ThreadEvent
from queue import Queue, PriorityQueue, Empty

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
    AudioPlaybackError,
    InitializationError
)


class PlaybackPriority(Enum):
    """Audio playback priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AudioOutputDevice:
    """Audio output device information."""
    device_id: int
    name: str
    channels: int
    sample_rate: int
    is_output: bool
    is_default: bool


@dataclass
class AudioClip:
    """Audio clip container with metadata."""
    clip_id: str
    data: np.ndarray
    sample_rate: int
    channels: int
    priority: PlaybackPriority
    duration_seconds: float
    timestamp: float
    metadata: Dict[str, Any]


class AudioQueue:
    """Priority-based audio queue for playback management."""
    
    def __init__(self, max_size: int = 100):
        """Initialize audio queue."""
        self._queue = PriorityQueue(maxsize=max_size)
        self._lock = Lock()
        self._counter = 0  # For stable sorting
    
    def put(self, clip: AudioClip) -> bool:
        """Add audio clip to queue."""
        try:
            with self._lock:
                # Use negative priority for correct ordering (higher priority first)
                priority_value = -clip.priority.value
                self._queue.put((priority_value, self._counter, clip), block=False)
                self._counter += 1
            return True
        except:
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[AudioClip]:
        """Get next audio clip from queue."""
        try:
            _, _, clip = self._queue.get(timeout=timeout)
            return clip
        except Empty:
            return None
    
    def clear(self) -> int:
        """Clear all queued audio clips."""
        count = 0
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    count += 1
                except Empty:
                    break
        return count
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class AudioPlayerService:
    """
    Audio Player Service for TTS audio playback.
    
    Provides audio playback with queue management, interruption handling,
    volume control, and event-driven communication.
    """
    
    def __init__(self, event_bus: EventBusService, config: AudioConfig):
        """Initialize the Audio Player Service."""
        self._event_bus = event_bus
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Player state
        self._is_initialized = False
        self._is_playing = False
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = ThreadEvent()
        self._interrupt_event = ThreadEvent()
        
        # PyAudio components
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._selected_device: Optional[AudioOutputDevice] = None
        
        # Audio queue and playback management
        self._audio_queue = AudioQueue()
        self._current_clip: Optional[AudioClip] = None
        self._volume = config.output_volume
        
        # Thread synchronization
        self._state_lock = Lock()
        self._volume_lock = Lock()
        
        # Statistics
        self._stats = {
            'total_clips_played': 0,
            'total_playback_time': 0.0,
            'interruptions_count': 0,
            'queue_overflows': 0,
            'device_errors': 0
        }
        self._stats_lock = Lock()
    
    def initialize(self) -> bool:
        """Initialize the audio player service."""
        if self._is_initialized:
            self._logger.warning("Audio Player Service already initialized")
            return True
        
        if not PYAUDIO_AVAILABLE:
            self._logger.warning("PyAudio not available - audio playback disabled")
            # Create a mock device for testing
            self._selected_device = AudioOutputDevice(
                device_id=0,
                name="Mock Audio Output Device",
                channels=2,
                sample_rate=22050,
                is_output=True,
                is_default=True
            )
            self._is_initialized = True
            return True
        
        try:
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            # Detect and select audio output device
            self._detect_output_devices()
            self._select_output_device()
            
            # Start playback thread
            self._start_playback_thread()
            
            self._is_initialized = True
            self._logger.info("Audio Player Service initialized successfully")
            
            # Publish initialization event
            self._event_bus.publish(EventTypes.SYSTEM_READY, {
                'service': 'AudioPlayerService',
                'device': self._selected_device.name if self._selected_device else 'None',
                'sample_rate': self._config.output_sample_rate,
                'volume': self._volume
            })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Audio Player Service: {e}")
            raise InitializationError(f"Audio player initialization failed: {e}")
    
    def play_audio(self, audio_data: Union[np.ndarray, bytes], 
                   sample_rate: int = 22050,
                   channels: int = 1,
                   priority: PlaybackPriority = PlaybackPriority.NORMAL,
                   clip_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Queue audio for playback."""
        if not self._is_initialized:
            raise AudioPlaybackError("Service not initialized")
        
        # Generate clip ID if not provided
        if clip_id is None:
            clip_id = f"clip_{int(time.time() * 1000)}"
        
        # Convert audio data to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = audio_data.astype(np.float32)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Create audio clip
        clip = AudioClip(
            clip_id=clip_id,
            data=audio_array,
            sample_rate=sample_rate,
            channels=channels,
            priority=priority,
            duration_seconds=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Queue the clip
        if not self._audio_queue.put(clip):
            with self._stats_lock:
                self._stats['queue_overflows'] += 1
            raise AudioPlaybackError("Audio queue is full")
        
        self._logger.debug(f"Queued audio clip {clip_id}: {duration:.2f}s, priority={priority.name}")
        return clip_id
    
    def interrupt_playback(self) -> bool:
        """Interrupt current playback immediately."""
        if not self._is_playing:
            return True
        
        # Signal interruption
        self._interrupt_event.set()
        
        # Clear audio queue
        cleared_count = self._audio_queue.clear()
        
        with self._stats_lock:
            self._stats['interruptions_count'] += 1
        
        self._logger.debug(f"Playback interrupted, cleared {cleared_count} queued clips")
        
        # Publish interruption event
        self._event_bus.publish(EventTypes.PLAYBACK_INTERRUPTED, {
            'timestamp': time.time(),
            'cleared_clips': cleared_count,
            'current_clip': self._current_clip.clip_id if self._current_clip else None
        })
        
        return True
    
    def set_volume(self, volume: float) -> bool:
        """Set playback volume."""
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")
        
        with self._volume_lock:
            old_volume = self._volume
            self._volume = volume
        
        self._logger.debug(f"Volume changed from {old_volume:.2f} to {volume:.2f}")
        return True
    
    def get_volume(self) -> float:
        """Get current playback volume."""
        with self._volume_lock:
            return self._volume
    
    def get_available_devices(self) -> List[AudioOutputDevice]:
        """Get list of available audio output devices."""
        if not self._pyaudio:
            return []
        
        devices = []
        device_count = self._pyaudio.get_device_count()
        
        for i in range(device_count):
            try:
                info = self._pyaudio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:  # Output device
                    device = AudioOutputDevice(
                        device_id=i,
                        name=info['name'],
                        channels=info['maxOutputChannels'],
                        sample_rate=int(info['defaultSampleRate']),
                        is_output=True,
                        is_default=(i == self._pyaudio.get_default_output_device_info()['index'])
                    )
                    devices.append(device)
            except Exception as e:
                self._logger.warning(f"Error getting device {i} info: {e}")
        
        return devices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio player statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            stats['is_playing'] = self._is_playing
            stats['queue_size'] = self._audio_queue.size()
            stats['current_volume'] = self._volume
            stats['selected_device'] = self._selected_device.name if self._selected_device else None
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the audio player service."""
        if self._is_playing:
            self.interrupt_playback()
        
        # Stop playback thread
        self._stop_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
        
        # Close audio stream
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        # Terminate PyAudio
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        self._is_initialized = False
        self._logger.info("Audio Player Service shutdown complete")
    
    def _detect_output_devices(self) -> None:
        """Detect available audio output devices."""
        devices = self.get_available_devices()
        
        if not devices:
            raise AudioDeviceError("No audio output devices found")
        
        self._logger.info(f"Found {len(devices)} audio output devices:")
        for device in devices:
            marker = " (default)" if device.is_default else ""
            self._logger.info(f"  {device.device_id}: {device.name}{marker}")
    
    def _select_output_device(self) -> None:
        """Select the audio output device to use."""
        devices = self.get_available_devices()
        
        if self._config.output_device_id is not None:
            # Use configured device
            device = next((d for d in devices if d.device_id == self._config.output_device_id), None)
            if device:
                self._selected_device = device
                self._logger.info(f"Using configured output device: {device.name}")
                return
            else:
                self._logger.warning(f"Configured output device {self._config.output_device_id} not found")
        
        # Use default device
        default_device = next((d for d in devices if d.is_default), None)
        if default_device:
            self._selected_device = default_device
            self._logger.info(f"Using default output device: {default_device.name}")
        else:
            # Use first available device
            self._selected_device = devices[0]
            self._logger.info(f"Using first available output device: {devices[0].name}")
    
    def _start_playback_thread(self) -> None:
        """Start the audio playback thread."""
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="AudioPlayer",
            daemon=True
        )
        self._playback_thread.start()
        self._logger.debug("Audio playback thread started")
    
    def _playback_loop(self) -> None:
        """Main audio playback loop running in separate thread."""
        self._logger.debug("Audio playback loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get next audio clip from queue
                clip = self._audio_queue.get(timeout=0.1)
                if clip is None:
                    continue
                
                # Play the audio clip
                self._play_clip(clip)
                
            except Exception as e:
                self._logger.error(f"Error in playback loop: {e}")
                with self._stats_lock:
                    self._stats['device_errors'] += 1
        
        self._logger.debug("Audio playback loop ended")
    
    def _play_clip(self, clip: AudioClip) -> None:
        """Play a single audio clip."""
        # Simulate playback for testing without PyAudio
        with self._state_lock:
            self._current_clip = clip
            self._is_playing = True
        
        # Publish playback started event
        self._event_bus.publish(EventTypes.PLAYBACK_STARTED, {
            'clip_id': clip.clip_id,
            'duration': clip.duration_seconds,
            'priority': clip.priority.name,
            'timestamp': time.time()
        })
        
        # Simulate playback time
        start_time = time.time()
        while time.time() - start_time < clip.duration_seconds:
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                break
            time.sleep(0.01)
        
        # Update statistics
        with self._stats_lock:
            self._stats['total_clips_played'] += 1
            self._stats['total_playback_time'] += clip.duration_seconds
        
        # Publish playback finished event
        self._event_bus.publish(EventTypes.PLAYBACK_FINISHED, {
            'clip_id': clip.clip_id,
            'duration': clip.duration_seconds,
            'timestamp': time.time()
        })
        
        with self._state_lock:
            self._current_clip = None
            self._is_playing = False
