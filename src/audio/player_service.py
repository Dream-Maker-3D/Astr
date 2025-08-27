import numpy as np
import pyaudio
import logging
import asyncio
import threading
from typing import Optional, Dict, Any
import time
from queue import Queue, Empty
import tempfile
import os

from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import AudioError, DeviceNotFoundError


class AudioPlayerService:
    """
    Audio playback service with interruption support for natural conversation.
    Implements Template Method pattern for consistent playback workflow.
    """
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self.config = ConfigurationManager()
        self._logger = logging.getLogger(__name__)
        
        # Audio configuration
        self._audio = None
        self._stream = None
        self._is_playing = False
        self._is_initialized = False
        
        # Audio parameters from config
        audio_config = self.config.get_audio_config()
        output_config = audio_config.get('output', {})
        
        self.device_id = output_config.get('device_id')  # None for auto-detect
        self.sample_rate = output_config.get('sample_rate', 22050)
        self.volume = output_config.get('volume', 0.8)
        self.chunk_size = 1024
        
        # Interruption handling
        conversation_config = self.config.get_conversation_config()
        self.interruption_response_ms = conversation_config.get('interruption_response_ms', 50)
        
        # Playback control
        self._playback_thread = None
        self._stop_playback_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._audio_queue = Queue()
        self._current_playback_id = None
        
        # Subscribe to synthesis and interruption events
        self.event_bus.subscribe(EventTypes.SPEECH_SYNTHESIS_COMPLETE, self._handle_synthesis_complete)
        self.event_bus.subscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
        self.event_bus.subscribe(EventTypes.SPEECH_DETECTED, self._handle_speech_detected)
    
    async def initialize(self) -> bool:
        """Initialize audio playback system."""
        try:
            self._audio = pyaudio.PyAudio()
            
            # Find and validate audio device
            if self.device_id is None:
                self.device_id = self._find_default_output_device()
            
            if not self._validate_device(self.device_id):
                raise DeviceNotFoundError(f"Audio output device {self.device_id} not found or invalid")
            
            self._is_initialized = True
            
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_STATUS_CHANGED,
                {'service': 'audio_player', 'status': 'initialized'}
            )
            
            self._logger.info(f"Audio player service initialized (device: {self.device_id}, rate: {self.sample_rate})")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize audio player: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'audio_player', 'error': str(e)}
            )
            return False
    
    def _find_default_output_device(self) -> int:
        """Find the default audio output device."""
        try:
            default_device = self._audio.get_default_output_device_info()
            return default_device['index']
        except Exception as e:
            self._logger.warning(f"Could not find default output device: {e}")
            return 0
    
    def _validate_device(self, device_id: int) -> bool:
        """Validate that the audio device exists and supports required format."""
        try:
            device_info = self._audio.get_device_info_by_index(device_id)
            
            # Check if device supports output
            if device_info['maxOutputChannels'] < 1:
                return False
            
            # Test if format is supported
            return self._audio.is_format_supported(
                rate=self.sample_rate,
                output_device=device_id,
                output_channels=1,
                output_format=pyaudio.paFloat32
            )
            
        except Exception as e:
            self._logger.error(f"Device validation failed: {e}")
            return False
    
    async def _handle_synthesis_complete(self, event_data: Dict[str, Any]) -> None:
        """Handle completed speech synthesis by queuing for playback."""
        try:
            audio_data = event_data['data'].get('audio_data')
            sample_rate = event_data['data'].get('sample_rate', self.sample_rate)
            text = event_data['data'].get('text', '')
            conversation_id = event_data['data'].get('conversation_id')
            
            if audio_data is not None:
                playback_id = f"playback_{time.time()}"
                
                # Queue audio for playback
                self._audio_queue.put({
                    'audio_data': audio_data,
                    'sample_rate': sample_rate,
                    'text': text,
                    'conversation_id': conversation_id,
                    'playback_id': playback_id
                })
                
                # Start playback if not already running
                if not self._is_playing:
                    await self._start_playback_loop()
                    
        except Exception as e:
            self._logger.error(f"Error handling synthesis complete: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'audio_player', 'error': str(e)}
            )
    
    async def _handle_interruption(self, event_data: Dict[str, Any]) -> None:
        """Handle interruption by immediately stopping current playback."""
        if self._is_playing:
            self._interrupt_event.set()
            
            await self.event_bus.publish_async(
                EventTypes.RESPONSE_INTERRUPTED,
                {
                    'playback_id': self._current_playback_id,
                    'timestamp': time.time()
                }
            )
            
            self._logger.debug("Audio playback interrupted")
    
    async def _handle_speech_detected(self, event_data: Dict[str, Any]) -> None:
        """Handle speech detection as potential interruption."""
        if self._is_playing:
            # Immediate interruption on speech detection during playback
            await self._handle_interruption(event_data)
    
    async def _start_playback_loop(self) -> None:
        """Start the main playback loop."""
        if self._playback_thread and self._playback_thread.is_alive():
            return  # Already running
        
        self._stop_playback_event.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
    
    def _playback_loop(self) -> None:
        """Main audio playback loop with interruption support."""
        try:
            while not self._stop_playback_event.is_set():
                try:
                    # Get next audio item from queue
                    audio_item = self._audio_queue.get(timeout=1.0)
                    
                    # Play the audio
                    asyncio.run(self._play_audio_item(audio_item))
                    
                except Empty:
                    # No audio to play, continue waiting
                    continue
                except Exception as e:
                    self._logger.error(f"Playback loop error: {e}")
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.event_bus.publish_async(
                                EventTypes.SYSTEM_ERROR,
                                {'service': 'audio_player', 'error': str(e)}
                            ), asyncio.get_event_loop()
                        )
                    except:
                        pass
                    
        except Exception as e:
            self._logger.error(f"Playback loop fatal error: {e}")
    
    async def _play_audio_item(self, audio_item: Dict[str, Any]) -> None:
        """Play a single audio item with interruption support."""
        try:
            audio_data = audio_item['audio_data']
            sample_rate = audio_item['sample_rate']
            text = audio_item['text']
            conversation_id = audio_item['conversation_id']
            playback_id = audio_item['playback_id']
            
            self._current_playback_id = playback_id
            self._is_playing = True
            self._interrupt_event.clear()
            
            # Publish playback start
            await self.event_bus.publish_async(
                EventTypes.AUDIO_PLAYBACK_START,
                {
                    'text': text,
                    'playback_id': playback_id,
                    'conversation_id': conversation_id
                }
            )
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
            
            # Apply volume
            audio_data = audio_data * self.volume
            
            # Ensure audio is in correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Create output stream
            stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_id,
                frames_per_buffer=self.chunk_size
            )
            
            try:
                # Play audio in chunks with interruption checking
                chunk_samples = self.chunk_size
                total_samples = len(audio_data)
                
                for start_idx in range(0, total_samples, chunk_samples):
                    # Check for interruption
                    if self._interrupt_event.is_set():
                        break
                    
                    # Get audio chunk
                    end_idx = min(start_idx + chunk_samples, total_samples)
                    chunk = audio_data[start_idx:end_idx]
                    
                    # Pad chunk if necessary
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    
                    # Play chunk
                    stream.write(chunk.tobytes())
                
                # Check if playback was interrupted
                if self._interrupt_event.is_set():
                    await self.event_bus.publish_async(
                        EventTypes.RESPONSE_INTERRUPTED,
                        {
                            'playback_id': playback_id,
                            'conversation_id': conversation_id
                        }
                    )
                else:
                    await self.event_bus.publish_async(
                        EventTypes.AUDIO_PLAYBACK_COMPLETE,
                        {
                            'playback_id': playback_id,
                            'conversation_id': conversation_id,
                            'text': text
                        }
                    )
                
            finally:
                stream.stop_stream()
                stream.close()
                self._is_playing = False
                self._current_playback_id = None
                
        except Exception as e:
            self._logger.error(f"Audio playback error: {e}")
            self._is_playing = False
            self._current_playback_id = None
            
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'audio_player', 'error': str(e)}
            )
    
    def _resample_audio(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple audio resampling."""
        if from_rate == to_rate:
            return audio_data
        
        try:
            from scipy.signal import resample
            target_length = int(len(audio_data) * to_rate / from_rate)
            return resample(audio_data, target_length).astype(np.float32)
        except ImportError:
            # Fallback to simple linear interpolation
            ratio = to_rate / from_rate
            target_length = int(len(audio_data) * ratio)
            return np.interp(
                np.linspace(0, len(audio_data) - 1, target_length),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.float32)
    
    async def play_audio_file(self, file_path: str) -> bool:
        """Play audio from file."""
        try:
            import soundfile as sf
            
            audio_data, sample_rate = sf.read(file_path)
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            audio_data = audio_data.astype(np.float32)
            
            # Queue for playback
            playback_id = f"file_playback_{time.time()}"
            self._audio_queue.put({
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'text': f"Playing {os.path.basename(file_path)}",
                'conversation_id': None,
                'playback_id': playback_id
            })
            
            if not self._is_playing:
                await self._start_playback_loop()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to play audio file {file_path}: {e}")
            return False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current audio device."""
        if self._audio and self.device_id is not None:
            try:
                return self._audio.get_device_info_by_index(self.device_id)
            except Exception as e:
                self._logger.error(f"Could not get device info: {e}")
                return {}
        return {}
    
    def list_audio_devices(self) -> list:
        """List all available audio output devices."""
        devices = []
        if self._audio:
            try:
                for i in range(self._audio.get_device_count()):
                    device_info = self._audio.get_device_info_by_index(i)
                    if device_info['maxOutputChannels'] > 0:  # Output device
                        devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxOutputChannels'],
                            'sample_rate': device_info['defaultSampleRate']
                        })
            except Exception as e:
                self._logger.error(f"Could not list audio devices: {e}")
        
        return devices
    
    async def stop_playback(self) -> None:
        """Stop current playback."""
        self._interrupt_event.set()
        
        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_playback()
        
        self._stop_playback_event.set()
        
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
        
        if self._audio:
            self._audio.terminate()
            self._audio = None
        
        self._is_initialized = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(EventTypes.SPEECH_SYNTHESIS_COMPLETE, self._handle_synthesis_complete)
        self.event_bus.unsubscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
        self.event_bus.unsubscribe(EventTypes.SPEECH_DETECTED, self._handle_speech_detected)
        
        await self.event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'audio_player', 'status': 'cleanup_complete'}
        )
        
        self._logger.info("Audio player service cleanup completed")