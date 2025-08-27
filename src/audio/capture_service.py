import numpy as np
import pyaudio
import logging
import asyncio
import threading
from typing import Optional, Callable, Dict, Any
from queue import Queue, Empty
import time

from ..core.event_bus import EventBusService, EventTypes
from ..core.config_manager import ConfigurationManager
from ..utils.exceptions import AudioError, DeviceNotFoundError


class AudioCaptureService:
    """
    Audio capture service for continuous microphone input.
    Implements continuous listening for natural conversation flow.
    """
    
    def __init__(self, event_bus: EventBusService):
        self.event_bus = event_bus
        self.config = ConfigurationManager()
        self._logger = logging.getLogger(__name__)
        
        # Audio configuration
        self._audio = None
        self._stream = None
        self._is_recording = False
        self._is_initialized = False
        
        # Audio parameters from config
        audio_config = self.config.get_audio_config()
        input_config = audio_config.get('input', {})
        
        self.device_id = input_config.get('device_id')  # None for auto-detect
        self.sample_rate = input_config.get('sample_rate', 16000)
        self.channels = input_config.get('channels', 1)
        self.chunk_size = input_config.get('chunk_size', 1024)
        self.buffer_size = input_config.get('buffer_size', 4096)
        self.continuous_listening = input_config.get('continuous_listening', True)
        
        # Voice Activity Detection parameters
        conversation_config = self.config.get_conversation_config()
        self.vad_threshold = conversation_config.get('voice_activity_threshold', 0.015)
        self.turn_taking_pause_ms = conversation_config.get('turn_taking_pause_ms', 1500)
        
        # Audio processing
        self._audio_buffer = Queue()
        self._capture_thread = None
        self._processing_thread = None
        self._stop_event = threading.Event()
        self._main_loop = None  # Store main event loop
        
        # VAD state
        self._is_speech_detected = False
        self._last_speech_time = 0
        self._silence_start = 0
        
        # Subscribe to system events
        self.event_bus.subscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
    
    async def initialize(self) -> bool:
        """Initialize audio capture system."""
        try:
            self._audio = pyaudio.PyAudio()
            
            # Find and validate audio device
            if self.device_id is None:
                self.device_id = self._find_default_input_device()
            
            if not self._validate_device(self.device_id):
                raise DeviceNotFoundError(f"Audio input device {self.device_id} not found or invalid")
            
            self._is_initialized = True
            
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_STATUS_CHANGED,
                {'service': 'audio_capture', 'status': 'initialized'}
            )
            
            self._logger.info(f"Audio capture service initialized (device: {self.device_id}, rate: {self.sample_rate})")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize audio capture: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'audio_capture', 'error': str(e)}
            )
            return False
    
    def _find_default_input_device(self) -> int:
        """Find the default audio input device."""
        try:
            default_device = self._audio.get_default_input_device_info()
            return default_device['index']
        except Exception as e:
            self._logger.warning(f"Could not find default input device: {e}")
            # Return device 0 as fallback
            return 0
    
    def _validate_device(self, device_id: int) -> bool:
        """Validate that the audio device exists and supports required format."""
        try:
            device_info = self._audio.get_device_info_by_index(device_id)
            
            # Check if device supports input
            if device_info['maxInputChannels'] < self.channels:
                return False
            
            # Test if format is supported
            return self._audio.is_format_supported(
                rate=self.sample_rate,
                input_device=device_id,
                input_channels=self.channels,
                input_format=pyaudio.paFloat32
            )
            
        except Exception as e:
            self._logger.error(f"Device validation failed: {e}")
            return False
    
    async def start_continuous_listening(self) -> bool:
        """Start continuous audio capture for natural conversation."""
        if not self._is_initialized:
            raise AudioError("Audio capture service not initialized")
        
        if self._is_recording:
            self._logger.warning("Audio capture already running")
            return True
        
        try:
            # Create audio stream
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=None  # Use blocking mode for better control
            )
            
            self._is_recording = True
            self._stop_event.clear()
            
            # Store the main event loop for thread communication
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._main_loop = asyncio.get_event_loop()
            
            # Start capture and processing threads
            self._capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self._processing_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            
            self._capture_thread.start()
            self._processing_thread.start()
            
            await self.event_bus.publish_async(
                EventTypes.CONVERSATION_STARTED,
                {'service': 'audio_capture', 'mode': 'continuous'}
            )
            
            self._logger.info("Continuous audio capture started")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start audio capture: {e}")
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'audio_capture', 'error': str(e)}
            )
            return False
    
    def _audio_capture_loop(self) -> None:
        """Main audio capture loop running in separate thread."""
        try:
            while not self._stop_event.is_set() and self._is_recording:
                if self._stream and self._stream.is_active():
                    # Read audio data
                    audio_data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Add to buffer for processing
                    self._audio_buffer.put({
                        'data': audio_array,
                        'timestamp': time.time()
                    })
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            self._logger.error(f"Audio capture loop error: {e}")
            try:
                if self._main_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.event_bus.publish_async(
                            EventTypes.SYSTEM_ERROR,
                            {'service': 'audio_capture', 'error': str(e)}
                        ), self._main_loop
                    ).result(timeout=0.1)
            except:
                pass
    
    def _audio_processing_loop(self) -> None:
        """Audio processing loop with Voice Activity Detection."""
        accumulated_audio = []
        
        try:
            while not self._stop_event.is_set() and self._is_recording:
                try:
                    # Get audio chunk with timeout
                    try:
                        audio_chunk = self._audio_buffer.get(timeout=0.1)
                    except Empty:
                        continue
                    audio_data = audio_chunk['data']
                    timestamp = audio_chunk['timestamp']
                    
                    # Perform Voice Activity Detection
                    is_speech = self._detect_voice_activity(audio_data)
                    
                    if is_speech:
                        if not self._is_speech_detected:
                            # Speech started
                            self._is_speech_detected = True
                            self._last_speech_time = timestamp
                            accumulated_audio = [audio_data]  # Start new accumulation
                            self._logger.info(f"Speech detected! Starting accumulation...")
                            
                            try:
                                if self._main_loop:
                                    asyncio.run_coroutine_threadsafe(
                                        self.event_bus.publish_async(
                                            EventTypes.SPEECH_DETECTED,
                                            {'timestamp': timestamp}
                                        ), self._main_loop
                                    ).result(timeout=0.1)
                            except:
                                pass
                        else:
                            # Continue speech
                            accumulated_audio.append(audio_data)
                            self._last_speech_time = timestamp
                    
                    else:  # No speech detected
                        if self._is_speech_detected:
                            # Check if we should end speech segment
                            silence_duration = (timestamp - self._last_speech_time) * 1000  # Convert to ms
                            
                            if silence_duration >= self.turn_taking_pause_ms:
                                # End of speech segment - process accumulated audio
                                if accumulated_audio:
                                    complete_audio = np.concatenate(accumulated_audio)
                                    self._logger.info(f"Sending audio for recognition: {len(complete_audio)} samples ({len(complete_audio)/self.sample_rate:.2f}s)")
                                    
                                    if not self._main_loop:
                                        continue
                                    asyncio.run_coroutine_threadsafe(
                                        self.event_bus.publish_async(
                                            EventTypes.AUDIO_DATA_RECEIVED,
                                            {
                                                'audio_data': complete_audio,
                                                'sample_rate': self.sample_rate,
                                                'timestamp': timestamp
                                            }
                                        ), self._main_loop
                                    ).result(timeout=0.1)
                                    
                                    # Also publish turn boundary detection
                                    asyncio.run_coroutine_threadsafe(
                                        self.event_bus.publish_async(
                                            EventTypes.TURN_BOUNDARY_DETECTED,
                                            {
                                                'audio_length': len(complete_audio) / self.sample_rate,
                                                'timestamp': timestamp
                                            }
                                        ), self._main_loop
                                    ).result(timeout=0.1)
                                
                                self._is_speech_detected = False
                                accumulated_audio.clear()  # Clear the accumulated audio
                        
                        # No action needed for non-speech audio
                        pass
                
                except Exception as e:
                    if not self._stop_event.is_set():
                        import traceback
                        self._logger.error(f"Audio processing error: {e}\n{traceback.format_exc()}")
                        continue
                    
        except Exception as e:
            self._logger.error(f"Audio processing loop error: {e}")
            try:
                if self._main_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.event_bus.publish_async(
                            EventTypes.SYSTEM_ERROR,
                            {'service': 'audio_capture', 'error': str(e)}
                        ), self._main_loop
                    ).result(timeout=0.1)
            except:
                pass
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Simple Voice Activity Detection based on energy threshold."""
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Simple threshold-based VAD
        return rms > self.vad_threshold
    
    async def _handle_interruption(self, event_data: Dict[str, Any]) -> None:
        """Handle interruption detection by immediately processing current audio."""
        if self._is_speech_detected:
            # Force end current speech segment for immediate processing
            self._is_speech_detected = False
            self._logger.debug("Interruption detected - processing current audio segment")
    
    async def stop_capture(self) -> None:
        """Stop audio capture."""
        self._is_recording = False
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        
        # Close audio stream
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        await self.event_bus.publish_async(
            EventTypes.CONVERSATION_ENDED,
            {'service': 'audio_capture'}
        )
        
        self._logger.info("Audio capture stopped")
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
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
        """List all available audio input devices."""
        devices = []
        if self._audio:
            try:
                for i in range(self._audio.get_device_count()):
                    device_info = self._audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:  # Input device
                        devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': device_info['defaultSampleRate']
                        })
            except Exception as e:
                self._logger.error(f"Could not list audio devices: {e}")
        
        return devices
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_capture()
        
        if self._audio:
            self._audio.terminate()
            self._audio = None
        
        self._is_initialized = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(EventTypes.INTERRUPTION_DETECTED, self._handle_interruption)
        
        await self.event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'audio_capture', 'status': 'cleanup_complete'}
        )
        
        self._logger.info("Audio capture service cleanup completed")