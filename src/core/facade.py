import asyncio
import logging
from typing import Dict, Any, Optional
import signal
import sys
from enum import Enum

from .event_bus import EventBusService, EventTypes
from .config_manager import ConfigurationManager
from ..audio.capture_service import AudioCaptureService
from ..audio.player_service import AudioPlayerService
from ..speech.recognition_service import SpeechRecognitionService
from ..speech.synthesis_service import SpeechSynthesisService
from ..ai.conversation_service import AIConversationService
from ..utils.exceptions import VoiceAssistantError


class SystemStatus(Enum):
    """System status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class VoiceAssistantFacade:
    """
    Main facade class implementing GoF Facade pattern.
    Coordinates all voice assistant subsystems and provides simplified interface.
    """
    
    def __init__(self, config_path: str = None):
        # Initialize logging first
        self._setup_logging()
        self._logger = logging.getLogger(__name__)
        
        # System state
        self._status = SystemStatus.UNINITIALIZED
        self._is_conversation_active = False
        
        # Core services (Singleton pattern)
        self.config = ConfigurationManager()
        if config_path:
            self.config.load_config(config_path)
        else:
            self.config.load_config()
        
        self.event_bus = EventBusService()
        
        # Initialize services
        self.audio_capture = AudioCaptureService(self.event_bus)
        self.audio_player = AudioPlayerService(self.event_bus)
        self.speech_recognition = SpeechRecognitionService(self.event_bus)
        self.speech_synthesis = SpeechSynthesisService(self.event_bus)
        self.ai_conversation = AIConversationService(self.event_bus)
        
        # Subscribe to system events for coordination
        self.event_bus.subscribe(EventTypes.SYSTEM_ERROR, self._handle_system_error)
        self.event_bus.subscribe(EventTypes.CONVERSATION_STARTED, self._handle_conversation_started)
        self.event_bus.subscribe(EventTypes.CONVERSATION_ENDED, self._handle_conversation_ended)
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self._logger.info("VoiceAssistantFacade initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('voice_assistant.log')
            ]
        )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self._logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            self._logger.warning(f"Could not setup signal handlers: {e}")
    
    async def initialize(self) -> bool:
        """
        Initialize all subsystems following Template Method pattern.
        Returns True if successful, False otherwise.
        """
        self._status = SystemStatus.INITIALIZING
        self._logger.info("Initializing voice assistant system...")
        
        try:
            # Validate configuration
            self.config.validate_required_config()
            
            # Initialize services in dependency order
            init_results = await asyncio.gather(
                self._initialize_event_system(),
                self._initialize_audio_systems(),
                self._initialize_speech_systems(),
                self._initialize_ai_system(),
                return_exceptions=True
            )
            
            # Check if all initializations succeeded
            for i, result in enumerate(init_results):
                if isinstance(result, Exception):
                    self._logger.error(f"Service initialization {i} failed: {result}")
                    self._status = SystemStatus.ERROR
                    return False
                elif not result:
                    self._logger.error(f"Service initialization {i} returned False")
                    self._status = SystemStatus.ERROR
                    return False
            
            # Perform health check
            if await self._perform_health_check():
                self._status = SystemStatus.READY
                
                await self.event_bus.publish_async(
                    EventTypes.SYSTEM_STATUS_CHANGED,
                    {'status': 'ready', 'all_services_initialized': True}
                )
                
                self._logger.info("Voice assistant system initialized successfully")
                return True
            else:
                self._status = SystemStatus.ERROR
                return False
                
        except Exception as e:
            self._logger.error(f"System initialization failed: {e}")
            self._status = SystemStatus.ERROR
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_ERROR,
                {'service': 'system', 'error': str(e), 'phase': 'initialization'}
            )
            return False
    
    async def _initialize_event_system(self) -> bool:
        """Initialize event system."""
        # Event bus is already initialized
        return True
    
    async def _initialize_audio_systems(self) -> bool:
        """Initialize audio capture and playback systems."""
        capture_init = await self.audio_capture.initialize()
        player_init = await self.audio_player.initialize()
        return capture_init and player_init
    
    async def _initialize_speech_systems(self) -> bool:
        """Initialize speech recognition and synthesis systems."""
        recognition_init = await self.speech_recognition.initialize()
        synthesis_init = await self.speech_synthesis.initialize()
        return recognition_init and synthesis_init
    
    async def _initialize_ai_system(self) -> bool:
        """Initialize AI conversation system."""
        return await self.ai_conversation.initialize()
    
    async def _perform_health_check(self) -> bool:
        """Perform system health check."""
        self._logger.info("Performing system health check...")
        
        # Check all services are ready
        services_ready = [
            self.audio_capture.is_recording if hasattr(self.audio_capture, 'is_recording') else True,
            self.speech_recognition.is_ready(),
            self.speech_synthesis.is_ready(),
            self.ai_conversation.is_ready()
        ]
        
        if not all(services_ready):
            self._logger.error("Health check failed: some services not ready")
            return False
        
        # Test AI connection
        if not await self.ai_conversation.test_connection():
            self._logger.error("Health check failed: AI service connection test failed")
            return False
        
        self._logger.info("System health check passed")
        return True
    
    async def start_conversation(self) -> bool:
        """
        Start natural conversation mode.
        Returns True if successful, False otherwise.
        """
        if self._status != SystemStatus.READY:
            raise VoiceAssistantError(f"Cannot start conversation, system status: {self._status}")
        
        try:
            self._status = SystemStatus.ACTIVE
            
            # Start continuous audio capture
            if await self.audio_capture.start_continuous_listening():
                # Start new AI conversation
                await self.ai_conversation.start_new_conversation()
                
                self._is_conversation_active = True
                
                await self.event_bus.publish_async(
                    EventTypes.SYSTEM_STATUS_CHANGED,
                    {'status': 'active', 'conversation_mode': 'natural'}
                )
                
                self._logger.info("Natural conversation mode started")
                return True
            else:
                self._status = SystemStatus.ERROR
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to start conversation: {e}")
            self._status = SystemStatus.ERROR
            return False
    
    async def stop_conversation(self) -> bool:
        """
        Stop conversation mode.
        Returns True if successful, False otherwise.
        """
        try:
            self._is_conversation_active = False
            
            # Stop audio capture
            await self.audio_capture.stop_capture()
            
            # Stop any ongoing playback
            await self.audio_player.stop_playback()
            
            # End AI conversation
            await self.ai_conversation.end_conversation()
            
            self._status = SystemStatus.READY
            
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_STATUS_CHANGED,
                {'status': 'ready', 'conversation_mode': 'stopped'}
            )
            
            self._logger.info("Conversation mode stopped")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to stop conversation: {e}")
            return False
    
    async def handle_voice_input(self) -> None:
        """
        Handle voice input processing.
        This method is called automatically by the event system.
        """
        # Voice input is handled automatically through event system
        # This method exists for interface completeness
        pass
    
    async def _handle_system_error(self, event_data: Dict[str, Any]) -> None:
        """Handle system errors."""
        error_info = event_data.get('data', {})
        service = error_info.get('service', 'unknown')
        error = error_info.get('error', 'unknown error')
        
        self._logger.error(f"System error in {service}: {error}")
        
        # Implement error recovery logic here if needed
        if self._is_conversation_active:
            # For critical errors, we might want to restart conversation
            pass
    
    async def _handle_conversation_started(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation started events."""
        conversation_id = event_data.get('data', {}).get('conversation_id')
        self._logger.debug(f"Conversation started: {conversation_id}")
    
    async def _handle_conversation_ended(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation ended events."""
        conversation_id = event_data.get('data', {}).get('conversation_id')
        self._logger.debug(f"Conversation ended: {conversation_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        Returns dictionary with status information.
        """
        return {
            'system_status': self._status.value,
            'conversation_active': self._is_conversation_active,
            'services': {
                'audio_capture': {
                    'initialized': hasattr(self.audio_capture, '_is_initialized') and self.audio_capture._is_initialized,
                    'recording': getattr(self.audio_capture, '_is_recording', False)
                },
                'audio_player': {
                    'initialized': hasattr(self.audio_player, '_is_initialized') and self.audio_player._is_initialized,
                    'playing': self.audio_player.is_playing()
                },
                'speech_recognition': {
                    'initialized': self.speech_recognition._is_initialized,
                    'ready': self.speech_recognition.is_ready()
                },
                'speech_synthesis': {
                    'initialized': self.speech_synthesis._is_initialized,
                    'ready': self.speech_synthesis.is_ready()
                },
                'ai_conversation': {
                    'initialized': self.ai_conversation._is_initialized,
                    'ready': self.ai_conversation.is_ready(),
                    'processing': self.ai_conversation.is_processing()
                }
            },
            'conversation': self.ai_conversation.get_conversation_stats(),
            'model_info': self.ai_conversation.get_model_info()
        }
    
    def is_ready(self) -> bool:
        """Check if system is ready for conversation."""
        return self._status in [SystemStatus.READY, SystemStatus.ACTIVE]
    
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self._is_conversation_active
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the voice assistant system.
        """
        if self._status == SystemStatus.SHUTTING_DOWN:
            return  # Already shutting down
        
        self._status = SystemStatus.SHUTTING_DOWN
        self._logger.info("Initiating system shutdown...")
        
        try:
            # Stop conversation if active
            if self._is_conversation_active:
                await self.stop_conversation()
            
            # Cleanup services in reverse dependency order
            cleanup_tasks = [
                self.ai_conversation.cleanup(),
                self.speech_synthesis.cleanup(),
                self.speech_recognition.cleanup(),
                self.audio_player.cleanup(),
                self.audio_capture.cleanup()
            ]
            
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Final cleanup
            self.event_bus.clear_history()
            
            self._status = SystemStatus.SHUTDOWN
            
            await self.event_bus.publish_async(
                EventTypes.SYSTEM_STATUS_CHANGED,
                {'status': 'shutdown', 'message': 'System shutdown complete'}
            )
            
            self._logger.info("Voice assistant system shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")
            self._status = SystemStatus.ERROR
    
    def get_event_history(self, limit: int = 50) -> list:
        """Get recent system event history."""
        return self.event_bus.get_event_history(limit=limit)
    
    def get_conversation_history(self) -> list:
        """Get conversation history."""
        return self.ai_conversation.get_conversation_history()
    
    async def send_test_message(self, message: str) -> Optional[str]:
        """Send a test message to AI (for testing purposes)."""
        try:
            response = await self.ai_conversation.send_message_direct(message)
            return response.get('text') if response else None
        except Exception as e:
            self._logger.error(f"Test message failed: {e}")
            return None