"""
Voice Assistant Facade - Main System Orchestrator.

This module implements the Facade pattern to provide a unified interface
for the complete voice conversation pipeline, coordinating all services
and managing the system lifecycle.
"""

import os
import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .event_bus import EventBusService
from .config_manager import ConfigurationManager
from .service_factory import ServiceFactory
from .conversation_state import ConversationStateManager, ConversationState
from .interruption_handler import InterruptionHandler
from ..audio.capture_service import AudioCaptureService
from ..audio.player_service import AudioPlayerService
from ..speech.recognition_service import SpeechRecognitionService
from ..speech.synthesis_service import SpeechSynthesisService
from ..ai.conversation_service import AIConversationService
from ..ai.types import AIConfig
from ..utils.exceptions import AstirError

logger = logging.getLogger(__name__)


class VoiceAssistantState:
    """Voice assistant system states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class VoiceAssistantFacade:
    """
    Main Voice Assistant Facade implementing the Facade pattern.
    
    Provides a unified interface for the complete voice conversation pipeline,
    coordinating all services and managing the system lifecycle for seamless
    natural voice conversations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Voice Assistant Facade.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path or "config/default.yaml"
        self.state = VoiceAssistantState.STOPPED
        self.start_time: Optional[datetime] = None
        
        # Core services
        self.config_manager: Optional[ConfigurationManager] = None
        self.event_bus: Optional[EventBusService] = None
        self.service_factory: Optional[ServiceFactory] = None
        self.conversation_manager: Optional[ConversationStateManager] = None
        self.interruption_handler: Optional[InterruptionHandler] = None
        
        # Voice pipeline services
        self.audio_capture: Optional[AudioCaptureService] = None
        self.audio_player: Optional[AudioPlayerService] = None
        self.speech_recognition: Optional[SpeechRecognitionService] = None
        self.speech_synthesis: Optional[SpeechSynthesisService] = None
        self.ai_conversation: Optional[AIConversationService] = None
        
        # System monitoring
        self.conversation_count = 0
        self.error_count = 0
        self.last_activity: Optional[datetime] = None
        
        # Lifecycle management
        self.shutdown_event = threading.Event()
        self.health_monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Voice Assistant Facade created")
    
    def initialize(self) -> bool:
        """
        Initialize the complete voice assistant system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("🚀 Initializing Voice Assistant System...")
            self.state = VoiceAssistantState.INITIALIZING
            self.start_time = datetime.now()
            
            # Step 1: Initialize configuration
            if not self._initialize_configuration():
                return False
            
            # Step 2: Initialize core services
            if not self._initialize_core_services():
                return False
            
            # Step 2.5: Initialize conversation management
            if not self._initialize_conversation_manager():
                return False
            
            # Step 2.6: Initialize interruption handling
            if not self._initialize_interruption_handler():
                return False
            
            # Step 3: Initialize voice pipeline services
            if not self._initialize_voice_services():
                return False
            
            # Step 4: Setup event subscriptions
            self._setup_event_subscriptions()
            
            # Step 5: Start health monitoring
            self._start_health_monitoring()
            
            self.state = VoiceAssistantState.READY
            self.last_activity = datetime.now()
            
            logger.info("✅ Voice Assistant System initialized successfully")
            
            # Publish system ready event
            self.event_bus.publish("SYSTEM_READY", {
                'timestamp': datetime.now().isoformat(),
                'initialization_time': (datetime.now() - self.start_time).total_seconds(),
                'services_count': self._get_active_services_count()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Voice Assistant System: {e}")
            self.state = VoiceAssistantState.ERROR
            return False
    
    def _initialize_configuration(self) -> bool:
        """Initialize configuration management."""
        try:
            logger.info("📋 Initializing configuration...")
            
            # Create configuration manager
            self.config_manager = ConfigurationManager()
            
            # Load configuration if file exists
            if Path(self.config_path).exists():
                config_data = self.config_manager.load_configuration(self.config_path)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("Using default configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            return False
    
    def _initialize_core_services(self) -> bool:
        """Initialize core services (Event Bus, Service Factory)."""
        try:
            logger.info("🔧 Initializing core services...")
            
            # Initialize Event Bus
            self.event_bus = EventBusService()
            if not self.event_bus.initialize():
                logger.error("Failed to initialize Event Bus")
                return False
            
            # Initialize Service Factory
            self.service_factory = ServiceFactory(self.event_bus)
            
            # Note: Service factory registration will be handled by individual services
            # Core services are already initialized and available
            
            logger.info("✅ Core services initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize core services: {e}")
            return False
    
    def _initialize_conversation_manager(self) -> bool:
        """
        Initialize conversation state management.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing conversation state manager...")
            
            # Create conversation state manager
            self.conversation_manager = ConversationStateManager(self.event_bus)
            
            # Register conversation state callbacks for system coordination
            self.conversation_manager.register_state_callback(
                ConversationState.LISTENING,
                self._on_conversation_listening
            )
            
            self.conversation_manager.register_state_callback(
                ConversationState.PROCESSING,
                self._on_conversation_processing
            )
            
            self.conversation_manager.register_state_callback(
                ConversationState.RESPONDING,
                self._on_conversation_responding
            )
            
            logger.info("✅ Conversation state manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation manager: {e}")
            return False
    
    def _initialize_interruption_handler(self) -> bool:
        """
        Initialize interruption and correction handling.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing interruption handler...")
            
            # Create interruption handler
            self.interruption_handler = InterruptionHandler(
                self.event_bus, 
                self.conversation_manager
            )
            
            logger.info("✅ Interruption handler initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize interruption handler: {e}")
            return False
    
    def _initialize_voice_services(self) -> bool:
        """Initialize voice pipeline services."""
        try:
            logger.info("🎤 Initializing voice pipeline services...")
            
            # Get configuration objects
            config_data = self.config_manager.get_config()
            
            # Initialize Audio Capture Service
            self.audio_capture = AudioCaptureService(self.event_bus, config_data.audio)
            if not self.audio_capture.initialize():
                logger.error("Failed to initialize Audio Capture Service")
                return False
            
            # Initialize Audio Player Service
            self.audio_player = AudioPlayerService(self.event_bus, config_data.audio)
            if not self.audio_player.initialize():
                logger.error("Failed to initialize Audio Player Service")
                return False
            
            # Initialize Speech Recognition Service
            self.speech_recognition = SpeechRecognitionService(self.event_bus, config_data.speech)
            if not self.speech_recognition.initialize():
                logger.error("Failed to initialize Speech Recognition Service")
                return False
            
            # Initialize Speech Synthesis Service
            self.speech_synthesis = SpeechSynthesisService(self.event_bus, config_data.speech)
            if not self.speech_synthesis.initialize():
                logger.error("Failed to initialize Speech Synthesis Service")
                return False
            
            # Initialize AI Conversation Service
            ai_config = self._create_ai_config()
            self.ai_conversation = AIConversationService(self.event_bus, ai_config)
            if not self.ai_conversation.initialize():
                logger.error("Failed to initialize AI Conversation Service")
                return False
            
            # Set service references for interruption handler
            if self.interruption_handler:
                self.interruption_handler.set_services(
                    audio_capture=self.audio_capture,
                    audio_player=self.audio_player,
                    speech_recognition=self.speech_recognition,
                    speech_synthesis=self.speech_synthesis,
                    ai_conversation=self.ai_conversation
                )
            
            logger.info("✅ Voice pipeline services initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice services: {e}")
            return False
    
    def _create_ai_config(self) -> AIConfig:
        """Create AI configuration from environment and config."""
        # Get OpenRouter API key from environment
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")
        
        # Create AI configuration
        ai_config = AIConfig(
            api_key=api_key,
            default_model="anthropic/claude-3-5-sonnet",
            max_tokens=4096,
            temperature=0.7,
            streaming_enabled=True,
            system_prompt="""You are a helpful AI assistant engaged in natural conversation. 
Keep responses concise and conversational. Use natural speech patterns with contractions. 
Avoid verbose explanations unless specifically asked. Be ready to be interrupted naturally."""
        )
        
        return ai_config
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for system coordination."""
        logger.info("📡 Setting up event subscriptions...")
        
        # Subscribe to system events
        self.event_bus.subscribe("CONVERSATION_STARTED", self._handle_conversation_started)
        self.event_bus.subscribe("CONVERSATION_ENDED", self._handle_conversation_ended)
        self.event_bus.subscribe("SYSTEM_ERROR", self._handle_system_error)
        
        # Subscribe to pipeline events for state management
        self.event_bus.subscribe("SPEECH_DETECTED", self._handle_speech_detected)
        self.event_bus.subscribe("TRANSCRIPTION_READY", self._handle_transcription_ready)
        self.event_bus.subscribe("AI_PROCESSING_STARTED", self._handle_ai_processing)
        self.event_bus.subscribe("AI_RESPONSE_READY", self._handle_ai_response)
        self.event_bus.subscribe("SYNTHESIS_STARTED", self._handle_synthesis_started)
        self.event_bus.subscribe("SYNTHESIS_COMPLETED", self._handle_synthesis_completed)
        
        logger.info("✅ Event subscriptions configured")
    
    def _handle_conversation_started(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation started event."""
        self.conversation_count += 1
        self.last_activity = datetime.now()
        logger.info(f"Conversation #{self.conversation_count} started")
    
    def _handle_conversation_ended(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation ended event."""
        self.last_activity = datetime.now()
        logger.info(f"Conversation #{self.conversation_count} ended")
    
    def _handle_system_error(self, event_data: Dict[str, Any]) -> None:
        """Handle system error event."""
        self.error_count += 1
        error_type = event_data.get('error_type', 'Unknown')
        error_message = event_data.get('error_message', 'No details')
        logger.error(f"System error #{self.error_count}: {error_type} - {error_message}")
    
    def _handle_speech_detected(self, event_data: Dict[str, Any]) -> None:
        """Handle speech detection event."""
        if self.state == VoiceAssistantState.READY:
            self.state = VoiceAssistantState.LISTENING
            logger.debug("State: READY → LISTENING")
    
    def _handle_transcription_ready(self, event_data: Dict[str, Any]) -> None:
        """Handle transcription ready event."""
        if self.state == VoiceAssistantState.LISTENING:
            self.state = VoiceAssistantState.PROCESSING
            logger.debug("State: LISTENING → PROCESSING")
    
    def _handle_ai_processing(self, event_data: Dict[str, Any]) -> None:
        """Handle AI processing started event."""
        # State already set to PROCESSING
        logger.debug("AI processing started")
    
    def _handle_ai_response(self, event_data: Dict[str, Any]) -> None:
        """Handle AI response ready event."""
        if self.state == VoiceAssistantState.PROCESSING:
            self.state = VoiceAssistantState.RESPONDING
            logger.debug("State: PROCESSING → RESPONDING")
    
    def _handle_synthesis_started(self, event_data: Dict[str, Any]) -> None:
        """Handle synthesis started event."""
        # State already set to RESPONDING
        logger.debug("Speech synthesis started")
    
    def _handle_synthesis_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle synthesis completed event."""
        if self.state == VoiceAssistantState.RESPONDING:
            self.state = VoiceAssistantState.READY
            logger.debug("State: RESPONDING → READY")
    
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_worker,
            name="HealthMonitor",
            daemon=True
        )
        self.health_monitor_thread.start()
        logger.info("✅ Health monitoring started")
    
    def _health_monitor_worker(self) -> None:
        """Health monitoring worker thread."""
        logger.info("Health monitor started")
        
        while not self.shutdown_event.is_set():
            try:
                # Check system health every 30 seconds
                self.shutdown_event.wait(30.0)
                
                if self.shutdown_event.is_set():
                    break
                
                # Perform health checks
                health_status = self.get_system_health()
                
                if not health_status['healthy']:
                    logger.warning(f"System health check failed: {health_status}")
                    
                    # Publish health warning
                    self.event_bus.publish("SYSTEM_HEALTH_WARNING", health_status)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
        
        logger.info("Health monitor stopped")
    
    def start_conversation_mode(self) -> bool:
        """
        Start continuous conversation mode.
        
        Returns:
            bool: True if conversation mode started successfully
        """
        try:
            if self.state != VoiceAssistantState.READY:
                logger.error(f"Cannot start conversation mode from state: {self.state}")
                return False
            
            logger.info("🎤 Starting continuous conversation mode...")
            
            # Start conversation through conversation manager
            if self.conversation_manager and self.conversation_manager.start_conversation():
                # Start audio capture (continuous listening)
                if not self.audio_capture.start_capture():
                    logger.error("Failed to start audio capture")
                    # End conversation if audio capture fails
                    self.conversation_manager.end_conversation()
                    return False
                
                self.state = VoiceAssistantState.LISTENING
                self.conversation_active = True
                self.conversation_count += 1
                self.last_activity = datetime.now()
                
                # Publish conversation mode started
                self.event_bus.publish("CONVERSATION_MODE_STARTED", {
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'continuous',
                    'conversation_id': self.conversation_manager.context.conversation_id
                })
                
                logger.info("✅ Continuous conversation mode started with state management")
                return True
            else:
                logger.error("Failed to start conversation through conversation manager")
                return False
            
        except Exception as e:
            logger.error(f"Failed to start conversation mode: {e}")
            return False
    
    def stop_conversation_mode(self) -> bool:
        """
        Stop continuous conversation mode.
        
        Returns:
            bool: True if conversation mode stopped successfully
        """
        try:
            logger.info("🛑 Stopping conversation mode...")
            
            # End conversation through conversation manager
            if self.conversation_manager:
                self.conversation_manager.end_conversation()
            
            # Stop audio capture
            if self.audio_capture:
                self.audio_capture.stop_capture()
            
            self.state = VoiceAssistantState.READY
            self.conversation_active = False
            
            # Publish conversation mode stopped
            self.event_bus.publish("CONVERSATION_MODE_STOPPED", {
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("✅ Conversation mode stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop conversation mode: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'state': self.state,
            'uptime_seconds': uptime,
            'conversation_count': self.conversation_count,
            'error_count': self.error_count,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'services': {
                'event_bus': self.event_bus._is_running if self.event_bus else False,
                'audio_capture': self.audio_capture._is_initialized if self.audio_capture else False,
                'audio_player': self.audio_player._is_initialized if self.audio_player else False,
                'speech_recognition': self.speech_recognition._is_initialized if self.speech_recognition else False,
                'speech_synthesis': self.speech_synthesis._is_initialized if self.speech_synthesis else False,
                'ai_conversation': self.ai_conversation.is_initialized if self.ai_conversation else False,
                'conversation_manager': self.conversation_manager is not None,
                'interruption_handler': self.interruption_handler is not None
            },
            'conversation': self.conversation_manager.get_conversation_stats() if self.conversation_manager else None,
            'interruption': self.interruption_handler.get_interruption_statistics() if self.interruption_handler else None
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        health_checks = []
        healthy = True
        
        # Check core services
        if not self.event_bus or not self.event_bus._is_running:
            health_checks.append("Event Bus not running")
            healthy = False
        
        # Check voice services
        services = [
            ('Audio Capture', self.audio_capture),
            ('Audio Player', self.audio_player),
            ('Speech Recognition', self.speech_recognition),
            ('Speech Synthesis', self.speech_synthesis),
            ('AI Conversation', self.ai_conversation)
        ]
        
        for service_name, service in services:
            if not service or not getattr(service, 'is_initialized', False):
                health_checks.append(f"{service_name} not initialized")
                healthy = False
        
        # Check for recent errors
        if self.error_count > 10:  # Threshold for concern
            health_checks.append(f"High error count: {self.error_count}")
            healthy = False
        
        return {
            'healthy': healthy,
            'checks': health_checks,
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def _get_active_services_count(self) -> int:
        """Get count of active services."""
        count = 0
        services = [
            self.event_bus,
            self.audio_capture,
            self.audio_player,
            self.speech_recognition,
            self.speech_synthesis,
            self.ai_conversation
        ]
        
        for service in services:
            if service and getattr(service, 'is_initialized', False):
                count += 1
        
        return count
    
    def shutdown(self) -> None:
        """Shutdown the complete voice assistant system."""
        logger.info("🛑 Shutting down Voice Assistant System...")
        self.state = VoiceAssistantState.SHUTTING_DOWN
        
        # Signal shutdown to health monitor
        self.shutdown_event.set()
        
        # Stop conversation mode if active
        if self.state in [VoiceAssistantState.LISTENING, VoiceAssistantState.PROCESSING, VoiceAssistantState.RESPONDING]:
            self.stop_conversation_mode()
        
        # Shutdown services in reverse order
        services_to_shutdown = [
            ('AI Conversation', self.ai_conversation),
            ('Speech Synthesis', self.speech_synthesis),
            ('Speech Recognition', self.speech_recognition),
            ('Audio Player', self.audio_player),
            ('Audio Capture', self.audio_capture),
            ('Event Bus', self.event_bus)
        ]
        
        for service_name, service in services_to_shutdown:
            if service:
                try:
                    logger.info(f"Shutting down {service_name}...")
                    service.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {service_name}: {e}")
        
        # Wait for health monitor to stop
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
        
        self.state = VoiceAssistantState.STOPPED
        
        total_uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"✅ Voice Assistant System shutdown complete (uptime: {total_uptime:.1f}s)")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise AstirError("Failed to initialize Voice Assistant System")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    # Conversation State Callbacks
    
    def _on_conversation_listening(self, old_state: ConversationState, new_state: ConversationState) -> None:
        """Handle transition to listening state."""
        logger.debug("Conversation state: LISTENING - Ready for user input")
        
        # Ensure audio capture is active
        if self.audio_capture and not self.audio_capture.is_capturing:
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                logger.error(f"Failed to start audio capture on listening state: {e}")
    
    def _on_conversation_processing(self, old_state: ConversationState, new_state: ConversationState) -> None:
        """Handle transition to processing state."""
        logger.debug("Conversation state: PROCESSING - Analyzing user input")
        
        # Update system activity
        self.last_activity = datetime.now()
    
    def _on_conversation_responding(self, old_state: ConversationState, new_state: ConversationState) -> None:
        """Handle transition to responding state."""
        logger.debug("Conversation state: RESPONDING - AI generating response")
        
        # Update system activity
        self.last_activity = datetime.now()
