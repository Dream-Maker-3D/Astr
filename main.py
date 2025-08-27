#!/usr/bin/env python3
"""
Astir Voice Assistant - Main Entry Point

A natural conversation voice assistant built with Python, OpenRouter AI,
Whisper STT, and Coqui TTS using Gang of Four design patterns and BDD methodology.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.event_bus import EventBusService, EventTypes
from src.core.config_manager import config_manager
from src.audio.capture_service import AudioCaptureService
from src.audio.player_service import AudioPlayerService, PlaybackPriority
from src.utils.exceptions import AstirError, InitializationError


def setup_logging():
    """Set up structured logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('voice_assistant.log')
        ]
    )
    return logging.getLogger(__name__)


def check_environment():
    """Check that required environment variables are set."""
    required_vars = ['OPENROUTER_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise InitializationError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def main():
    """Main application entry point."""
    logger = setup_logging()
    logger.info("🚀 Starting Astir Voice Assistant...")
    
    try:
        # Check environment
        check_environment()
        logger.info("✅ Environment check passed")
        
        # Load configuration
        try:
            config = config_manager.load_configuration()
            logger.info("✅ Configuration loaded successfully")
            logger.info(f"   - AI Model: {config.ai.model}")
            logger.info(f"   - Log Level: {config.system.log_level}")
        except Exception as e:
            logger.warning(f"⚠️  Configuration loading failed, using defaults: {e}")
            # Create minimal default configuration for testing
            from src.core.config_manager import Configuration
            config = Configuration()
        
        # Initialize Event Bus
        event_bus = EventBusService()
        event_bus.initialize()
        logger.info("✅ Event Bus initialized")
        
        # Initialize Audio Services
        audio_capture = AudioCaptureService(event_bus, config.audio)
        audio_capture.initialize()
        logger.info("✅ Audio Capture Service initialized")
        
        audio_player = AudioPlayerService(event_bus, config.audio)
        audio_player.initialize()
        logger.info("✅ Audio Player Service initialized")
        
        # Set up event handlers
        def system_ready_handler(data):
            logger.info(f"🎉 System ready: {data}")
        
        def system_shutdown_handler(data):
            logger.info(f"🛑 System shutdown: {data}")
        
        def audio_data_handler(data):
            if data.get('status') == 'capture_started':
                logger.info(f"🎤 Audio capture started: {data.get('device', 'unknown')}")
        
        def speech_detected_handler(data):
            logger.info(f"🗣️  Speech detected: RMS={data.get('rms_level', 0):.4f}")
        
        def speech_ended_handler(data):
            duration = data.get('duration_seconds', 0)
            logger.info(f"🔇 Speech ended: {duration:.2f}s")
            
            # Demo: Play a test response when speech ends
            import numpy as np
            sample_rate = 22050
            duration = 0.5  # 0.5 second beep
            frequency = 800  # Higher frequency beep
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep_audio = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            try:
                audio_player.play_audio(
                    audio_data=beep_audio,
                    sample_rate=sample_rate,
                    priority=PlaybackPriority.HIGH,
                    clip_id="response_beep"
                )
                logger.info("🔊 Playing response beep")
            except Exception as e:
                logger.warning(f"Could not play response beep: {e}")
        
        def playback_started_handler(data):
            clip_id = data.get('clip_id', 'unknown')
            duration = data.get('duration', 0)
            logger.info(f"🔊 Playback started: {clip_id} ({duration:.2f}s)")
        
        def playback_finished_handler(data):
            clip_id = data.get('clip_id', 'unknown')
            logger.info(f"🔇 Playback finished: {clip_id}")
        
        # Subscribe to events
        event_bus.subscribe(EventTypes.SYSTEM_READY, system_ready_handler)
        event_bus.subscribe(EventTypes.SYSTEM_SHUTDOWN, system_shutdown_handler)
        event_bus.subscribe(EventTypes.AUDIO_DATA_RECEIVED, audio_data_handler)
        event_bus.subscribe(EventTypes.SPEECH_DETECTED, speech_detected_handler)
        event_bus.subscribe(EventTypes.SPEECH_ENDED, speech_ended_handler)
        event_bus.subscribe(EventTypes.PLAYBACK_STARTED, playback_started_handler)
        event_bus.subscribe(EventTypes.PLAYBACK_FINISHED, playback_finished_handler)
        
        logger.info("✅ Event handlers registered")
        
        # TODO: Initialize other services (Audio, Speech, AI, etc.)
        # This will be implemented in subsequent phases
        
        logger.info("🎤 Voice Assistant is ready!")
        logger.info("📊 System Statistics:")
        
        # Event Bus Statistics
        eb_stats = event_bus.get_statistics()
        logger.info(f"   - Event subscriptions: {eb_stats.active_subscriptions}")
        logger.info(f"   - Worker threads: {event_bus._worker_threads}")
        
        # Audio Services Statistics
        capture_stats = audio_capture.get_statistics()
        player_stats = audio_player.get_statistics()
        logger.info(f"   - Input device: {capture_stats.get('selected_device', 'None')}")
        logger.info(f"   - Output device: {player_stats.get('selected_device', 'None')}")
        logger.info(f"   - Audio capture ready: {capture_stats.get('is_capturing', False)}")
        logger.info(f"   - Audio player ready: {player_stats.get('is_playing', False)}")
        
        # Start audio capture for demonstration
        try:
            audio_capture.start_capture()
            logger.info("🎧 Audio capture started - listening for speech...")
            logger.info("🔊 Audio player ready - will beep when you finish speaking")
        except Exception as e:
            logger.warning(f"⚠️  Audio capture not available: {e}")
        
        # Keep the application running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        
    except AstirError as e:
        logger.error(f"❌ Astir Error: {e.message}")
        if e.details:
            logger.error(f"   Details: {e.details}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            event_bus.publish(EventTypes.SYSTEM_SHUTDOWN, {
                "reason": "Application shutdown",
                "timestamp": "now"
            })
            audio_capture.shutdown()
            audio_player.shutdown()
            event_bus.shutdown()
            logger.info("✅ Cleanup completed")
        except:
            pass


if __name__ == "__main__":
    main()
