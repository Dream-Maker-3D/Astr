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
        
        # Initialize Event Bus
        event_bus = EventBusService()
        event_bus.initialize()
        logger.info("✅ Event Bus initialized")
        
        # Set up event handlers for demonstration
        def system_ready_handler(data):
            logger.info(f"🎉 System ready: {data}")
        
        def system_shutdown_handler(data):
            logger.info(f"🛑 System shutdown: {data}")
        
        # Subscribe to system events
        event_bus.subscribe(EventTypes.SYSTEM_READY, system_ready_handler)
        event_bus.subscribe(EventTypes.SYSTEM_SHUTDOWN, system_shutdown_handler)
        
        logger.info("✅ Event handlers registered")
        
        # TODO: Initialize other services (Audio, Speech, AI, etc.)
        # This will be implemented in subsequent phases
        
        logger.info("🎤 Voice Assistant is ready!")
        logger.info("📊 Event Bus Statistics:")
        stats = event_bus.get_statistics()
        logger.info(f"   - Active subscriptions: {stats.active_subscriptions}")
        logger.info(f"   - Worker threads: {event_bus._worker_threads}")
        logger.info(f"   - Queue size: {stats.queue_size}")
        
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
            event_bus.shutdown()
            logger.info("✅ Cleanup completed")
        except:
            pass


if __name__ == "__main__":
    main()
