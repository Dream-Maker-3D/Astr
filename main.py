#!/usr/bin/env python3
"""
Astir Voice Assistant - Main Entry Point.

This is the main entry point for the Astir Voice Assistant system,
providing natural voice conversations using the complete pipeline:
Audio Input → STT (Whisper) → AI (OpenRouter) → TTS (Coqui) → Audio Output
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.facade import VoiceAssistantFacade, VoiceAssistantState
from src.utils.exceptions import AstirError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('voice_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Global voice assistant instance
voice_assistant: Optional[VoiceAssistantFacade] = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    
    if voice_assistant:
        voice_assistant.shutdown()
    
    sys.exit(0)


def check_environment():
    """Check environment requirements."""
    logger.info("🔍 Checking environment requirements...")
    
    # Check for OpenRouter API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.warning("⚠️  OPENROUTER_API_KEY not found in environment variables")
        logger.warning("   Set your API key with: export OPENROUTER_API_KEY='your-key-here'")
        return False
    else:
        logger.info("✅ OpenRouter API key found")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    else:
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    return True


def print_banner():
    """Print the Astir Voice Assistant banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                🎤 ASTIR VOICE ASSISTANT 🎤                   ║
    ║                                                              ║
    ║              Natural Voice Conversations with AI             ║
    ║                                                              ║
    ║    Audio Input → STT → AI (OpenRouter) → TTS → Audio Out    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_status(voice_assistant: VoiceAssistantFacade):
    """Print current system status."""
    status = voice_assistant.get_system_status()
    health = voice_assistant.get_system_health()
    
    print("\n📊 SYSTEM STATUS:")
    print(f"   State: {status['state']}")
    print(f"   Uptime: {status['uptime_seconds']:.1f}s")
    print(f"   Conversations: {status['conversation_count']}")
    print(f"   Errors: {status['error_count']}")
    print(f"   Health: {'✅ Healthy' if health['healthy'] else '❌ Issues detected'}")
    
    print("\n🔧 SERVICES STATUS:")
    for service_name, service_status in status['services'].items():
        status_icon = "✅" if service_status else "❌"
        print(f"   {service_name}: {status_icon}")
    
    if not health['healthy']:
        print("\n⚠️  HEALTH ISSUES:")
        for check in health['checks']:
            print(f"   - {check}")


def interactive_mode(voice_assistant: VoiceAssistantFacade):
    """Run interactive mode with user commands."""
    print("\n🎤 INTERACTIVE MODE")
    print("Commands:")
    print("  'start' - Start continuous conversation mode")
    print("  'stop'  - Stop conversation mode")
    print("  'status' - Show system status")
    print("  'health' - Show system health")
    print("  'quit'  - Shutdown system")
    print()
    
    try:
        while True:
            try:
                command = input("astir> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'start':
                    if voice_assistant.state == VoiceAssistantState.READY:
                        print("🎤 Starting continuous conversation mode...")
                        if voice_assistant.start_conversation_mode():
                            print("✅ Conversation mode started - speak naturally!")
                            print("   (The system is now listening continuously)")
                        else:
                            print("❌ Failed to start conversation mode")
                    else:
                        print(f"❌ Cannot start from state: {voice_assistant.state}")
                
                elif command == 'stop':
                    if voice_assistant.state in [VoiceAssistantState.LISTENING, 
                                                VoiceAssistantState.PROCESSING, 
                                                VoiceAssistantState.RESPONDING]:
                        print("🛑 Stopping conversation mode...")
                        if voice_assistant.stop_conversation_mode():
                            print("✅ Conversation mode stopped")
                        else:
                            print("❌ Failed to stop conversation mode")
                    else:
                        print("❌ Conversation mode not active")
                
                elif command == 'status':
                    print_system_status(voice_assistant)
                
                elif command == 'health':
                    health = voice_assistant.get_system_health()
                    if health['healthy']:
                        print("✅ System is healthy")
                    else:
                        print("❌ System health issues detected:")
                        for check in health['checks']:
                            print(f"   - {check}")
                
                elif command == 'help':
                    print("Commands: start, stop, status, health, quit")
                
                elif command == '':
                    continue  # Empty input
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit gracefully.")
            except EOFError:
                break
                
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")


def main():
    """Main entry point for the Voice Assistant."""
    global voice_assistant
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Print banner
        print_banner()
        
        # Check environment
        if not check_environment():
            logger.error("❌ Environment check failed")
            sys.exit(1)
        
        # Initialize Voice Assistant
        logger.info("🚀 Starting Astir Voice Assistant...")
        
        voice_assistant = VoiceAssistantFacade()
        
        # Initialize the system
        if not voice_assistant.initialize():
            logger.error("❌ Failed to initialize Voice Assistant")
            sys.exit(1)
        
        # Print initial status
        print_system_status(voice_assistant)
        
        # Run interactive mode
        interactive_mode(voice_assistant)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if voice_assistant:
            logger.info("🛑 Shutting down Voice Assistant...")
            voice_assistant.shutdown()
        
        logger.info("✅ Astir Voice Assistant shutdown complete")


if __name__ == "__main__":
    main()
