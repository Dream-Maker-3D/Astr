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
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.facade import VoiceAssistantFacade, VoiceAssistantState
from src.utils.exceptions import AstirError

# Global variables
logger = None  # Will be configured after parsing arguments

# Global voice assistant instance
voice_assistant: Optional[VoiceAssistantFacade] = None


def configure_logging(debug: bool = False, quiet: bool = False):
    """Configure logging based on debug and quiet flags."""
    global logger
    
    # Determine log level
    if quiet:
        console_level = logging.WARNING
        file_level = logging.INFO
    elif debug:
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    else:
        console_level = logging.INFO
        file_level = logging.INFO
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s') if not debug else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers will filter
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('voice_assistant.log')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    if not debug:
        logging.getLogger('TTS').setLevel(logging.WARNING)
        logging.getLogger('whisper').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Astir Voice Assistant - Natural voice conversations with AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Normal mode with minimal output
  python main.py --debug            # Debug mode with verbose logging
  python main.py --quiet            # Quiet mode with minimal console output
  python main.py --debug --quiet    # Debug to file only, minimal console output

The voice assistant provides natural conversation using:
  Audio Input → Whisper STT → OpenRouter AI → Coqui TTS → Audio Output
        """
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - minimal console output (warnings and errors only)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    
    if voice_assistant:
        voice_assistant.shutdown()
    
    sys.exit(0)


def check_environment(debug: bool = False):
    """Check environment requirements."""
    if debug:
        logger.info("🔍 Checking environment requirements...")
    
    # Check for OpenRouter API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("❌ OPENROUTER_API_KEY not found in environment variables")
        logger.error("   Set your API key with: export OPENROUTER_API_KEY='your-key-here'")
        return False
    elif debug:
        logger.info("✅ OpenRouter API key found")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    elif debug:
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
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging based on arguments
    configure_logging(debug=args.debug, quiet=args.quiet)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Print banner (unless quiet mode)
        if not args.quiet:
            print_banner()
        
        # Check environment
        if not check_environment(debug=args.debug):
            logger.error("❌ Environment check failed")
            sys.exit(1)
        
        # Initialize Voice Assistant
        if args.debug:
            logger.info("🚀 Starting Astir Voice Assistant...")
        elif not args.quiet:
            print("🚀 Starting Astir Voice Assistant...")
        
        voice_assistant = VoiceAssistantFacade()
        
        # Initialize the system
        if not voice_assistant.initialize():
            logger.error("❌ Failed to initialize Voice Assistant")
            sys.exit(1)
        
        # Print initial status (unless quiet mode)
        if not args.quiet:
            print_system_status(voice_assistant)
        
        # Run interactive mode
        interactive_mode(voice_assistant)
        
    except KeyboardInterrupt:
        if args.debug:
            logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup
        if voice_assistant:
            if args.debug:
                logger.info("🛑 Shutting down Voice Assistant...")
            voice_assistant.shutdown()
        
        if args.debug:
            logger.info("✅ Astir Voice Assistant shutdown complete")


if __name__ == "__main__":
    main()
