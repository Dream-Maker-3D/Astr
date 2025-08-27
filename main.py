#!/usr/bin/env python3
"""
Voice Assistant Main Entry Point

Natural conversation voice assistant powered by:
- OpenAI Whisper (Speech-to-Text)
- Coqui TTS (Text-to-Speech)  
- Claude AI (Conversation)

Features:
- Continuous listening without wake words
- Natural interruption handling
- Voice cloning support
- GoF design patterns implementation
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.facade import VoiceAssistantFacade, SystemStatus
from src.utils.exceptions import VoiceAssistantError


class VoiceAssistantCLI:
    """Command Line Interface for the Voice Assistant."""
    
    def __init__(self):
        self.assistant: VoiceAssistantFacade = None
        self.running = False
    
    async def run(self, config_path: str = None, test_mode: bool = False) -> None:
        """Run the voice assistant with specified configuration."""
        try:
            print("🎤 Voice Assistant Starting...")
            print("=" * 50)
            
            # Initialize the voice assistant
            self.assistant = VoiceAssistantFacade(config_path)
            
            print("⚙️  Initializing services...")
            if not await self.assistant.initialize():
                print("❌ Failed to initialize voice assistant")
                return
            
            print("✅ All services initialized successfully")
            
            if test_mode:
                await self._run_test_mode()
            else:
                await self._run_conversation_mode()
                
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            await self._shutdown()
    
    async def _run_conversation_mode(self) -> None:
        """Run in natural conversation mode."""
        print("\n🗣️  Starting natural conversation mode...")
        print("💡 Just start speaking - no wake words needed!")
        print("🔄 Say 'goodbye' or press Ctrl+C to exit")
        print("-" * 50)
        
        # Start conversation
        if not await self.assistant.start_conversation():
            print("❌ Failed to start conversation mode")
            return
        
        self.running = True
        print("🟢 Voice Assistant is now listening...")
        
        try:
            # Keep the main loop running
            while self.running and self.assistant.is_active():
                await asyncio.sleep(1)
                
                # Check system status periodically
                status = self.assistant.get_status()
                if status['system_status'] == SystemStatus.ERROR.value:
                    print("❌ System error detected, shutting down...")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Conversation interrupted by user")
        finally:
            await self.assistant.stop_conversation()
    
    async def _run_test_mode(self) -> None:
        """Run in test mode with manual message input."""
        print("\n🧪 Running in test mode...")
        print("💡 Type messages to test the AI conversation")
        print("🔄 Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                message = input("\n👤 You: ").strip()
                
                if message.lower() in ['quit', 'exit', 'goodbye']:
                    print("🔚 Exiting test mode...")
                    break
                
                if message:
                    print("🤖 Assistant: ", end="", flush=True)
                    response = await self.assistant.send_test_message(message)
                    if response:
                        print(response)
                    else:
                        print("❌ No response received")
                
            except KeyboardInterrupt:
                print("\n🛑 Test mode interrupted")
                break
            except EOFError:
                print("\n🔚 End of input detected")
                break
    
    async def _shutdown(self) -> None:
        """Gracefully shutdown the assistant."""
        if self.assistant:
            print("\n🔄 Shutting down voice assistant...")
            await self.assistant.shutdown()
            print("✅ Shutdown complete")
        
        self.running = False
    
    def _print_status(self) -> None:
        """Print current system status."""
        if self.assistant:
            status = self.assistant.get_status()
            
            print("\n📊 System Status:")
            print(f"   Status: {status['system_status']}")
            print(f"   Conversation Active: {status['conversation_active']}")
            
            print("\n🔧 Services:")
            for service, info in status['services'].items():
                status_icon = "✅" if info.get('initialized', False) else "❌"
                print(f"   {status_icon} {service.replace('_', ' ').title()}")
            
            if status['conversation'].get('current_conversation_id'):
                print(f"\n💬 Conversation ID: {status['conversation']['current_conversation_id']}")
                print(f"   Duration: {status['conversation']['conversation_duration']:.1f}s")
                print(f"   History Length: {status['conversation']['history_length']}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Natural Voice Assistant with Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run with default config
  python main.py --config config/custom.yaml # Run with custom config
  python main.py --test                       # Run in test mode
  python main.py --status                     # Show system status and exit
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run in test mode (text input/output only)'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        print("🐛 Debug logging enabled")
    
    # Check for required environment variables
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("💡 Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return 1
    
    cli = VoiceAssistantCLI()
    
    if args.status:
        # Just show status and exit
        assistant = VoiceAssistantFacade(args.config)
        if await assistant.initialize():
            cli.assistant = assistant
            cli._print_status()
            await assistant.shutdown()
        return 0
    
    # Run the voice assistant
    try:
        await cli.run(args.config, args.test)
        return 0
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal application error: {e}")
        sys.exit(1)