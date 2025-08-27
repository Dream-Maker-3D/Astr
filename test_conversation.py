#!/usr/bin/env python3
"""
Test conversational functionality without audio dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.config_manager import ConfigurationManager
from src.core.event_bus import EventBusService, EventTypes
from src.ai.conversation_service import AIConversationService


async def test_conversation():
    """Test conversation functionality."""
    print("🗣️ Testing AI Conversation Service...")
    
    try:
        # Initialize components
        config = ConfigurationManager()
        config.load_config()
        
        event_bus = EventBusService()
        ai_service = AIConversationService(event_bus)
        
        # Initialize AI service
        print("⚙️  Initializing AI conversation service...")
        if await ai_service.initialize():
            print("✅ AI conversation service initialized")
        else:
            print("❌ Failed to initialize AI service")
            return False
        
        # Test direct conversation
        print("\n💬 Testing direct conversation...")
        response = await ai_service.send_message_direct("Hello! Please respond with just 'Hello there!'")
        
        if response and response.get('text'):
            print(f"🤖 AI Response: {response['text']}")
            print(f"   Response time: {response.get('response_time', 0):.2f}s")
        else:
            print("❌ No response from AI")
            return False
        
        # Test conversation flow
        print("\n🔄 Testing conversation flow...")
        
        # Start a new conversation
        conversation_id = await ai_service.start_new_conversation()
        print(f"✅ Started conversation: {conversation_id[:8]}...")
        
        # Test multiple exchanges
        test_messages = [
            "What's 2 + 2?",
            "Can you explain that in more detail?",
            "Thank you, that was helpful."
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. 👤 User: {message}")
            response = await ai_service.send_message_direct(message, conversation_id)
            
            if response and response.get('text'):
                print(f"   🤖 AI: {response['text']}")
            else:
                print("   ❌ No response")
                return False
        
        # Get conversation stats
        stats = ai_service.get_conversation_stats()
        print(f"\n📊 Conversation Stats:")
        print(f"   History length: {stats['history_length']}")
        print(f"   Duration: {stats['conversation_duration']:.1f}s")
        
        # End conversation
        await ai_service.end_conversation()
        print("✅ Conversation ended properly")
        
        # Cleanup
        await ai_service.cleanup()
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Testing Voice Assistant Conversation System")
    print("=" * 55)
    
    success = await test_conversation()
    
    if success:
        print("\n✅ Conversation test passed! AI integration is working perfectly.")
        print("💡 Ready for audio system integration.")
        return 0
    else:
        print("\n❌ Conversation test failed.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal test error: {e}")
        sys.exit(1)