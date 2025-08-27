#!/usr/bin/env python3
"""
Basic test of voice assistant core components without audio dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.config_manager import ConfigurationManager
from src.core.event_bus import EventBusService, EventTypes
from src.ai.openrouter_client import OpenRouterClient


async def test_basic_components():
    """Test basic components without audio dependencies."""
    print("🧪 Testing basic voice assistant components...")
    
    try:
        # Test ConfigurationManager (Singleton pattern)
        print("\n1️⃣ Testing ConfigurationManager...")
        config = ConfigurationManager()
        config.load_config()
        
        # Verify we can get configuration values
        ai_config = config.get_ai_config()
        print(f"✅ AI Model: {ai_config.get('model')}")
        print(f"✅ Max Tokens: {ai_config.get('max_tokens')}")
        
        # Test EventBusService (Observer pattern)
        print("\n2️⃣ Testing EventBusService...")
        event_bus = EventBusService()
        
        received_events = []
        def test_handler(event_data):
            received_events.append(event_data)
            print(f"✅ Event received: {event_data['type']}")
        
        event_bus.subscribe(EventTypes.SYSTEM_STATUS_CHANGED, test_handler)
        
        # Publish test event
        await event_bus.publish_async(
            EventTypes.SYSTEM_STATUS_CHANGED,
            {'service': 'test', 'status': 'testing'}
        )
        
        await asyncio.sleep(0.1)  # Give time for async event processing
        
        if received_events:
            print(f"✅ Event system working: {len(received_events)} events processed")
        else:
            print("❌ Event system not working")
            return False
        
        # Test OpenRouterClient
        print("\n3️⃣ Testing OpenRouter AI integration...")
        ai_client = OpenRouterClient()
        
        if await ai_client.initialize():
            print("✅ OpenRouter client initialized successfully")
            
            # Test connection
            print("🔍 Testing API connection...")
            response = await ai_client.send_message("Hello, please respond with just 'Connection test successful'")
            
            if response and response.get('text'):
                print(f"✅ OpenRouter API Response: {response['text'].strip()}")
                print(f"   Response time: {response.get('response_time', 0):.2f}s")
                print(f"   Tokens used: {response.get('usage', {})}")
            else:
                print("❌ No response from OpenRouter API")
                return False
        else:
            print("❌ Failed to initialize OpenRouter client")
            return False
        
        print("\n🎉 All basic components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting basic component tests...")
    print("=" * 50)
    
    success = await test_basic_components()
    
    if success:
        print("\n✅ Basic test passed! Core architecture is working.")
        print("💡 The system is ready for audio component integration.")
        return 0
    else:
        print("\n❌ Basic test failed. Check configuration and API key.")
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