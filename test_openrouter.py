#!/usr/bin/env python3
"""
Test OpenRouter AI integration to verify it works.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ai.openrouter_client import OpenRouterClient


async def test_openrouter():
    """Test OpenRouter functionality directly."""
    print("🤖 Testing OpenRouter AI integration...")
    
    try:
        # Initialize OpenRouter client
        client = OpenRouterClient()
        
        print("⚙️  Initializing OpenRouter client...")
        if await client.initialize():
            print("✅ OpenRouter client initialized successfully")
        else:
            print("❌ Failed to initialize OpenRouter client")
            return False
        
        # Test connection
        print("🔍 Testing API connection...")
        if await client.test_connection():
            print("✅ OpenRouter API connection successful")
        else:
            print("❌ OpenRouter API connection failed")
            return False
        
        # Test a simple message
        test_message = "Hello! Can you respond with just 'Connection test successful' please?"
        
        print(f"💬 Sending test message: '{test_message}'")
        response = await client.send_message(test_message)
        
        if response and response.get('text'):
            print(f"✅ OpenRouter Response: {response['text']}")
            print(f"   Response time: {response['response_time']:.2f}s")
            print(f"   Model: {response['model']}")
            print(f"   Tokens used: {response['usage']}")
            return True
        else:
            print("❌ No response received from OpenRouter")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Testing OpenRouter AI Integration")
    print("=" * 40)
    
    success = await test_openrouter()
    
    if success:
        print("\n✅ OpenRouter test passed! AI integration is working.")
        print("💡 Ready for voice assistant integration.")
        return 0
    else:
        print("\n❌ OpenRouter test failed.")
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
