#!/usr/bin/env python3
"""
Test real-time speech handling with natural conversation patterns.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ai.openrouter_client import OpenRouterClient


async def test_realtime_speech_patterns():
    """Test how well the AI handles real-time speech patterns."""
    print("🗣️  Testing Real-Time Speech Handling...")
    
    try:
        # Initialize OpenRouter client
        client = OpenRouterClient()
        
        if not await client.initialize():
            print("❌ Failed to initialize AI client")
            return False
        
        # Test scenarios that simulate real speech patterns
        test_scenarios = [
            {
                "name": "Incomplete thought",
                "input": "I was thinking about... um... getting a new laptop but I'm not sure if...",
                "expect": "Brief, helpful response about laptop buying"
            },
            {
                "name": "Quick question", 
                "input": "What's the weather like?",
                "expect": "Very brief response"
            },
            {
                "name": "Interruption simulation",
                "input": "Can you tell me about... actually, never mind, what time is it?",
                "expect": "Responds to the time question, ignores the first part"
            },
            {
                "name": "Casual speech",
                "input": "Hey, I'm kinda hungry, what should I eat?",
                "expect": "Casual, brief food suggestion"
            },
            {
                "name": "Correction handling",
                "input": "No wait, I meant Python not Java",
                "expect": "Smoothly switches to Python without acknowledging correction"
            }
        ]
        
        print(f"\n🧪 Running {len(test_scenarios)} real-time speech tests...\n")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"{i}️⃣  **{scenario['name']}**")
            print(f"   Input: \"{scenario['input']}\"")
            
            start_time = asyncio.get_event_loop().time()
            response = await client.send_message(scenario['input'])
            end_time = asyncio.get_event_loop().time()
            
            response_time = end_time - start_time
            
            if response and response.get('text'):
                response_text = response['text'].strip()
                word_count = len(response_text.split())
                
                print(f"   Response: \"{response_text}\"")
                print(f"   ⏱️  Time: {response_time:.2f}s | Words: {word_count} | Model: {response.get('model', 'unknown')}")
                
                # Evaluate response quality
                if response_time <= 2.0:
                    print("   ✅ Good response time")
                elif response_time <= 3.0:
                    print("   ⚠️  Acceptable response time")
                else:
                    print("   ❌ Slow response time")
                
                if word_count <= 20:
                    print("   ✅ Appropriately brief")
                elif word_count <= 35:
                    print("   ⚠️  Somewhat verbose")
                else:
                    print("   ❌ Too verbose for real-time")
                    
            else:
                print("   ❌ No response received")
                return False
            
            print()
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        print("🎯 **Real-Time Speech Evaluation Complete**")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Real-Time Speech Pattern Testing")
    print("=" * 50)
    
    success = await test_realtime_speech_patterns()
    
    if success:
        print("\n✅ Real-time speech testing complete!")
        print("💡 Review the response times and brevity for optimal real-time performance.")
        return 0
    else:
        print("\n❌ Real-time speech testing failed.")
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
