#!/usr/bin/env python3
"""
Compare different AI models for real-time speech performance.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ai.openrouter_client import OpenRouterClient


async def test_model_performance(model_name: str, test_input: str):
    """Test a specific model's performance."""
    try:
        client = OpenRouterClient()
        # Override the model for this test
        client.model = model_name
        
        if not await client.initialize():
            return None
        
        start_time = asyncio.get_event_loop().time()
        response = await client.send_message(test_input)
        end_time = asyncio.get_event_loop().time()
        
        if response and response.get('text'):
            return {
                'model': model_name,
                'response_time': end_time - start_time,
                'response_text': response['text'].strip(),
                'word_count': len(response['text'].strip().split()),
                'actual_model': response.get('model', 'unknown')
            }
        return None
        
    except Exception as e:
        print(f"   ❌ {model_name} failed: {e}")
        return None


async def main():
    """Test different models for real-time speech."""
    print("🚀 AI Model Performance Comparison for Real-Time Speech")
    print("=" * 60)
    
    # Models to test (in order of preference for real-time)
    models_to_test = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "openai/gpt-4o", 
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemini-flash-1.5"
    ]
    
    test_input = "Hey, what's a good quick lunch idea?"
    
    print(f"🧪 Testing with input: \"{test_input}\"\n")
    
    results = []
    
    for model in models_to_test:
        print(f"Testing {model}...")
        result = await test_model_performance(model, test_input)
        if result:
            results.append(result)
            print(f"   ✅ {result['response_time']:.2f}s | {result['word_count']} words")
        else:
            print(f"   ❌ Failed")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Sort by response time
    results.sort(key=lambda x: x['response_time'])
    
    print("\n🏆 **RESULTS RANKED BY SPEED**")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        speed_emoji = "🚀" if result['response_time'] < 1.5 else "⚡" if result['response_time'] < 2.5 else "🐌"
        brevity_emoji = "✅" if result['word_count'] <= 15 else "⚠️" if result['word_count'] <= 25 else "❌"
        
        print(f"{i}. {speed_emoji} **{result['model']}**")
        print(f"   Time: {result['response_time']:.2f}s | Words: {result['word_count']} {brevity_emoji}")
        print(f"   Response: \"{result['response_text'][:100]}{'...' if len(result['response_text']) > 100 else ''}\"")
        print(f"   Actual Model: {result['actual_model']}")
        print()
    
    if results:
        best = results[0]
        print(f"🥇 **RECOMMENDATION FOR REAL-TIME SPEECH:**")
        print(f"   Model: {best['model']}")
        print(f"   Speed: {best['response_time']:.2f}s")
        print(f"   Brevity: {best['word_count']} words")
        
        return 0
    else:
        print("❌ No models worked")
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
