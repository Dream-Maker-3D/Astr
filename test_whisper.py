#!/usr/bin/env python3
"""
Test Whisper integration without full audio stack.
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.speech.strategies.whisper_stt import WhisperSTTStrategy


async def test_whisper():
    """Test Whisper STT functionality."""
    print("🎤 Testing Whisper Speech Recognition...")
    
    try:
        # Initialize Whisper STT
        whisper_stt = WhisperSTTStrategy(model_name="tiny", language="en")
        
        print("⚙️  Loading Whisper model...")
        if await whisper_stt.initialize():
            print("✅ Whisper model loaded successfully")
        else:
            print("❌ Failed to load Whisper model")
            return False
        
        # Create a test audio signal (silence)
        print("🔊 Testing with synthetic audio...")
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        
        # Generate white noise (more likely to produce transcription)
        test_audio = np.random.normal(0, 0.1, samples).astype(np.float32)
        
        # Test transcription
        result = await whisper_stt.transcribe(test_audio, sample_rate)
        
        if result:
            print(f"✅ Whisper transcription result:")
            print(f"   Text: '{result.get('text', 'No text')}'")
            print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"   Language: {result.get('language', 'unknown')}")
        else:
            print("❌ No transcription result")
            return False
        
        # Test with actual silence
        print("\n🔇 Testing with silence...")
        silence_audio = np.zeros(samples, dtype=np.float32)
        silence_result = await whisper_stt.transcribe(silence_audio, sample_rate)
        
        if silence_result:
            print(f"✅ Silence transcription:")
            print(f"   Text: '{silence_result.get('text', 'No text')}'")
        
        # Cleanup
        await whisper_stt.cleanup()
        print("✅ Whisper cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Testing Whisper Speech Recognition")
    print("=" * 40)
    
    success = await test_whisper()
    
    if success:
        print("\n✅ Whisper test passed! Speech recognition is working.")
        print("💡 Ready for full voice assistant integration.")
        return 0
    else:
        print("\n❌ Whisper test failed.")
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