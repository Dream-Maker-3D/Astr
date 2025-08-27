#!/usr/bin/env python3
"""
Test Coqui TTS directly to verify it works.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.speech.strategies.coqui_tts import CoquiTTSStrategy


async def test_tts():
    """Test TTS functionality directly."""
    print("🎤 Testing Coqui TTS...")
    
    try:
        # Initialize Coqui TTS
        tts = CoquiTTSStrategy(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            language="en"
        )
        
        print("⚙️  Loading TTS model...")
        if await tts.initialize():
            print("✅ TTS model loaded successfully")
        else:
            print("❌ Failed to load TTS model")
            return False
        
        # Check available speakers
        speakers = tts.get_available_speakers()
        print(f"📢 Available speakers: {speakers}")
        
        # Test text
        test_text = "Hello! I am your voice assistant. I can speak naturally now."
        
        # Use Daisy Studious (young female voice) if available, otherwise first speaker
        speaker_id = "Daisy Studious" if "Daisy Studious" in speakers else (speakers[0] if speakers else None)
        
        print(f"🗣️  Synthesizing: '{test_text}'")
        if speaker_id:
            print(f"   Using speaker: {speaker_id}")
            audio_data = await tts.synthesize(test_text, speaker_id=speaker_id)
        else:
            # Create a reference wav for voice cloning
            print("   Creating reference audio for voice cloning...")
            import tempfile
            
            # Generate a short sample audio (white noise as reference)
            sample_rate = 22050
            duration = 3.0  # seconds
            samples = int(sample_rate * duration)
            reference_audio = np.random.normal(0, 0.01, samples).astype(np.float32)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                import scipy.io.wavfile as wav
                temp_path = temp_file.name
                wav.write(temp_path, sample_rate, reference_audio)
            
            print(f"   Using reference audio: {temp_path}")
            audio_data = await tts.synthesize(test_text, speaker_wav=temp_path)
        
        if audio_data is not None and len(audio_data) > 0:
            print(f"✅ Audio synthesized: {len(audio_data)} samples")
            print(f"   Duration: {len(audio_data) / tts.get_sample_rate():.2f} seconds")
            
            # Play the audio
            print("🔊 Playing audio...")
            sd.play(audio_data, tts.get_sample_rate())
            sd.wait()  # Wait for playback to finish
            print("✅ Audio playback complete")
        else:
            print("❌ No audio data generated")
            return False
        
        # Cleanup
        await tts.cleanup()
        print("✅ TTS cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Testing Coqui TTS Text-to-Speech")
    print("=" * 40)
    
    success = await test_tts()
    
    if success:
        print("\n✅ TTS test passed! Voice synthesis is working.")
        return 0
    else:
        print("\n❌ TTS test failed.")
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