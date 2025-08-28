#!/usr/bin/env python3
"""
Simple audio test to check microphone input levels.
"""

import pyaudio
import numpy as np
import time

def test_microphone():
    """Test microphone input and show RMS levels."""
    print("🎤 Testing Microphone Input...")
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("✅ Audio stream opened")
        print("🎤 Recording for 10 seconds - please speak!")
        print("📊 Watching RMS levels (threshold: 0.015)...")
        
        max_rms = 0.0
        speech_count = 0
        
        for i in range(int(RATE / CHUNK * 10)):  # 10 seconds
            # Read audio data
            data = stream.read(CHUNK)
            
            # Convert to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > max_rms:
                max_rms = rms
            
            if rms > 0.015:
                speech_count += 1
                print(f"🗣️  RMS: {rms:.4f} (above threshold)")
            elif rms > 0.005:
                print(f"🔊 RMS: {rms:.4f} (audible)")
            
            time.sleep(0.01)
        
        print(f"\n📊 Results:")
        print(f"   Max RMS: {max_rms:.4f}")
        print(f"   Times above 0.015: {speech_count}")
        
        if max_rms < 0.005:
            print("⚠️  Very quiet - check microphone")
        elif max_rms < 0.015:
            print("⚠️  Below speech threshold - consider lowering to", max_rms * 0.7)
        else:
            print("✅ Good audio levels detected")
        
        # Clean up
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        p.terminate()

if __name__ == "__main__":
    test_microphone()
