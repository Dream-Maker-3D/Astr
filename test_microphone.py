#!/usr/bin/env python3
"""
Test microphone input directly.
"""

import pyaudio
import numpy as np
import time

def test_microphone():
    """Test microphone capture and display audio levels."""
    print("🎤 Testing Microphone Input...")
    print("=" * 50)
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    DEVICE_ID = 16  # Same as configured in the assistant
    
    p = pyaudio.PyAudio()
    
    # Get device info
    device_info = p.get_device_info_by_index(DEVICE_ID)
    print(f"📱 Using device: {device_info['name']}")
    print(f"   Max input channels: {device_info['maxInputChannels']}")
    print(f"   Default sample rate: {device_info['defaultSampleRate']}")
    print()
    
    try:
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=DEVICE_ID,
            frames_per_buffer=CHUNK
        )
        
        print("🔊 Monitoring audio levels (speak into microphone)...")
        print("   Press Ctrl+C to stop")
        print("-" * 50)
        
        # Monitor for 30 seconds
        start_time = time.time()
        max_level = 0
        speech_detected = False
        
        while time.time() - start_time < 30:
            try:
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
                
                # Calculate RMS (root mean square) for volume level
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Update max level
                if rms > max_level:
                    max_level = rms
                
                # Create visual meter
                meter_length = int(rms * 500)  # Scale for display
                meter = "█" * meter_length
                
                # Check if this could be speech (threshold similar to VAD)
                if rms > 0.015:  # Using the VAD threshold from config
                    if not speech_detected:
                        speech_detected = True
                        print(f"\n🗣️  SPEECH DETECTED! (RMS: {rms:.4f})")
                    status = "SPEAKING"
                    color = "\033[92m"  # Green
                else:
                    speech_detected = False
                    status = "SILENT  "
                    color = "\033[90m"  # Gray
                
                # Display level
                print(f"\r{color}[{status}] RMS: {rms:6.4f} | Max: {max_level:6.4f} | {meter:<50}\033[0m", end="", flush=True)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error reading audio: {e}")
                continue
        
        print(f"\n\n📊 Summary:")
        print(f"   Max audio level detected: {max_level:.4f}")
        print(f"   Speech threshold: 0.015")
        
        if max_level < 0.001:
            print("   ⚠️  WARNING: Very low or no audio detected!")
            print("   Check microphone connection and permissions")
        elif max_level < 0.015:
            print("   ⚠️  Audio detected but below speech threshold")
            print("   Try speaking louder or adjusting microphone")
        else:
            print("   ✅ Microphone is working properly!")
        
    except Exception as e:
        print(f"❌ Failed to open audio stream: {e}")
        print("\nTroubleshooting:")
        print("1. Check if device ID 16 is correct")
        print("2. Verify microphone permissions")
        print("3. Ensure no other application is using the microphone")
        
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_microphone()