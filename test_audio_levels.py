#!/usr/bin/env python3
"""
Test audio capture and RMS levels to debug speech detection.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audio.capture_service import AudioCaptureService, SimpleVAD
from src.core.event_bus import EventBusService

def test_audio_levels():
    """Test audio capture and show RMS levels."""
    print("🎤 Testing Audio Capture and RMS Levels...")
    
    # Create event bus
    event_bus = EventBusService()
    
    # Create audio capture service
    audio_capture = AudioCaptureService(event_bus)
    
    if not audio_capture.initialize():
        print("❌ Failed to initialize audio capture")
        return
    
    print("✅ Audio capture initialized")
    print("🎤 Starting audio capture for 10 seconds...")
    print("📢 Please speak into your microphone!")
    
    # Start capture
    if not audio_capture.start_capture():
        print("❌ Failed to start audio capture")
        return
    
    # Monitor for 10 seconds
    start_time = time.time()
    max_rms = 0.0
    speech_detected_count = 0
    
    try:
        while time.time() - start_time < 10.0:
            # Get current stats
            stats = audio_capture.get_capture_statistics()
            current_rms = stats.get('current_rms_level', 0.0)
            
            if current_rms > max_rms:
                max_rms = current_rms
            
            if current_rms > 0.015:  # Default threshold
                speech_detected_count += 1
                print(f"🗣️  Speech detected! RMS: {current_rms:.4f}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    
    # Stop capture
    audio_capture.stop_capture()
    
    print(f"\n📊 Results:")
    print(f"   Max RMS Level: {max_rms:.4f}")
    print(f"   Speech Detection Threshold: 0.015")
    print(f"   Times above threshold: {speech_detected_count}")
    
    if max_rms < 0.005:
        print("⚠️  Very low audio levels - check microphone connection")
    elif max_rms < 0.015:
        print("⚠️  Audio levels below speech threshold - may need to lower threshold")
    else:
        print("✅ Audio levels look good for speech detection")
    
    # Shutdown
    audio_capture.shutdown()
    event_bus.shutdown()

if __name__ == "__main__":
    test_audio_levels()
