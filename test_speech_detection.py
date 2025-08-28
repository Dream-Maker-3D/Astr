#!/usr/bin/env python3
"""
Test speech detection in the voice assistant.
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_speech_detection():
    """Test speech detection with detailed logging."""
    print("🎤 Testing Speech Detection...")
    
    try:
        from src.core.facade import VoiceAssistantFacade
        
        # Create voice assistant
        voice_assistant = VoiceAssistantFacade()
        
        if not voice_assistant.initialize():
            print("❌ Failed to initialize voice assistant")
            return
        
        print("✅ Voice assistant initialized")
        print("🎤 Starting conversation mode for 15 seconds...")
        print("📢 Please speak clearly into your microphone!")
        
        # Start conversation mode
        if not voice_assistant.start_conversation_mode():
            print("❌ Failed to start conversation mode")
            return
        
        # Wait and listen
        time.sleep(15)
        
        # Stop conversation
        voice_assistant.stop_conversation_mode()
        
        print("✅ Test completed - check logs above for speech detection events")
        
        # Shutdown
        voice_assistant.shutdown()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speech_detection()
