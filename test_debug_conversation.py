#!/usr/bin/env python3
"""
Test conversation with debug logging.
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_debug_conversation():
    """Test conversation with debug logging."""
    print("🎤 Testing Conversation with Debug Logging...")
    
    try:
        from src.core.facade import VoiceAssistantFacade
        
        # Create voice assistant
        voice_assistant = VoiceAssistantFacade()
        
        if not voice_assistant.initialize():
            print("❌ Failed to initialize voice assistant")
            return
        
        print("✅ Voice assistant initialized")
        print("🎤 Starting conversation mode for 20 seconds...")
        print("📢 SPEAK NOW - say something like 'Hello, how are you?'")
        
        # Start conversation mode
        if not voice_assistant.start_conversation_mode():
            print("❌ Failed to start conversation mode")
            return
        
        # Wait and listen
        time.sleep(20)
        
        # Stop conversation
        voice_assistant.stop_conversation_mode()
        
        print("✅ Test completed - check logs above for transcription and AI events")
        
        # Shutdown
        voice_assistant.shutdown()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_conversation()
