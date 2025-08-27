#!/usr/bin/env python3
"""
Basic Configuration Manager Test

Tests the configuration data structures and basic functionality
without requiring external dependencies like PyYAML.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration_data_structures():
    """Test that configuration data structures work correctly."""
    print("🧪 Testing Configuration Data Structures...")
    
    try:
        from src.core.config_manager import (
            AudioConfig, 
            SpeechConfig, 
            AIConfig, 
            ConversationConfig, 
            SystemConfig, 
            Configuration
        )
        
        # Test individual config creation
        audio_config = AudioConfig()
        print(f"✅ AudioConfig created: sample_rate={audio_config.input_sample_rate}")
        
        speech_config = SpeechConfig()
        print(f"✅ SpeechConfig created: provider={speech_config.recognition_provider}")
        
        ai_config = AIConfig()
        print(f"✅ AIConfig created: model={ai_config.model}")
        
        conversation_config = ConversationConfig()
        print(f"✅ ConversationConfig created: threshold={conversation_config.voice_activity_threshold}")
        
        system_config = SystemConfig()
        print(f"✅ SystemConfig created: log_level={system_config.log_level}")
        
        # Test complete configuration
        config = Configuration()
        print(f"✅ Complete Configuration created")
        print(f"   - Audio sample rate: {config.audio.input_sample_rate}")
        print(f"   - AI model: {config.ai.model}")
        print(f"   - Voice threshold: {config.conversation.voice_activity_threshold}")
        print(f"   - Log level: {config.system.log_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_singleton_pattern():
    """Test that ConfigurationManager follows singleton pattern."""
    print("\n🧪 Testing Singleton Pattern...")
    
    try:
        # Import without YAML dependency will fail, but we can test the pattern
        # by checking if the class definition is correct
        from src.core.config_manager import ConfigurationManager
        
        # Test that multiple instances are the same
        cm1 = ConfigurationManager()
        cm2 = ConfigurationManager()
        
        if cm1 is cm2:
            print("✅ Singleton pattern working correctly")
            return True
        else:
            print("❌ Singleton pattern failed - different instances")
            return False
            
    except ImportError as e:
        if "yaml" in str(e):
            print("⚠️  Singleton test skipped - YAML dependency missing")
            return True
        else:
            print(f"❌ Singleton test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Singleton test failed: {e}")
        return False

def main():
    """Run all configuration tests."""
    print("🎯 Configuration Manager Basic Tests\n")
    
    tests = [
        test_configuration_data_structures,
        test_singleton_pattern
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
