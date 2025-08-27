"""
Audio processing services for the Astir Voice Assistant.

This package contains audio capture, playback, and preprocessing services
that provide the foundation for voice interaction.
"""

# Audio services
from .capture_service import AudioCaptureService, AudioDevice, AudioData, VoiceActivityDetector
from .player_service import AudioPlayerService, AudioOutputDevice, AudioClip, PlaybackPriority, AudioQueue

__all__ = [
    'AudioCaptureService',
    'AudioDevice', 
    'AudioData',
    'VoiceActivityDetector',
    'AudioPlayerService',
    'AudioOutputDevice',
    'AudioClip',
    'PlaybackPriority',
    'AudioQueue'
]
