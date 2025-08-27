class VoiceAssistantError(Exception):
    """Base exception for voice assistant system."""
    pass


class AudioError(VoiceAssistantError):
    """Audio-related errors."""
    pass


class SpeechRecognitionError(VoiceAssistantError):
    """Speech recognition errors."""
    pass


class SpeechSynthesisError(VoiceAssistantError):
    """Speech synthesis errors."""
    pass


class AIServiceError(VoiceAssistantError):
    """AI service communication errors."""
    pass


class ConfigurationError(VoiceAssistantError):
    """Configuration-related errors."""
    pass


class DeviceNotFoundError(AudioError):
    """Audio device not found."""
    pass


class ModelLoadError(VoiceAssistantError):
    """Model loading errors."""
    pass


class NetworkError(VoiceAssistantError):
    """Network communication errors."""
    pass