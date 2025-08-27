"""
Configuration Manager - Singleton Pattern Implementation

This module implements centralized configuration management for the Astir Voice Assistant,
providing YAML-based configuration with environment variable overrides and validation.
"""

import os
import logging
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from threading import Lock

from ..utils.exceptions import (
    ConfigurationError, 
    ConfigValidationError, 
    ConfigLoadError
)


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    # Input settings
    input_device_id: Optional[int] = None
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_chunk_size: int = 1024
    input_buffer_size: int = 4096
    continuous_listening: bool = True
    
    # Output settings
    output_device_id: Optional[int] = None
    output_sample_rate: int = 22050
    output_volume: float = 0.8


@dataclass
class SpeechConfig:
    """Speech recognition and synthesis configuration."""
    # Recognition settings
    recognition_provider: str = "whisper"
    recognition_model: str = "base"
    recognition_language: str = "en"
    confidence_threshold: float = 0.6
    continuous_mode: bool = True
    
    # Synthesis settings
    synthesis_provider: str = "coqui"
    synthesis_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    synthesis_voice: Optional[str] = None
    speaking_rate: float = 1.1
    pitch: float = 0.0
    naturalness: str = "high"


@dataclass
class AIConfig:
    """AI service configuration."""
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "anthropic/claude-3.5-sonnet"
    max_tokens: int = 150
    temperature: float = 0.8
    timeout: int = 20
    conversation_mode: bool = True
    system_prompt: str = field(default_factory=lambda: """You are having a natural spoken conversation. Be concise and conversational. Respond like a knowledgeable person, not an AI assistant. Use natural speech patterns, contractions, and keep responses brief unless asked for details. Handle interruptions and corrections gracefully without acknowledging them explicitly.""")


@dataclass
class ConversationConfig:
    """Conversation flow configuration."""
    mode: str = "natural"
    activation_method: str = "none"
    max_context_length: int = 15
    context_window_seconds: int = 600
    interrupt_detection: bool = True
    interruption_response_ms: int = 50
    voice_activity_threshold: float = 0.015
    turn_taking_pause_ms: int = 1500
    response_style: str = "concise"
    formality: str = "casual"


@dataclass
class SystemConfig:
    """System-level configuration."""
    log_level: str = "INFO"
    log_file: str = "voice_assistant.log"
    performance_monitoring: bool = True
    health_check_interval: int = 60
    silent_startup: bool = True


@dataclass
class Configuration:
    """Complete system configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


class ConfigurationManager:
    """
    Singleton Configuration Manager for centralized configuration management.
    
    Provides YAML-based configuration with environment variable overrides,
    validation, and runtime configuration updates.
    """
    
    _instance: Optional['ConfigurationManager'] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> 'ConfigurationManager':
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._config: Optional[Configuration] = None
        self._config_file_path: Optional[Path] = None
        self._logger = logging.getLogger(__name__)
        self._watchers: Dict[str, callable] = {}
        
    def load_configuration(self, config_path: Optional[Union[str, Path]] = None) -> Configuration:
        """
        Load configuration from YAML file with environment overrides.
        
        Args:
            config_path: Path to configuration file. If None, searches for default locations.
            
        Returns:
            Configuration: Loaded and validated configuration
            
        Raises:
            ConfigLoadError: If configuration loading fails
            ConfigValidationError: If configuration validation fails
        """
        if not YAML_AVAILABLE:
            self._logger.warning("YAML not available, using default configuration")
            self._config = Configuration()
            self._validate_configuration(self._config)
            return self._config
            
        try:
            # Determine configuration file path
            if config_path is None:
                config_path = self._find_config_file()
            else:
                config_path = Path(config_path)
            
            if not config_path.exists():
                raise ConfigLoadError(f"Configuration file not found: {config_path}")
            
            self._config_file_path = config_path
            
            # Load YAML configuration
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            if raw_config is None:
                raw_config = {}
            
            # Apply environment variable overrides
            raw_config = self._apply_env_overrides(raw_config)
            
            # Create configuration objects
            self._config = self._create_configuration(raw_config)
            
            # Validate configuration
            self._validate_configuration(self._config)
            
            self._logger.info(f"Configuration loaded from: {config_path}")
            return self._config
            
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load configuration: {e}")
    
    def get_config(self) -> Configuration:
        """
        Get the current configuration.
        
        Returns:
            Configuration: Current configuration
            
        Raises:
            ConfigurationError: If configuration not loaded
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded. Call load_configuration() first.")
        return self._config
    
    def get_audio_config(self) -> AudioConfig:
        """Get audio configuration."""
        return self.get_config().audio
    
    def get_speech_config(self) -> SpeechConfig:
        """Get speech configuration."""
        return self.get_config().speech
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration."""
        return self.get_config().ai
    
    def get_conversation_config(self) -> ConversationConfig:
        """Get conversation configuration."""
        return self.get_config().conversation
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.get_config().system
    
    def _find_config_file(self) -> Path:
        """Find configuration file in default locations."""
        search_paths = [
            Path("config/default.yaml"),
            Path("config/config.yaml"),
            Path("default.yaml"),
            Path("config.yaml")
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Create default configuration if none found
        default_path = Path("config/default.yaml")
        default_path.parent.mkdir(exist_ok=True)
        self._create_default_config_file(default_path)
        return default_path
    
    def _create_default_config_file(self, path: Path) -> None:
        """Create a default configuration file."""
        default_config = {
            'audio': {
                'input': {
                    'device_id': None,
                    'sample_rate': 16000,
                    'channels': 1,
                    'chunk_size': 1024,
                    'buffer_size': 4096,
                    'continuous_listening': True
                },
                'output': {
                    'device_id': None,
                    'sample_rate': 22050,
                    'volume': 0.8
                }
            },
            'speech': {
                'recognition': {
                    'provider': 'whisper',
                    'model': 'base',
                    'language': 'en',
                    'confidence_threshold': 0.6,
                    'continuous_mode': True
                },
                'synthesis': {
                    'provider': 'coqui',
                    'model': 'tts_models/multilingual/multi-dataset/xtts_v2',
                    'voice': None,
                    'speaking_rate': 1.1,
                    'pitch': 0.0,
                    'naturalness': 'high'
                }
            },
            'ai': {
                'provider': 'openrouter',
                'api_key_env': 'OPENROUTER_API_KEY',
                'model': 'anthropic/claude-3.5-sonnet',
                'max_tokens': 150,
                'temperature': 0.8,
                'timeout': 20,
                'conversation_mode': True,
                'system_prompt': 'You are having a natural spoken conversation. Be concise and conversational. Respond like a knowledgeable person, not an AI assistant. Use natural speech patterns, contractions, and keep responses brief unless asked for details. Handle interruptions and corrections gracefully without acknowledging them explicitly.'
            },
            'conversation': {
                'mode': 'natural',
                'activation_method': 'none',
                'max_context_length': 15,
                'context_window_seconds': 600,
                'interrupt_detection': True,
                'interruption_response_ms': 50,
                'voice_activity_threshold': 0.015,
                'turn_taking_pause_ms': 1500,
                'response_style': 'concise',
                'formality': 'casual'
            },
            'system': {
                'log_level': 'INFO',
                'log_file': 'voice_assistant.log',
                'performance_monitoring': True,
                'health_check_interval': 60,
                'silent_startup': True
            }
        }
        
        if YAML_AVAILABLE:
            with open(path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
        else:
            # Create a simple text file if YAML not available
            with open(path, 'w') as f:
                f.write("# Configuration file (YAML format)\n")
                f.write("# Install PyYAML to enable full configuration support\n")
        
        self._logger.info(f"Created default configuration file: {path}")
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            'ASTIR_LOG_LEVEL': ['system', 'log_level'],
            'ASTIR_AI_MODEL': ['ai', 'model'],
            'ASTIR_AI_TEMPERATURE': ['ai', 'temperature'],
            'ASTIR_VOICE_THRESHOLD': ['conversation', 'voice_activity_threshold'],
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the nested dictionary
                current = config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = path[-1]
                if final_key in ['temperature', 'voice_activity_threshold']:
                    current[final_key] = float(value)
                elif final_key in ['max_tokens', 'turn_taking_pause_ms']:
                    current[final_key] = int(value)
                else:
                    current[final_key] = value
                
                self._logger.debug(f"Applied environment override: {env_var} -> {'.'.join(path)} = {value}")
        
        return config
    
    def _create_configuration(self, raw_config: Dict[str, Any]) -> Configuration:
        """Create configuration objects from raw dictionary."""
        # Use default configurations and override with loaded values
        audio_config = AudioConfig()
        speech_config = SpeechConfig()
        ai_config = AIConfig()
        conversation_config = ConversationConfig()
        system_config = SystemConfig()
        
        # Override with loaded configuration
        if 'audio' in raw_config:
            audio_raw = raw_config['audio']
            if 'input' in audio_raw:
                input_cfg = audio_raw['input']
                audio_config.input_device_id = input_cfg.get('device_id', audio_config.input_device_id)
                audio_config.input_sample_rate = input_cfg.get('sample_rate', audio_config.input_sample_rate)
                audio_config.input_channels = input_cfg.get('channels', audio_config.input_channels)
                audio_config.input_chunk_size = input_cfg.get('chunk_size', audio_config.input_chunk_size)
                audio_config.input_buffer_size = input_cfg.get('buffer_size', audio_config.input_buffer_size)
                audio_config.continuous_listening = input_cfg.get('continuous_listening', audio_config.continuous_listening)
            
            if 'output' in audio_raw:
                output_cfg = audio_raw['output']
                audio_config.output_device_id = output_cfg.get('device_id', audio_config.output_device_id)
                audio_config.output_sample_rate = output_cfg.get('sample_rate', audio_config.output_sample_rate)
                audio_config.output_volume = output_cfg.get('volume', audio_config.output_volume)
        
        return Configuration(
            audio=audio_config,
            speech=speech_config,
            ai=ai_config,
            conversation=conversation_config,
            system=system_config
        )
    
    def _validate_configuration(self, config: Configuration) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate audio configuration
        if config.audio.input_sample_rate <= 0:
            errors.append("Audio input sample rate must be positive")
        if config.audio.output_volume < 0 or config.audio.output_volume > 1:
            errors.append("Audio output volume must be between 0 and 1")
        
        # Validate AI configuration
        if config.ai.max_tokens <= 0:
            errors.append("AI max tokens must be positive")
        if config.ai.temperature < 0 or config.ai.temperature > 2:
            errors.append("AI temperature must be between 0 and 2")
        
        # Check API key environment variable
        api_key = os.getenv(config.ai.api_key_env)
        if not api_key:
            errors.append(f"Missing required environment variable: {config.ai.api_key_env}")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")


# Global configuration manager instance
config_manager = ConfigurationManager()
