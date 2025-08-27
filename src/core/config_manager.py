import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from ..utils.exceptions import ConfigurationError


class ConfigurationManager:
    """Singleton configuration manager following GoF Singleton pattern."""
    
    _instance: Optional['ConfigurationManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ConfigurationManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config_data: Dict[str, Any] = {}
            self._config_path: Optional[Path] = None
            self._load_environment()
            ConfigurationManager._initialized = True
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()
    
    def load_config(self, config_path: str = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/default.yaml"
        
        self._config_path = Path(config_path)
        
        if not self._config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(self._config_path, 'r') as file:
                self._config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'audio.input.sample_rate')."""
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable value."""
        return os.getenv(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration section."""
        return self.get('audio', {})
    
    def get_speech_config(self) -> Dict[str, Any]:
        """Get speech configuration section."""
        return self.get('speech', {})
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration section."""
        return self.get('ai', {})
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation configuration section."""
        return self.get('conversation', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration section."""
        return self.get('system', {})
    
    def validate_required_config(self) -> None:
        """Validate that all required configuration is present."""
        required_keys = [
            'ai.api_key_env',
            'audio.input.sample_rate',
            'audio.output.sample_rate',
            'speech.recognition.provider',
            'speech.synthesis.provider'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration: {missing_keys}")
        
        # Validate API key exists in environment
        api_key_env = self.get('ai.api_key_env')
        if not self.get_env(api_key_env):
            raise ConfigurationError(f"Missing environment variable: {api_key_env}")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        if self._config_path:
            self.load_config(str(self._config_path))