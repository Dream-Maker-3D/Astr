"""
Core services for the Astir Voice Assistant.

This package contains the fundamental services that provide the foundation
for the voice assistant system, including event bus, configuration, and facade.
"""

from .event_bus import EventBusService, EventTypes, Event, EventMetadata, EventBusStatistics
from .config_manager import (
    ConfigurationManager, 
    Configuration, 
    AudioConfig, 
    SpeechConfig, 
    AIConfig, 
    ConversationConfig, 
    SystemConfig,
    config_manager
)

__all__ = [
    'EventBusService',
    'EventTypes', 
    'Event',
    'EventMetadata',
    'EventBusStatistics',
    'ConfigurationManager',
    'Configuration',
    'AudioConfig',
    'SpeechConfig', 
    'AIConfig',
    'ConversationConfig',
    'SystemConfig',
    'config_manager'
]
