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
from .service_factory import (
    ServiceFactory,
    ServiceMetadata,
    ServiceLifetime,
    ServiceStatus,
    ServiceStatistics,
    ServiceFactoryError,
    DependencyResolutionError,
    CircularDependencyError,
    get_service_factory,
    initialize_service_factory
)
from .facade import VoiceAssistantFacade, VoiceAssistantState

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
    'config_manager',
    'ServiceFactory',
    'ServiceMetadata',
    'ServiceLifetime',
    'ServiceStatus',
    'ServiceStatistics',
    'ServiceFactoryError',
    'DependencyResolutionError',
    'CircularDependencyError',
    'get_service_factory',
    'initialize_service_factory',
    'VoiceAssistantFacade',
    'VoiceAssistantState'
]
