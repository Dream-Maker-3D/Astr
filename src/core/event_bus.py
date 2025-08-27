import asyncio
import logging
from typing import Dict, List, Callable, Any, Awaitable
from collections import defaultdict
from enum import Enum

from ..utils.exceptions import VoiceAssistantError


class EventTypes:
    """Event type constants for the voice assistant system."""
    
    # Audio events
    AUDIO_DATA_RECEIVED = "audio.data.received"
    AUDIO_PLAYBACK_START = "audio.playback.start"
    AUDIO_PLAYBACK_COMPLETE = "audio.playback.complete"
    AUDIO_DEVICE_ERROR = "audio.device.error"
    
    # Speech events
    SPEECH_DETECTED = "speech.detected"
    SPEECH_RECOGNIZED = "speech.recognized"
    SPEECH_SYNTHESIS_START = "speech.synthesis.start"
    SPEECH_SYNTHESIS_COMPLETE = "speech.synthesis.complete"
    SPEECH_INTERRUPTED = "speech.interrupted"
    
    # AI events
    AI_REQUEST_SENT = "ai.request.sent"
    AI_RESPONSE_RECEIVED = "ai.response.received"
    AI_ERROR = "ai.error"
    
    # Conversation events (natural flow)
    NATURAL_SPEECH_DETECTED = "speech.natural.detected"
    TURN_BOUNDARY_DETECTED = "turn.boundary.detected"
    INTERRUPTION_DETECTED = "interruption.detected"
    CLARIFICATION_NEEDED = "clarification.needed"
    TOPIC_SHIFT_DETECTED = "topic.shift.detected"
    
    # Response events
    RESPONSE_STREAMING_START = "response.streaming.start"
    RESPONSE_INTERRUPTED = "response.interrupted"
    RESPONSE_RESUMED = "response.resumed"
    
    # System events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_ENDED = "conversation.ended"
    CONVERSATION_PAUSE = "conversation.pause"
    CONVERSATION_RESUME = "conversation.resume"
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS_CHANGED = "system.status.changed"
    CONTEXT_UPDATED = "context.updated"


class EventBusService:
    """
    Central event bus implementing Observer pattern for decoupled communication.
    Supports both synchronous and asynchronous event handlers.
    """
    
    def __init__(self):
        self._sync_observers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_observers: Dict[str, List[Callable[..., Awaitable]]] = defaultdict(list)
        self._logger = logging.getLogger(__name__)
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe a synchronous handler to an event type."""
        if asyncio.iscoroutinefunction(handler):
            self._async_observers[event_type].append(handler)
            self._logger.debug(f"Subscribed async handler to {event_type}")
        else:
            self._sync_observers[event_type].append(handler)
            self._logger.debug(f"Subscribed sync handler to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        try:
            if handler in self._sync_observers[event_type]:
                self._sync_observers[event_type].remove(handler)
                self._logger.debug(f"Unsubscribed sync handler from {event_type}")
            elif handler in self._async_observers[event_type]:
                self._async_observers[event_type].remove(handler)
                self._logger.debug(f"Unsubscribed async handler from {event_type}")
        except ValueError:
            self._logger.warning(f"Handler not found for event type: {event_type}")
    
    def publish(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Publish an event synchronously to all sync handlers."""
        if data is None:
            data = {}
        
        event_data = {
            'type': event_type,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self._add_to_history(event_data)
        self._logger.debug(f"Publishing event: {event_type}")
        
        # Notify synchronous observers
        for handler in self._sync_observers[event_type]:
            try:
                handler(event_data)
            except Exception as e:
                self._logger.error(f"Error in sync event handler for {event_type}: {e}")
    
    async def publish_async(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Publish an event asynchronously to all handlers."""
        if data is None:
            data = {}
        
        event_data = {
            'type': event_type,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self._add_to_history(event_data)
        self._logger.debug(f"Publishing async event: {event_type}")
        
        # Notify synchronous observers
        for handler in self._sync_observers[event_type]:
            try:
                handler(event_data)
            except Exception as e:
                self._logger.error(f"Error in sync event handler for {event_type}: {e}")
        
        # Notify asynchronous observers
        tasks = []
        for handler in self._async_observers[event_type]:
            try:
                task = asyncio.create_task(handler(event_data))
                tasks.append(task)
            except Exception as e:
                self._logger.error(f"Error creating async task for {event_type}: {e}")
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self._logger.error(f"Error in async event handlers for {event_type}: {e}")
    
    def _add_to_history(self, event_data: Dict[str, Any]) -> None:
        """Add event to history with size limit."""
        self._event_history.append(event_data)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
    
    def get_event_history(self, event_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent event history, optionally filtered by event type."""
        if event_type:
            filtered_events = [e for e in self._event_history if e['type'] == event_type]
            return filtered_events[-limit:]
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type."""
        return (len(self._sync_observers[event_type]) + 
                len(self._async_observers[event_type]))
    
    def get_all_event_types(self) -> List[str]:
        """Get all event types that have subscribers."""
        all_types = set(self._sync_observers.keys()) | set(self._async_observers.keys())
        return list(all_types)