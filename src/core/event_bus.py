"""
Event Bus Service - Observer Pattern Implementation

This module implements the core event-driven communication system for the
Astir Voice Assistant, providing decoupled communication between components.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, List, Callable, Any, Optional, Set
from weakref import WeakSet

from ..utils.exceptions import (
    EventBusError, 
    EventValidationError, 
    SubscriptionError, 
    PublishingError
)


@dataclass
class EventMetadata:
    """Standard metadata included with all events."""
    timestamp: datetime
    source_service: str
    event_id: str
    correlation_id: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=critical


@dataclass
class Event:
    """Complete event structure."""
    type: str
    data: Dict[str, Any]
    metadata: EventMetadata


@dataclass
class EventBusStatistics:
    """Event bus performance statistics."""
    total_events_published: int
    total_events_delivered: int
    events_per_second: float
    average_delivery_time_ms: float
    active_subscriptions: int
    queue_size: int
    error_count: int
    uptime_seconds: int
    memory_usage_mb: float
    event_type_stats: Dict[str, Dict[str, Any]]


class EventTypes:
    """Standard event types used throughout the system."""
    
    # System lifecycle events
    SYSTEM_READY = "SYSTEM_READY"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    
    # Audio events
    AUDIO_DATA_RECEIVED = "AUDIO_DATA_RECEIVED"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    SPEECH_ENDED = "SPEECH_ENDED"
    INTERRUPTION_DETECTED = "INTERRUPTION_DETECTED"
    AUDIO_DEVICE_ERROR = "AUDIO_DEVICE_ERROR"
    
    # Speech recognition events
    SPEECH_RECOGNIZED = "SPEECH_RECOGNIZED"
    LOW_CONFIDENCE_RECOGNITION = "LOW_CONFIDENCE_RECOGNITION"
    RECOGNITION_ERROR = "RECOGNITION_ERROR"
    
    # AI conversation events
    AI_RESPONSE_RECEIVED = "AI_RESPONSE_RECEIVED"
    AI_ERROR = "AI_ERROR"
    CONVERSATION_STARTED = "CONVERSATION_STARTED"
    CONVERSATION_ENDED = "CONVERSATION_ENDED"
    
    # TTS events
    TTS_AUDIO_GENERATED = "TTS_AUDIO_GENERATED"
    TTS_ERROR = "TTS_ERROR"
    
    # Playback events
    PLAYBACK_STARTED = "PLAYBACK_STARTED"
    PLAYBACK_FINISHED = "PLAYBACK_FINISHED"
    PLAYBACK_INTERRUPTED = "PLAYBACK_INTERRUPTED"
    PLAYBACK_ERROR = "PLAYBACK_ERROR"
    
    # Configuration events
    CONFIG_CHANGED = "CONFIG_CHANGED"
    CONFIG_ERROR = "CONFIG_ERROR"


class EventBusService:
    """
    Event-driven communication service implementing the Observer pattern.
    Provides synchronous and asynchronous event publishing with type-safe handlers.
    """
    
    def __init__(self, max_queue_size: int = 1000, worker_threads: int = 4):
        """
        Initialize the Event Bus Service.
        
        Args:
            max_queue_size: Maximum number of queued async events
            worker_threads: Number of worker threads for async processing
        """
        self._max_queue_size = max_queue_size
        self._worker_threads = worker_threads
        
        # Core state
        self._is_initialized = False
        self._is_running = False
        self._shutdown_event = threading.Event()
        
        # Subscription management
        self._subscribers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self._subscription_lock = threading.RLock()
        
        # Async event processing
        self._event_queue: Queue = Queue(maxsize=max_queue_size)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_futures: List = []
        
        # Statistics and monitoring
        self._start_time = time.time()
        self._stats = {
            'total_events_published': 0,
            'total_events_delivered': 0,
            'error_count': 0,
            'event_type_stats': defaultdict(lambda: {
                'count': 0,
                'total_time_ms': 0,
                'errors': 0
            })
        }
        self._stats_lock = threading.Lock()
        
        # Logging
        self._logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """
        Initialize the event bus and start worker threads.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            EventBusError: If initialization fails
        """
        if self._is_initialized:
            self._logger.warning("Event Bus already initialized")
            return True
            
        try:
            # Start thread pool for async processing
            self._executor = ThreadPoolExecutor(
                max_workers=self._worker_threads,
                thread_name_prefix="EventBus"
            )
            
            # Start worker threads
            for i in range(self._worker_threads):
                future = self._executor.submit(self._async_worker)
                self._worker_futures.append(future)
            
            self._is_initialized = True
            self._is_running = True
            self._start_time = time.time()
            
            self._logger.info(
                f"Event Bus initialized with {self._worker_threads} worker threads"
            )
            
            # Publish system ready event
            self.publish(EventTypes.SYSTEM_READY, {
                'timestamp': datetime.now().isoformat(),
                'worker_threads': self._worker_threads,
                'max_queue_size': self._max_queue_size
            })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Event Bus: {e}")
            raise EventBusError(f"Initialization failed: {e}")
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the event bus.
        Processes all pending events before stopping.
        """
        if not self._is_running:
            return
            
        self._logger.info("Shutting down Event Bus...")
        
        # Signal shutdown
        self._is_running = False
        self._shutdown_event.set()
        
        # Process remaining events
        try:
            while not self._event_queue.empty():
                time.sleep(0.01)  # Allow workers to process
        except:
            pass
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Clear subscriptions
        with self._subscription_lock:
            self._subscribers.clear()
        
        self._logger.info("Event Bus shutdown complete")
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: Callable that processes the event data
            
        Returns:
            str: Subscription ID for later unsubscription
            
        Raises:
            ValueError: If event_type is invalid or handler is not callable
        """
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type must be a non-empty string")
        
        if not callable(handler):
            raise ValueError("handler must be callable")
        
        subscription_id = str(uuid.uuid4())
        
        with self._subscription_lock:
            self._subscribers[event_type][subscription_id] = handler
        
        self._logger.debug(
            f"Subscribed {handler.__name__} to {event_type} "
            f"(ID: {subscription_id[:8]}...)"
        )
        
        return subscription_id
    
    def unsubscribe(self, event_type: str, subscription_id: str) -> bool:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: The event type to unsubscribe from
            subscription_id: The subscription ID returned by subscribe()
            
        Returns:
            bool: True if unsubscribed successfully, False if not found
        """
        with self._subscription_lock:
            if event_type in self._subscribers:
                if subscription_id in self._subscribers[event_type]:
                    del self._subscribers[event_type][subscription_id]
                    
                    # Clean up empty event type
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
                    
                    self._logger.debug(
                        f"Unsubscribed {subscription_id[:8]}... from {event_type}"
                    )
                    return True
        
        return False
    
    def publish(self, event_type: str, data: Dict[str, Any], 
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an event synchronously to all subscribers.
        
        Args:
            event_type: The type of event being published
            data: Event payload data
            metadata: Optional metadata (timestamp, source, etc.)
            
        Raises:
            EventValidationError: If event data is invalid
            EventBusError: If publishing fails
        """
        if not self._is_running:
            raise EventBusError("Event Bus is not running")
        
        # Validate inputs
        if not event_type or not isinstance(event_type, str):
            raise EventValidationError("event_type must be a non-empty string")
        
        if not isinstance(data, dict):
            raise EventValidationError("data must be a dictionary")
        
        # Create event metadata
        event_metadata = EventMetadata(
            timestamp=datetime.now(),
            source_service=metadata.get('source', 'unknown') if metadata else 'unknown',
            event_id=str(uuid.uuid4()),
            correlation_id=metadata.get('correlation_id') if metadata else None,
            priority=metadata.get('priority', 0) if metadata else 0
        )
        
        # Create event
        event = Event(
            type=event_type,
            data=data,
            metadata=event_metadata
        )
        
        # Deliver to subscribers
        self._deliver_event_sync(event)
    
    async def publish_async(self, event_type: str, data: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an event asynchronously to all subscribers.
        
        Args:
            event_type: The type of event being published
            data: Event payload data
            metadata: Optional metadata
            
        Raises:
            EventValidationError: If event data is invalid
            EventBusError: If publishing fails
        """
        if not self._is_running:
            raise EventBusError("Event Bus is not running")
        
        # Create event (same validation as sync)
        event_metadata = EventMetadata(
            timestamp=datetime.now(),
            source_service=metadata.get('source', 'unknown') if metadata else 'unknown',
            event_id=str(uuid.uuid4()),
            correlation_id=metadata.get('correlation_id') if metadata else None,
            priority=metadata.get('priority', 0) if metadata else 0
        )
        
        event = Event(
            type=event_type,
            data=data,
            metadata=event_metadata
        )
        
        # Queue for async processing
        try:
            self._event_queue.put_nowait(event)
        except:
            raise PublishingError("Event queue is full")
    
    def get_subscribers(self, event_type: str) -> List[str]:
        """
        Get list of subscription IDs for an event type.
        
        Args:
            event_type: The event type to query
            
        Returns:
            List[str]: List of subscription IDs
        """
        with self._subscription_lock:
            if event_type in self._subscribers:
                return list(self._subscribers[event_type].keys())
        return []
    
    def get_statistics(self) -> EventBusStatistics:
        """
        Get event bus performance and usage statistics.
        
        Returns:
            EventBusStatistics: Current statistics
        """
        with self._stats_lock:
            uptime = time.time() - self._start_time
            events_per_second = self._stats['total_events_published'] / max(uptime, 1)
            
            # Calculate average delivery time
            total_time = sum(
                stats['total_time_ms'] 
                for stats in self._stats['event_type_stats'].values()
            )
            total_events = max(self._stats['total_events_delivered'], 1)
            avg_delivery_time = total_time / total_events
            
            return EventBusStatistics(
                total_events_published=self._stats['total_events_published'],
                total_events_delivered=self._stats['total_events_delivered'],
                events_per_second=events_per_second,
                average_delivery_time_ms=avg_delivery_time,
                active_subscriptions=sum(
                    len(handlers) for handlers in self._subscribers.values()
                ),
                queue_size=self._event_queue.qsize(),
                error_count=self._stats['error_count'],
                uptime_seconds=int(uptime),
                memory_usage_mb=0.0,  # TODO: Implement memory tracking
                event_type_stats=dict(self._stats['event_type_stats'])
            )
    
    def _deliver_event_sync(self, event: Event) -> None:
        """Deliver event synchronously to all subscribers."""
        start_time = time.time()
        
        with self._subscription_lock:
            subscribers = self._subscribers.get(event.type, {}).copy()
        
        if not subscribers:
            return
        
        # Update statistics
        with self._stats_lock:
            self._stats['total_events_published'] += 1
        
        # Deliver to each subscriber
        for subscription_id, handler in subscribers.items():
            try:
                handler(event.data)
                
                with self._stats_lock:
                    self._stats['total_events_delivered'] += 1
                    
            except Exception as e:
                self._logger.error(
                    f"Error in event handler {subscription_id[:8]}... "
                    f"for {event.type}: {e}"
                )
                
                with self._stats_lock:
                    self._stats['error_count'] += 1
                    self._stats['event_type_stats'][event.type]['errors'] += 1
        
        # Update timing statistics
        delivery_time_ms = (time.time() - start_time) * 1000
        with self._stats_lock:
            stats = self._stats['event_type_stats'][event.type]
            stats['count'] += 1
            stats['total_time_ms'] += delivery_time_ms
    
    def _async_worker(self) -> None:
        """Worker thread for processing async events."""
        self._logger.debug(f"Async worker started: {threading.current_thread().name}")
        
        while self._is_running:
            try:
                # Get event from queue with timeout
                event = self._event_queue.get(timeout=0.1)
                
                # Deliver event
                self._deliver_event_sync(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self._logger.error(f"Error in async worker: {e}")
                with self._stats_lock:
                    self._stats['error_count'] += 1
        
        self._logger.debug(f"Async worker stopped: {threading.current_thread().name}")
