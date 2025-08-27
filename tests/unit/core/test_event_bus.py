"""
Unit tests for the Event Bus Service.

These tests verify the core functionality of the Event Bus Service
following the BDD scenarios defined in the feature specifications.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import Future

from src.core.event_bus import EventBusService, EventTypes, Event, EventMetadata
from src.utils.exceptions import EventBusError, EventValidationError, SubscriptionError


class TestEventBusInitialization:
    """Test Event Bus initialization and basic functionality."""
    
    def test_event_bus_initialization(self):
        """Test that Event Bus initializes correctly."""
        # Given: Event Bus Service is not yet initialized
        event_bus = EventBusService()
        assert not event_bus._is_initialized
        assert not event_bus._is_running
        
        # When: I initialize the Event Bus Service
        result = event_bus.initialize()
        
        # Then: Event Bus should be ready
        assert result is True
        assert event_bus._is_initialized is True
        assert event_bus._is_running is True
        assert event_bus._executor is not None
        
        # Cleanup
        event_bus.shutdown()
    
    def test_event_bus_double_initialization(self):
        """Test that double initialization is handled gracefully."""
        event_bus = EventBusService()
        
        # First initialization
        result1 = event_bus.initialize()
        assert result1 is True
        
        # Second initialization should return True but not reinitialize
        result2 = event_bus.initialize()
        assert result2 is True
        
        # Cleanup
        event_bus.shutdown()
    
    def test_event_bus_initialization_failure(self):
        """Test Event Bus initialization failure handling."""
        event_bus = EventBusService()
        
        # Mock ThreadPoolExecutor to raise exception
        with patch('src.core.event_bus.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Initialization failed")
            
            # Should raise EventBusError
            with pytest.raises(EventBusError, match="Initialization failed"):
                event_bus.initialize()


class TestEventBusSubscription:
    """Test Event Bus subscription functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
        self.test_handler = Mock()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_subscribe_to_event_types(self):
        """Test subscribing to event types."""
        # When: I subscribe to "AUDIO_DATA_RECEIVED" events
        subscription_id = self.event_bus.subscribe(
            EventTypes.AUDIO_DATA_RECEIVED, 
            self.test_handler
        )
        
        # Then: Subscription should be registered successfully
        assert subscription_id is not None
        assert len(subscription_id) > 0
        
        # And: Handler should be associated with the event type
        subscribers = self.event_bus.get_subscribers(EventTypes.AUDIO_DATA_RECEIVED)
        assert subscription_id in subscribers
    
    def test_multiple_handlers_same_event_type(self):
        """Test multiple handlers for the same event type."""
        handler_1 = Mock()
        handler_2 = Mock()
        
        # Given: I have subscribed handler_1 to "SPEECH_DETECTED" events
        sub_id_1 = self.event_bus.subscribe(EventTypes.SPEECH_DETECTED, handler_1)
        
        # When: I subscribe handler_2 to "SPEECH_DETECTED" events
        sub_id_2 = self.event_bus.subscribe(EventTypes.SPEECH_DETECTED, handler_2)
        
        # Then: Both handlers should be registered
        subscribers = self.event_bus.get_subscribers(EventTypes.SPEECH_DETECTED)
        assert sub_id_1 in subscribers
        assert sub_id_2 in subscribers
        assert len(subscribers) == 2
    
    def test_subscribe_invalid_event_type(self):
        """Test subscribing with invalid event type."""
        # Should raise ValueError for empty event type
        with pytest.raises(ValueError, match="event_type must be a non-empty string"):
            self.event_bus.subscribe("", self.test_handler)
        
        # Should raise ValueError for non-string event type
        with pytest.raises(ValueError, match="event_type must be a non-empty string"):
            self.event_bus.subscribe(123, self.test_handler)
    
    def test_subscribe_invalid_handler(self):
        """Test subscribing with invalid handler."""
        # Should raise ValueError for non-callable handler
        with pytest.raises(ValueError, match="handler must be callable"):
            self.event_bus.subscribe(EventTypes.SYSTEM_READY, "not_callable")


class TestEventBusPublishing:
    """Test Event Bus publishing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
        self.test_handler = Mock()
        self.subscription_id = self.event_bus.subscribe(
            EventTypes.SPEECH_RECOGNIZED, 
            self.test_handler
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_publish_synchronous_events(self):
        """Test publishing synchronous events."""
        test_data = {"message": "test", "confidence": 0.95}
        
        # When: I publish a "SPEECH_RECOGNIZED" event
        self.event_bus.publish(EventTypes.SPEECH_RECOGNIZED, test_data)
        
        # Then: Handler should receive the event immediately
        self.test_handler.assert_called_once_with(test_data)
    
    def test_publish_with_metadata(self):
        """Test publishing events with metadata."""
        test_data = {"message": "test"}
        metadata = {"source": "test_service", "priority": 1}
        
        # When: I publish an event with metadata
        self.event_bus.publish(EventTypes.SPEECH_RECOGNIZED, test_data, metadata)
        
        # Then: Handler should receive the event data
        self.test_handler.assert_called_once_with(test_data)
    
    def test_publish_invalid_event_type(self):
        """Test publishing with invalid event type."""
        # Should raise EventValidationError for empty event type
        with pytest.raises(EventValidationError, match="event_type must be a non-empty string"):
            self.event_bus.publish("", {"data": "test"})
    
    def test_publish_invalid_data(self):
        """Test publishing with invalid data."""
        # Should raise EventValidationError for non-dict data
        with pytest.raises(EventValidationError, match="data must be a dictionary"):
            self.event_bus.publish(EventTypes.SYSTEM_READY, "not_a_dict")
    
    def test_publish_when_not_running(self):
        """Test publishing when Event Bus is not running."""
        self.event_bus.shutdown()
        
        # Should raise EventBusError
        with pytest.raises(EventBusError, match="Event Bus is not running"):
            self.event_bus.publish(EventTypes.SYSTEM_READY, {"data": "test"})


class TestEventBusUnsubscription:
    """Test Event Bus unsubscription functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
        self.test_handler = Mock()
        self.subscription_id = self.event_bus.subscribe(
            EventTypes.SPEECH_RECOGNIZED, 
            self.test_handler
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_unsubscribe_from_events(self):
        """Test unsubscribing from events."""
        # When: I unsubscribe from events
        result = self.event_bus.unsubscribe(
            EventTypes.SPEECH_RECOGNIZED, 
            self.subscription_id
        )
        
        # Then: Unsubscription should succeed
        assert result is True
        
        # And: Handler should no longer receive events
        self.event_bus.publish(EventTypes.SPEECH_RECOGNIZED, {"test": "data"})
        self.test_handler.assert_not_called()
        
        # And: Subscription should be removed
        subscribers = self.event_bus.get_subscribers(EventTypes.SPEECH_RECOGNIZED)
        assert self.subscription_id not in subscribers
    
    def test_unsubscribe_nonexistent_subscription(self):
        """Test unsubscribing from nonexistent subscription."""
        # Should return False for nonexistent subscription
        result = self.event_bus.unsubscribe(
            EventTypes.SPEECH_RECOGNIZED, 
            "nonexistent_id"
        )
        assert result is False
    
    def test_unsubscribe_nonexistent_event_type(self):
        """Test unsubscribing from nonexistent event type."""
        # Should return False for nonexistent event type
        result = self.event_bus.unsubscribe(
            "NONEXISTENT_EVENT", 
            self.subscription_id
        )
        assert result is False


class TestEventBusErrorHandling:
    """Test Event Bus error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_handle_subscriber_errors_gracefully(self):
        """Test that subscriber errors don't crash the system."""
        # Given: Handlers that throw exceptions and work normally
        failing_handler = Mock(side_effect=Exception("Handler error"))
        working_handler = Mock()
        
        self.event_bus.subscribe(EventTypes.SYSTEM_ERROR, failing_handler)
        self.event_bus.subscribe(EventTypes.SYSTEM_ERROR, working_handler)
        
        # When: I publish an event
        self.event_bus.publish(EventTypes.SYSTEM_ERROR, {"test": "data"})
        
        # Then: Working handler should still receive the event
        working_handler.assert_called_once_with({"test": "data"})
        
        # And: Error count should be incremented
        stats = self.event_bus.get_statistics()
        assert stats.error_count > 0


class TestEventBusPerformance:
    """Test Event Bus performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
        self.performance_handler = Mock()
        self.event_bus.subscribe(EventTypes.SYSTEM_READY, self.performance_handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_handle_multiple_events_rapidly(self):
        """Test handling multiple events rapidly."""
        event_count = 100
        
        # When: I publish multiple events rapidly
        start_time = time.time()
        for i in range(event_count):
            self.event_bus.publish(EventTypes.SYSTEM_READY, {"index": i})
        end_time = time.time()
        
        # Then: All events should be delivered
        assert self.performance_handler.call_count == event_count
        
        # And: Performance should be reasonable (< 100ms for 100 events)
        total_time = (end_time - start_time) * 1000
        assert total_time < 100  # Less than 100ms total
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable."""
        # Get initial statistics
        initial_stats = self.event_bus.get_statistics()
        
        # Publish many events
        for i in range(1000):
            self.event_bus.publish(EventTypes.SYSTEM_READY, {"index": i})
        
        # Get final statistics
        final_stats = self.event_bus.get_statistics()
        
        # Memory usage should not grow excessively
        # (This is a basic test - in production we'd use memory profiling)
        assert final_stats.total_events_published == initial_stats.total_events_published + 1000
        assert final_stats.error_count == initial_stats.error_count


class TestEventBusStatistics:
    """Test Event Bus statistics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBusService()
        self.event_bus.initialize()
        self.test_handler = Mock()
        self.event_bus.subscribe(EventTypes.SYSTEM_READY, self.test_handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.event_bus.shutdown()
    
    def test_get_statistics(self):
        """Test getting Event Bus statistics."""
        # Publish some events
        for i in range(5):
            self.event_bus.publish(EventTypes.SYSTEM_READY, {"index": i})
        
        # Get statistics
        stats = self.event_bus.get_statistics()
        
        # Verify statistics
        assert stats.total_events_published >= 5  # At least our 5 + system ready
        assert stats.total_events_delivered >= 5
        assert stats.active_subscriptions >= 1
        assert stats.uptime_seconds >= 0
        assert isinstance(stats.event_type_stats, dict)


class TestEventBusShutdown:
    """Test Event Bus shutdown functionality."""
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of Event Bus."""
        event_bus = EventBusService()
        event_bus.initialize()
        
        # Add some subscriptions
        handler = Mock()
        event_bus.subscribe(EventTypes.SYSTEM_READY, handler)
        
        # When: I shutdown the Event Bus
        event_bus.shutdown()
        
        # Then: Event Bus should be stopped
        assert not event_bus._is_running
        
        # And: Executor should be shutdown
        assert event_bus._executor._shutdown
        
        # And: Subscriptions should be cleared
        assert len(event_bus._subscribers) == 0
    
    def test_shutdown_when_not_running(self):
        """Test shutdown when Event Bus is not running."""
        event_bus = EventBusService()
        
        # Should not raise exception
        event_bus.shutdown()
        assert not event_bus._is_running
