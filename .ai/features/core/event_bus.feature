@core @event_bus
Feature: Event Bus Service - Observer Pattern Implementation
  As a system architect
  I want a decoupled event-driven communication system
  So that components can communicate without tight coupling

  Background:
    Given the Event Bus Service is initialized
    And the system is in a clean state
    And no events are pending

  @smoke @critical
  Scenario: Event Bus initialization and basic functionality
    Given the Event Bus Service is not yet initialized
    When I initialize the Event Bus Service
    Then the Event Bus should be ready to accept subscriptions
    And the Event Bus should be ready to publish events
    And the Event Bus should have no active subscriptions

  @subscription
  Scenario: Subscribe to event types
    Given the Event Bus Service is running
    When I subscribe to "AUDIO_DATA_RECEIVED" events with handler "audio_handler"
    Then the subscription should be registered successfully
    And the handler should be associated with the event type
    And the Event Bus should confirm the subscription

  @subscription
  Scenario: Multiple handlers for same event type
    Given the Event Bus Service is running
    And I have subscribed "handler_1" to "SPEECH_DETECTED" events
    When I subscribe "handler_2" to "SPEECH_DETECTED" events
    Then both handlers should be registered for "SPEECH_DETECTED"
    And both handlers should receive events when "SPEECH_DETECTED" is published
    And the order of handler execution should be deterministic

  @publishing @synchronous
  Scenario: Publish synchronous events
    Given the Event Bus Service is running
    And I have subscribed "test_handler" to "TEST_EVENT" events
    When I publish a "TEST_EVENT" with data {"message": "test"}
    Then the "test_handler" should receive the event immediately
    And the event data should contain {"message": "test"}
    And the event should include a timestamp
    And the event should include the event type "TEST_EVENT"

  @publishing @asynchronous
  Scenario: Publish asynchronous events
    Given the Event Bus Service is running
    And I have subscribed "async_handler" to "ASYNC_TEST_EVENT" events
    When I publish an async "ASYNC_TEST_EVENT" with data {"async": true}
    Then the "async_handler" should receive the event asynchronously
    And the event processing should not block the publisher
    And the event should be delivered within 100ms

  @unsubscription
  Scenario: Unsubscribe from events
    Given the Event Bus Service is running
    And I have subscribed "temp_handler" to "TEMP_EVENT" events
    When I unsubscribe "temp_handler" from "TEMP_EVENT" events
    Then the handler should no longer receive "TEMP_EVENT" events
    And the subscription should be removed from the Event Bus
    And publishing "TEMP_EVENT" should not call "temp_handler"

  @error_handling
  Scenario: Handle subscriber errors gracefully
    Given the Event Bus Service is running
    And I have subscribed "failing_handler" that throws exceptions to "ERROR_TEST" events
    And I have subscribed "working_handler" to "ERROR_TEST" events
    When I publish an "ERROR_TEST" event
    Then the "failing_handler" should receive the event and throw an exception
    And the exception should be logged but not propagate
    And the "working_handler" should still receive and process the event
    And the Event Bus should continue operating normally

  @performance
  Scenario: Handle high-frequency events
    Given the Event Bus Service is running
    And I have subscribed "performance_handler" to "HIGH_FREQ_EVENT" events
    When I publish 1000 "HIGH_FREQ_EVENT" events rapidly
    Then all events should be delivered to "performance_handler"
    And the average delivery time should be less than 1ms per event
    And no events should be lost
    And memory usage should remain stable

  @integration @audio
  Scenario: Audio pipeline event flow
    Given the Event Bus Service is running
    And I have subscribed handlers for the audio pipeline:
      | Event Type              | Handler              |
      | AUDIO_DATA_RECEIVED     | speech_recognition   |
      | SPEECH_DETECTED         | conversation_manager |
      | SPEECH_RECOGNIZED       | ai_service           |
      | AI_RESPONSE_RECEIVED    | tts_service          |
      | TTS_AUDIO_GENERATED     | audio_player         |
    When I publish an "AUDIO_DATA_RECEIVED" event with audio data
    Then the event should trigger the complete audio pipeline
    And each handler should receive the appropriate event
    And the pipeline should complete within 3 seconds
    And the final "TTS_AUDIO_GENERATED" event should be published

  @cleanup
  Scenario: Event Bus cleanup and shutdown
    Given the Event Bus Service is running
    And there are active subscriptions and pending events
    When I shutdown the Event Bus Service
    Then all pending events should be processed before shutdown
    And all subscriptions should be cleanly removed
    And no memory leaks should occur
    And the Event Bus should be in a clean shutdown state
