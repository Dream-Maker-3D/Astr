@core @factory
Feature: Service Factory - Dependency Injection and Service Management
  As a voice assistant system
  I want a centralized service factory for dependency injection
  So that services can be created, configured, and managed consistently

  Background:
    Given the Service Factory is available
    And the Configuration Manager is initialized
    And the Event Bus Service is running
    And the system is ready for service creation

  @smoke @critical
  Scenario: Service Factory initialization
    Given the Service Factory is not yet initialized
    When I initialize the Service Factory
    Then the factory should register all available service types
    And the factory should validate service dependencies
    And the factory should be ready to create services
    And the factory should support singleton and transient service lifetimes

  @service_creation
  Scenario: Create core services
    Given the Service Factory is initialized
    When I request creation of core services:
      | Service Type           | Lifetime  | Dependencies              |
      | EventBusService        | Singleton | None                      |
      | ConfigurationManager   | Singleton | None                      |
      | AudioCaptureService    | Singleton | EventBus, AudioConfig     |
      | AudioPlayerService     | Singleton | EventBus, AudioConfig     |
    Then each service should be created successfully
    And singleton services should return the same instance on subsequent calls
    And all dependencies should be properly injected
    And services should be fully initialized and ready

  @dependency_injection
  Scenario: Automatic dependency resolution
    Given the Service Factory is initialized
    And I have services with complex dependencies:
      | Service               | Requires                    |
      | SpeechRecognitionSvc  | EventBus, AudioCapture, STT |
      | SpeechSynthesisSvc    | EventBus, AudioPlayer, TTS  |
      | ConversationService   | EventBus, Recognition, AI   |
    When I request creation of "ConversationService"
    Then the factory should automatically resolve all dependencies
    And create required services in the correct order
    And inject dependencies into each service
    And return a fully configured ConversationService

  @service_registration
  Scenario: Register new service types
    Given the Service Factory is initialized
    When I register a new service type:
      | Service Class      | Interface           | Lifetime  | Dependencies |
      | CustomAudioFilter  | IAudioProcessor     | Transient | AudioConfig  |
      | AdvancedVAD        | IVoiceDetector      | Singleton | EventBus     |
      | CloudTTSProvider   | ISpeechSynthesizer  | Singleton | APIConfig    |
    Then the new service types should be registered successfully
    And the factory should be able to create instances of new services
    And dependency resolution should work for new services
    And service metadata should be available for introspection

  @service_lifecycle
  Scenario: Service lifecycle management
    Given the Service Factory is initialized
    And I have created several services
    When I request service lifecycle operations:
      | Operation | Service Type        | Expected Result    |
      | Create    | AudioCaptureService | New instance       |
      | Get       | AudioCaptureService | Same instance      |
      | Recreate  | AudioCaptureService | Fresh instance     |
      | Dispose   | AudioCaptureService | Cleanup completed  |
    Then each operation should execute successfully
    And singleton services should maintain single instance semantics
    And disposed services should be properly cleaned up
    And service references should be updated correctly

  @configuration_integration
  Scenario: Configuration-driven service creation
    Given the Service Factory is initialized
    And the configuration specifies service settings:
      | Service              | Setting           | Value              |
      | AudioCaptureService  | input_device_id   | 2                  |
      | AudioPlayerService   | output_volume     | 0.8                |
      | SpeechRecognition    | model_size        | base               |
      | SpeechSynthesis      | voice_id          | female_young       |
    When I create services through the factory
    Then services should be configured according to the settings
    And configuration changes should be applied automatically
    And invalid configurations should be rejected with clear errors
    And default values should be used for missing settings

  @service_discovery
  Scenario: Service discovery and introspection
    Given the Service Factory is initialized
    And multiple services are registered
    When I query the factory for service information
    Then I should be able to list all registered service types
    And get metadata for each service (dependencies, lifetime, interface)
    And check if a specific service type is available
    And get the current instance count for each service type
    And retrieve service creation statistics

  @error_handling
  Scenario: Service creation error handling
    Given the Service Factory is initialized
    When service creation encounters errors:
      | Error Type              | Scenario                    | Expected Behavior           |
      | Missing Dependency      | Required service not found  | Clear error with details    |
      | Circular Dependency     | Service A needs B needs A   | Detect and report cycle     |
      | Configuration Error     | Invalid service config      | Validation error with fix   |
      | Initialization Failure  | Service init throws error   | Graceful failure handling   |
    Then appropriate exceptions should be thrown
    And error messages should be clear and actionable
    And the factory should remain in a consistent state
    And other services should not be affected

  @performance_optimization
  Scenario: Factory performance optimization
    Given the Service Factory is initialized
    When I perform intensive service operations:
      | Operation           | Count | Time Limit |
      | Create services     | 100   | <100ms     |
      | Resolve dependencies| 50    | <50ms      |
      | Service lookups     | 1000  | <10ms      |
    Then all operations should complete within time limits
    And memory usage should remain stable
    And CPU usage should be minimal
    And service creation should be thread-safe

  @mock_services
  Scenario: Mock service support for testing
    Given the Service Factory is initialized
    And I want to test with mock services
    When I register mock implementations:
      | Interface           | Mock Implementation    | Test Scenario        |
      | IAudioCapture       | MockAudioCapture       | No microphone        |
      | IAudioPlayer        | MockAudioPlayer        | No speakers          |
      | ISpeechRecognition  | MockSTTService         | Offline testing      |
      | ISpeechSynthesis    | MockTTSService         | Silent testing       |
    Then mock services should be created instead of real ones
    And the system should function normally with mocks
    And test scenarios should be isolated and predictable
    And mock services should support test verification

  @plugin_support
  Scenario: Plugin service registration
    Given the Service Factory is initialized
    And I have external plugin services:
      | Plugin Name       | Service Type        | Plugin Path           |
      | CustomSTT         | ISpeechRecognition  | plugins/custom_stt    |
      | CloudTTS          | ISpeechSynthesis    | plugins/cloud_tts     |
      | AdvancedVAD       | IVoiceDetector      | plugins/advanced_vad  |
    When I load plugins through the factory
    Then plugin services should be registered automatically
    And plugin dependencies should be resolved correctly
    And plugins should integrate seamlessly with core services
    And plugin failures should not crash the system

  @service_monitoring
  Scenario: Service health monitoring
    Given the Service Factory is initialized
    And services are created and running
    When I monitor service health
    Then I should be able to check service status (running/stopped/error)
    And get service performance metrics (creation time, memory usage)
    And receive notifications when services fail or restart
    And access service logs and diagnostic information
    And trigger service health checks on demand

  @concurrent_access
  Scenario: Thread-safe service creation
    Given the Service Factory is initialized
    When multiple threads simultaneously request services:
      | Thread | Service Type        | Operation |
      | 1      | AudioCaptureService | Create    |
      | 2      | AudioCaptureService | Get       |
      | 3      | AudioPlayerService  | Create    |
      | 4      | EventBusService     | Get       |
    Then all operations should complete successfully
    And singleton services should maintain single instance
    And no race conditions should occur
    And service creation should be atomic
    And thread safety should be maintained throughout

  @service_replacement
  Scenario: Hot service replacement
    Given the Service Factory is initialized
    And services are running in production
    When I need to replace a service implementation:
      | Current Service     | New Service         | Replacement Type |
      | BasicSTTService     | AdvancedSTTService  | Upgrade          |
      | LocalTTSService     | CloudTTSService     | Migration        |
      | SimpleVAD           | MLBasedVAD          | Enhancement      |
    Then the factory should support graceful service replacement
    And existing service instances should be properly disposed
    And new services should maintain the same interface
    And dependent services should be notified of changes
    And the replacement should be transparent to clients

  @integration_testing
  Scenario: Factory integration with Event Bus
    Given the Service Factory is integrated with the Event Bus
    When service lifecycle events occur:
      | Event Type           | Trigger                | Expected Publication |
      | Service Created      | New service instance   | SERVICE_CREATED      |
      | Service Initialized  | Service ready          | SERVICE_READY        |
      | Service Failed       | Service error          | SERVICE_ERROR        |
      | Service Disposed     | Service cleanup        | SERVICE_DISPOSED     |
    Then appropriate events should be published to the Event Bus
    And event data should include service metadata
    And other services should be able to react to lifecycle events
    And the factory should handle event publication failures gracefully
