@audio @capture
Feature: Audio Capture Service - Continuous Voice Input
  As a voice assistant user
  I want the system to continuously listen for my voice
  So that I can have natural conversations without activation words

  Background:
    Given the Audio Capture Service is initialized
    And the microphone is available and accessible
    And the Event Bus Service is running
    And the system is ready to capture audio

  @smoke @critical
  Scenario: Audio Capture Service initialization
    Given the Audio Capture Service is not yet initialized
    When I initialize the Audio Capture Service
    Then the service should detect available audio input devices
    And the service should select the default microphone
    And the service should configure audio parameters (16kHz, mono, 16-bit)
    And the service should be ready to start capturing

  @device_selection
  Scenario: Audio input device selection
    Given multiple audio input devices are available:
      | Device Name          | Device ID | Status    |
      | Built-in Microphone  | 0         | Available |
      | USB Headset Mic      | 1         | Available |
      | Bluetooth Headset    | 2         | Available |
    When I configure the Audio Capture Service to use device "1"
    Then the service should use "USB Headset Mic" for audio input
    And the service should confirm the device selection
    And the service should test the selected device

  @continuous_listening
  Scenario: Start continuous audio listening
    Given the Audio Capture Service is initialized
    When I start continuous listening
    Then the service should begin capturing audio continuously
    And audio data should be captured in real-time
    And the service should publish "AUDIO_DATA_RECEIVED" events
    And the audio buffer should maintain a rolling window of recent audio

  @voice_activity_detection
  Scenario: Voice Activity Detection (VAD)
    Given the Audio Capture Service is running in continuous mode
    And the VAD threshold is set to 0.005
    When I speak into the microphone with normal volume
    Then the service should detect voice activity
    And a "SPEECH_DETECTED" event should be published
    And the service should start accumulating speech audio
    And the speech detection confidence should be included

  @voice_activity_detection
  Scenario: Background noise filtering
    Given the Audio Capture Service is running in continuous mode
    And there is background noise below the VAD threshold
    When background noise is present without speech
    Then the service should not detect voice activity
    And no "SPEECH_DETECTED" events should be published
    And the audio buffer should continue updating normally
    And system resources should remain efficient

  @speech_segmentation
  Scenario: Speech segment detection and processing
    Given the Audio Capture Service is running
    And voice activity has been detected
    When I speak a complete sentence and then pause for 800ms
    Then the service should detect the end of the speech segment
    And the accumulated speech audio should be published as "AUDIO_DATA_RECEIVED"
    And the audio data should include the complete speech segment
    And the service should reset for the next speech segment

  @interruption_handling
  Scenario: Handle speech interruptions
    Given the Audio Capture Service is running
    And the system is currently speaking (TTS output)
    When I start speaking (interrupting the system)
    Then the service should immediately detect my voice activity
    And an "INTERRUPTION_DETECTED" event should be published
    And the system should stop its current speech output
    And my speech should be captured and processed normally

  @audio_quality
  Scenario: Audio quality and preprocessing
    Given the Audio Capture Service is running
    When audio is captured from the microphone
    Then the audio should be preprocessed for optimal quality:
      | Processing Step      | Expected Result           |
      | Noise Reduction      | Background noise reduced  |
      | Normalization        | Consistent volume levels  |
      | Format Conversion    | 16kHz mono 16-bit format |
      | Buffer Management    | Smooth audio flow         |

  @performance @latency
  Scenario: Low-latency audio capture
    Given the Audio Capture Service is running
    When I speak into the microphone
    Then the audio should be captured with minimal latency
    And the time from speech to "SPEECH_DETECTED" event should be < 100ms
    And the audio processing should not introduce noticeable delays
    And the system should maintain real-time performance

  @error_handling @device_failure
  Scenario: Handle microphone device failures
    Given the Audio Capture Service is running
    When the microphone device becomes unavailable
    Then the service should detect the device failure
    And a "AUDIO_DEVICE_ERROR" event should be published
    And the service should attempt to reconnect to the device
    And if reconnection fails, the service should try alternative devices
    And the user should be notified of the audio issue

  @error_handling @permission_denied
  Scenario: Handle microphone permission issues
    Given the Audio Capture Service is attempting to initialize
    When microphone permissions are denied by the system
    Then the service should detect the permission issue
    And a clear error message should be provided to the user
    And the service should provide instructions for granting permissions
    And the service should gracefully handle the permission denial

  @configuration
  Scenario: Audio capture configuration
    Given the Audio Capture Service is initialized
    When I configure the audio capture settings:
      | Setting              | Value    |
      | Sample Rate          | 16000    |
      | Channels             | 1        |
      | Chunk Size           | 1024     |
      | VAD Threshold        | 0.005    |
      | Turn Taking Pause    | 800ms    |
    Then the service should apply the new configuration
    And the audio capture should use the specified settings
    And the configuration should be validated before application

  @integration @speech_recognition
  Scenario: Integration with Speech Recognition Service
    Given the Audio Capture Service is running
    And the Speech Recognition Service is subscribed to audio events
    When I speak "Hello, how are you today?"
    Then the Audio Capture Service should detect and segment my speech
    And the speech audio should be published as "AUDIO_DATA_RECEIVED"
    And the Speech Recognition Service should receive the audio data
    And the complete pipeline should process the speech within 2 seconds

  @performance @resource_usage
  Scenario: Efficient resource usage during continuous operation
    Given the Audio Capture Service is running continuously
    When the service operates for an extended period (1 hour)
    Then CPU usage should remain below 10%
    And memory usage should remain stable (no memory leaks)
    And audio quality should remain consistent
    And the service should handle thousands of audio chunks efficiently

  @cleanup
  Scenario: Audio Capture Service cleanup and shutdown
    Given the Audio Capture Service is running
    And audio is being actively captured
    When I shutdown the Audio Capture Service
    Then all audio capture should stop immediately
    And audio device resources should be released properly
    And any pending audio data should be processed
    And no audio-related processes should remain running
