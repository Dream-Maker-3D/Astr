Feature: Real-time Voice Conversation
  As a user
  I want to have natural voice conversations with an AI assistant
  So that I can interact seamlessly without wake words or interruptions

  Background:
    Given the voice assistant is running
    And the microphone is connected and working
    And the speakers are connected and working
    And the LLM service is available

  Scenario: Basic voice interaction
    Given the assistant is in listening mode
    When I say "Hello, how are you today?"
    Then the assistant should transcribe my speech in real-time
    And the assistant should generate a conversational response
    And the assistant should speak the response back to me
    And the response should be completed within 2 seconds

  Scenario: Continuous conversation flow
    Given the assistant is in listening mode
    When I say "What's the weather like?"
    And the assistant responds with weather information
    And I immediately say "What about tomorrow?"
    Then the assistant should understand the context
    And provide tomorrow's weather information
    And maintain conversation continuity

  Scenario: Real-time transcription streaming
    Given the assistant is listening
    When I start speaking "The quick brown fox jumps over the lazy dog"
    Then the transcription should appear in real-time chunks
    And each chunk should be processed as it arrives
    And the complete sentence should be accurate

  Scenario: Interrupt handling
    Given the assistant is speaking a response
    When I start speaking before the response is complete
    Then the assistant should stop speaking immediately
    And start listening to my new input
    And process my interruption as a new conversation turn

  Scenario: Background noise filtering
    Given there is moderate background noise
    When I speak to the assistant
    Then the voice activity detector should distinguish my voice from noise
    And the transcription should be accurate despite the noise
    And the assistant should respond appropriately

  Scenario: Silence handling
    Given the assistant is listening
    When there is silence for 3 seconds
    Then the assistant should remain in listening mode
    And not generate unnecessary responses
    And continue waiting for user input

  Scenario: Long response streaming
    Given I ask a complex question requiring a long response
    When the assistant generates a lengthy answer
    Then the response should stream in real-time chunks
    And I should hear the response as it's being generated
    And the audio should be continuous without gaps

  Scenario: Error recovery
    Given the assistant encounters a network error during LLM processing
    When the error occurs
    Then the assistant should provide a graceful error message
    And attempt to reconnect automatically
    And resume normal operation when connection is restored

  Scenario: Multi-turn context preservation
    Given we are having a conversation about cooking
    When I ask "How do I make pasta?"
    And the assistant explains pasta making
    And I ask "What about the sauce?"
    Then the assistant should remember we're discussing pasta
    And provide sauce recommendations for pasta
    And maintain the cooking context throughout

  Scenario: Voice activity detection accuracy
    Given the assistant is in listening mode
    When I clear my throat or make non-speech sounds
    Then the voice activity detector should not trigger transcription
    And the assistant should continue listening for actual speech
    And only process intentional spoken words