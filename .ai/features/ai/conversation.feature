@ai @conversation @openrouter @claude
Feature: AI Conversation Service - OpenRouter Integration
  As a voice assistant user
  I want the system to have natural AI-powered conversations
  So that I can interact with the assistant in a human-like way

  Background:
    Given the AI Conversation Service is available
    And the Event Bus Service is running
    And the Configuration Manager is initialized
    And the OpenRouter API key is configured
    And the system is ready for AI conversations

  @smoke @critical
  Scenario: AI Conversation Service initialization
    Given the AI Conversation Service is not yet initialized
    When I initialize the AI Conversation Service
    Then the service should load the OpenRouter client configuration
    And the service should validate API credentials
    And the service should configure the default AI model
    And the service should register with the Event Bus
    And the service should be ready to process conversations

  @openrouter_client
  Scenario: OpenRouter client configuration and model selection
    Given the AI Conversation Service is initializing
    And the configuration specifies AI model settings:
      | Model Provider | Model Name           | Context Window | Max Tokens | Temperature |
      | anthropic      | claude-3-5-sonnet    | 200000        | 4096       | 0.7         |
      | openai         | gpt-4-turbo          | 128000        | 4096       | 0.7         |
      | meta-llama     | llama-3.1-70b        | 128000        | 4096       | 0.7         |
    When I configure the OpenRouter client
    Then the client should be initialized with the API key
    And the default model should be set to "claude-3-5-sonnet"
    And model capabilities should be validated
    And rate limiting should be configured
    And the client should be ready for API calls

  @conversation_context
  Scenario: Conversation context management
    Given the AI Conversation Service is initialized
    And I have an ongoing conversation with context:
      | Turn | Speaker | Message                           | Timestamp |
      | 1    | User    | "What's the weather like today?"  | 10:00:00  |
      | 2    | AI      | "It's sunny and 75 degrees."     | 10:00:05  |
      | 3    | User    | "Should I wear a jacket?"         | 10:00:10  |
    When I send a follow-up message: "What about tomorrow?"
    Then the service should maintain conversation context
    And the AI should understand "tomorrow" refers to weather
    And the response should be contextually appropriate
    And conversation history should be preserved

  @natural_prompting
  Scenario: Natural conversational system prompts
    Given the AI Conversation Service uses natural conversation prompts
    When I configure the system prompt for natural conversation
    Then the prompt should encourage concise responses
    And the prompt should promote natural speech patterns
    And the prompt should discourage verbose explanations
    And the prompt should enable interruption handling
    And the prompt should maintain conversational personality

  @streaming_responses
  Scenario: Real-time streaming AI responses
    Given the AI Conversation Service supports streaming responses
    And streaming is enabled for real-time conversation
    When I send a message: "Tell me about artificial intelligence"
    Then the service should start streaming the response immediately
    And response chunks should be received within 200ms intervals
    And each chunk should be published as "AI_RESPONSE_CHUNK" events
    And the complete response should be assembled correctly
    And the final response should be published as "AI_RESPONSE_READY"

  @interruption_handling
  Scenario: Conversation interruption support
    Given the AI Conversation Service is generating a response
    And the current response is 50% complete
    When I send an interruption signal
    Then the current response generation should stop immediately
    And an "AI_RESPONSE_INTERRUPTED" event should be published
    And the conversation context should be preserved
    And the service should be ready for the next user input
    And no partial response should be sent to TTS

  @model_switching
  Scenario: Runtime AI model switching
    Given the AI Conversation Service is using "claude-3-5-sonnet"
    And I want to switch to a different model for specific tasks
    When I request model switching to "gpt-4-turbo"
    Then the service should validate the new model availability
    And the OpenRouter client should be reconfigured
    And the conversation context should be preserved
    And subsequent responses should use the new model
    And model switching should be logged for monitoring

  @error_handling
  Scenario: AI service error handling and recovery
    Given the AI Conversation Service handles various error conditions
    When AI processing encounters errors:
      | Error Type           | Scenario                    | Expected Behavior              |
      | API rate limit       | Too many requests           | Retry with exponential backoff |
      | Invalid API key      | Authentication failure      | Publish error event, fallback  |
      | Model unavailable    | Selected model offline      | Switch to backup model         |
      | Network timeout      | Connection issues           | Retry with timeout increase    |
      | Context too long     | Conversation history full   | Summarize and truncate context |
      | Invalid response     | Malformed API response      | Request regeneration           |
    Then appropriate error handling should occur
    And fallback strategies should be employed
    And error events should be published to the Event Bus
    And the system should recover gracefully

  @response_processing
  Scenario: AI response processing for TTS
    Given the AI Conversation Service receives responses from OpenRouter
    When the AI generates a response: "Well, that's an interesting question! Let me think... Actually, I believe the answer is quite straightforward."
    Then the service should process the response for natural speech
    And filler words should be preserved for naturalness
    And appropriate pauses should be marked
    And the response should be formatted for TTS synthesis
    And an "AI_RESPONSE_READY" event should be published with TTS-ready text

  @conversation_flow
  Scenario: Natural conversation flow management
    Given the AI Conversation Service manages conversation flow
    And the system is in continuous listening mode
    When I have a natural conversation:
      | User Input                          | Expected AI Behavior           |
      | "Hi there"                         | Casual greeting response       |
      | "What time is it?"                 | Provide current time           |
      | "Actually, never mind"             | Acknowledge and move on        |
      | "Can you help me with something?"  | Offer assistance naturally     |
      | "Thanks, that's all"               | Polite conversation ending     |
    Then each response should be contextually appropriate
    And conversation state should be managed properly
    And natural turn-taking should be maintained

  @performance_optimization
  Scenario: Low-latency AI response generation
    Given the AI Conversation Service is optimized for real-time conversation
    And performance targets are defined:
      | Metric                    | Target Value | Measurement Method |
      | First response chunk      | <500ms       | Time to first token |
      | Complete response         | <2000ms      | End-to-end latency |
      | Context processing        | <100ms       | Context preparation |
      | Model switching           | <200ms       | Configuration change |
    When I process continuous conversation for 10 minutes
    Then all performance targets should be met consistently
    And response latency should remain stable
    And memory usage should not grow over time
    And conversation quality should be maintained

  @rate_limiting
  Scenario: API rate limiting and usage monitoring
    Given the AI Conversation Service monitors API usage
    And rate limits are configured for OpenRouter
    When I make multiple rapid requests:
      | Request # | Interval | Expected Behavior    |
      | 1-10      | 100ms    | Normal processing    |
      | 11-20     | 50ms     | Rate limit warning   |
      | 21-30     | 10ms     | Rate limit triggered |
    Then the service should respect rate limits
    And requests should be queued when limits are reached
    And usage statistics should be tracked
    And rate limit events should be published

  @context_summarization
  Scenario: Conversation context summarization
    Given the AI Conversation Service has a long conversation history
    And the context window is approaching the limit (190k tokens)
    When I continue the conversation
    Then the service should automatically summarize older context
    And important conversation details should be preserved
    And the summary should maintain conversation continuity
    And the context window should be optimized
    And summarization should be transparent to the user

  @multi_model_support
  Scenario: Multiple AI model support and capabilities
    Given the AI Conversation Service supports multiple models
    And different models have different capabilities:
      | Model              | Strengths                    | Use Cases           |
      | claude-3-5-sonnet  | Reasoning, analysis          | Complex questions   |
      | gpt-4-turbo        | Creative, conversational     | General chat        |
      | llama-3.1-70b      | Fast, efficient              | Quick responses     |
    When I configure model selection based on conversation type
    Then the service should choose appropriate models automatically
    And model switching should be seamless
    And conversation quality should be optimized for each use case

  @event_integration
  Scenario: Event Bus integration for AI events
    Given the AI Conversation Service is integrated with the Event Bus
    When various AI operations occur:
      | Operation              | Event Type                | Event Data                    |
      | Start processing       | AI_PROCESSING_STARTED     | user_input, model, timestamp |
      | Response chunk ready   | AI_RESPONSE_CHUNK         | chunk_data, chunk_id, is_final |
      | Response complete      | AI_RESPONSE_READY         | full_text, metadata, timing   |
      | Response interrupted   | AI_RESPONSE_INTERRUPTED   | reason, partial_text, timestamp |
      | Model switched         | AI_MODEL_CHANGED          | old_model, new_model, reason   |
      | Error occurred         | AI_ERROR                  | error_type, message, recovery  |
      | Context summarized     | AI_CONTEXT_SUMMARIZED     | summary, tokens_saved, timestamp |
    Then appropriate events should be published to the Event Bus
    And event data should include all relevant metadata
    And other services should be able to subscribe to AI events
    And event timing should be accurate and consistent

  @configuration_integration
  Scenario: Configuration system integration
    Given the AI Conversation Service uses the Configuration Manager
    And AI conversation settings are configured:
      | Setting              | Value                    | Description                    |
      | default_model        | claude-3-5-sonnet        | Primary AI model to use        |
      | max_tokens           | 4096                     | Maximum response length        |
      | temperature          | 0.7                      | Response creativity level      |
      | context_window       | 200000                   | Maximum context tokens         |
      | streaming_enabled    | true                     | Enable response streaming      |
      | retry_attempts       | 3                        | API retry count               |
      | timeout_seconds      | 30                       | Request timeout               |
    When the service initializes and processes conversations
    Then it should use the configured settings
    And settings should be validated for correctness
    And invalid settings should use safe defaults
    And configuration changes should be applied dynamically

  @conversation_memory
  Scenario: Conversation memory and persistence
    Given the AI Conversation Service maintains conversation memory
    And conversation sessions can span multiple interactions
    When I have conversations across different time periods:
      | Session | Time      | Topic                    | Context Needed        |
      | 1       | Morning   | "Plan my day"           | Calendar, preferences |
      | 2       | Afternoon | "How's my day going?"   | Reference to morning  |
      | 3       | Evening   | "Summarize today"       | Full day context      |
    Then the service should maintain session continuity
    And relevant context should be recalled appropriately
    And conversation memory should be managed efficiently
    And privacy and data retention should be respected

  @natural_speech_patterns
  Scenario: Natural speech pattern generation
    Given the AI Conversation Service generates natural speech patterns
    When I ask various types of questions:
      | Question Type        | Example Input                | Expected Speech Pattern       |
      | Simple factual       | "What's 2+2?"               | Brief, direct answer          |
      | Complex explanation  | "How does photosynthesis work?" | Structured, clear explanation |
      | Casual conversation  | "How are you today?"        | Natural, conversational tone  |
      | Emotional support    | "I'm feeling stressed"      | Empathetic, supportive response |
    Then responses should match natural human speech patterns
    And tone should be appropriate for the context
    And response length should match the question complexity
    And conversational markers should be included naturally

  @integration_with_speech_services
  Scenario: Integration with Speech Recognition and Synthesis
    Given the AI Conversation Service is integrated with speech services
    And the complete pipeline is: STT → AI → TTS
    When I speak: "What's the capital of France?"
    Then the Speech Recognition Service should transcribe my speech
    And the transcription should be sent to AI processing
    And the AI should generate an appropriate response
    And the response should be sent to Speech Synthesis
    And I should hear the synthesized answer
    And the entire pipeline should complete within 3 seconds
