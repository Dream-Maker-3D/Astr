# Architecture Overview

## Design Philosophy

This voice AI assistant is built using **SOLID principles**, **GoF design patterns**, and **single responsibility programming** to create a maintainable, extensible, and robust system.

## Core Design Patterns Used

### 1. Observer Pattern (Event Bus)
- **Implementation**: `EventBus.js` with EventEmitter3
- **Purpose**: Loose coupling between components
- **Benefits**: Components communicate through events, making the system highly modular

### 2. Strategy Pattern (Service Providers)
- **Implementation**: Pluggable STT, TTS, and LLM services
- **Purpose**: Easy switching between different AI providers
- **Benefits**: Can swap OpenRouter models (Claude, GPT, Llama) without code changes

### 3. State Pattern (State Management)
- **Implementation**: `StateManager.js` with defined state transitions
- **Purpose**: Manage complex conversation states
- **Benefits**: Clear state flow, prevents invalid transitions

### 4. Factory Pattern (Service Initialization)
- **Implementation**: `VoiceAssistant.js` orchestrates service creation
- **Purpose**: Centralized service instantiation and configuration
- **Benefits**: Dependency injection, easier testing

### 5. Command Pattern (Event Handling)
- **Implementation**: Event handlers as discrete commands
- **Purpose**: Encapsulate actions as objects
- **Benefits**: Easy to extend, undo/redo capabilities

## SOLID Principles Application

### Single Responsibility Principle (SRP)
- `AudioCapture`: Only handles microphone input
- `SpeechToTextService`: Only converts speech to text
- `LLMService`: Only handles AI conversation
- `ConversationManager`: Only manages conversation flow
- Each class has one reason to change

### Open/Closed Principle (OCP)
- Services are open for extension (new providers) but closed for modification
- New STT/TTS/LLM providers can be added without changing existing code
- Event system allows new features without modifying core components

### Liskov Substitution Principle (LSP)
- Service interfaces are consistent across providers
- Any STT service can replace another STT service
- Configuration drives behavior, not hardcoded logic

### Interface Segregation Principle (ISP)
- Small, focused interfaces for each service type
- Components only depend on methods they actually use
- No forced dependencies on unused functionality

### Dependency Inversion Principle (DIP)
- High-level components depend on abstractions, not concrete implementations
- `VoiceAssistant` depends on service interfaces, not specific providers
- Configuration injection enables different deployment scenarios

## Component Architecture

```
┌─────────────────────┐
│   VoiceAssistant    │  ← Main Orchestrator
│   (Dependency       │
│    Injection)       │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│     EventBus        │  ← Observer Pattern
│   (Event Emitter)   │
└─────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌────────────┐
│ Audio  │  │Conversation│  ← Strategy Pattern
│Services│  │  Manager   │
└────────┘  └────────────┘
    │             │
    ▼             ▼
┌────────┐  ┌────────────┐
│Speech  │  │    LLM     │  ← Service Abstractions
│Services│  │  Service   │
└────────┘  └────────────┘
```

## Event-Driven Flow

1. **Audio Input**: Microphone → AudioCapture → AUDIO_DATA_RECEIVED event
2. **Voice Detection**: VoiceActivityDetector → SPEECH_DETECTED event
3. **Transcription**: SpeechToTextService → TRANSCRIPTION_COMPLETE event
4. **AI Processing**: OpenRouterService → AI_RESPONSE_CHUNK events
5. **Speech Synthesis**: TextToSpeechService → TTS_AUDIO_CHUNK events
6. **Audio Output**: AudioPlayer → PLAYBACK_FINISHED event

## State Machine

```
INITIALIZING → READY → LISTENING → PROCESSING_SPEECH
     ↓          ↑           ↓              ↓
   ERROR    PAUSED    GENERATING_RESPONSE   ↓
     ↑          ↓           ↓              ↓
  SHUTDOWN ← READY ←    SPEAKING ←─────────┘
```

## Extensibility Points

### New AI Providers
1. Implement service interface (e.g., new STT provider)
2. Add configuration options
3. Register with factory/orchestrator
4. Zero changes to existing code

### New Features
1. Create new service class
2. Subscribe to relevant events
3. Publish new events if needed
4. Update configuration schema

### New Event Types
1. Add to `EVENT_TYPES` enum
2. Document event payload structure
3. Components can subscribe/publish immediately

## Configuration Strategy

- **Hierarchical Configuration**: Default → Environment → User overrides
- **Environment-Specific**: Development/Production/Test configs
- **Runtime Modification**: Services can be reconfigured without restart
- **Validation**: Configuration schema validation prevents invalid states

## Error Handling Strategy

- **Graceful Degradation**: System continues working with reduced functionality
- **Circuit Breaker**: Failed services don't crash entire system
- **Recovery**: Automatic retry with exponential backoff
- **Logging**: Comprehensive error context for debugging

## Testing Strategy

- **Unit Tests**: Each component tested in isolation
- **Integration Tests**: Event flow and service interaction
- **Behavior Tests**: Gherkin scenarios for user interactions
- **Mock Services**: Test without external API dependencies

## Performance Considerations

- **Streaming**: Real-time processing, not batch
- **Buffering**: Smart buffering to minimize latency
- **Resource Management**: Cleanup and memory management
- **Concurrency**: Async/await throughout, non-blocking operations

## Security Principles

- **API Key Management**: Environment variables, never hardcoded
- **Input Validation**: All external inputs validated
- **Error Information**: No sensitive data in error messages
- **Audio Privacy**: Temporary files cleaned up immediately

## Deployment Architecture

```
Development:
┌─────────────┐
│ Local Dev   │
│ Environment │
└─────────────┘

Production (Future):
┌─────────────┐    ┌─────────────┐
│   Docker    │    │   Cloud     │
│ Container   │ or │  Service    │
└─────────────┘    └─────────────┘
```

This architecture ensures the system is maintainable, testable, and can evolve to meet future requirements while preserving the core design principles.