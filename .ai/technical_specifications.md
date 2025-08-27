# Technical Specifications - Voice Assistant System

## System Overview
A real-time voice assistant built with Python, integrating OpenAI Whisper (STT), Coqui TTS, and Claude AI using Gang of Four design patterns and behavior-driven development principles.

## Architecture Summary

### Core Components
1. **VoiceAssistantFacade** - Main system coordinator (Facade Pattern)
2. **EventBusService** - Event-driven communication hub (Observer Pattern)
3. **AudioCaptureService** - Microphone input management
4. **SpeechRecognitionService** - Speech-to-text using Whisper (Strategy Pattern)
5. **AIConversationService** - Claude API integration (Command Pattern)
6. **SpeechSynthesisService** - Text-to-speech using Coqui TTS (Strategy Pattern)
7. **AudioPlayerService** - Speaker output management
8. **ConfigurationManager** - System configuration (Singleton Pattern)

## Technology Stack

### Core Technologies
- **Language**: Python 3.9+
- **STT**: OpenAI Whisper (faster-whisper for production)
- **TTS**: Coqui TTS with XTTS-v2 model
- **AI**: Claude API (Anthropic)
- **Audio I/O**: PyAudio + SoundDevice
- **Event System**: Custom Observer pattern implementation

### Dependencies
```python
# Core audio processing
pyaudio>=0.2.11
sounddevice>=0.4.6
soundfile>=0.12.1
librosa>=0.10.1

# Speech recognition
openai-whisper>=20231117
faster-whisper>=0.10.0  # Production alternative

# Text-to-speech
coqui-tts>=0.22.0

# AI integration
anthropic>=0.18.0

# Utilities
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
```

### System Requirements
- **Python**: 3.9 to 3.11 (3.12 not supported by Coqui TTS)
- **RAM**: Minimum 8GB, 16GB recommended for voice cloning
- **GPU**: Optional but highly recommended for Coqui TTS performance
- **Storage**: 4-6GB for models (Whisper + Coqui TTS)
- **Network**: Required for Claude API calls

## Project Structure
```
Astir/
├── .ai/                          # Planning documentation
│   ├── system_architecture.uml   # UML diagrams
│   ├── sequence_diagram.uml      # Sequence flow
│   ├── voice_assistant.feature   # BDD scenarios
│   ├── UNDONE.md                 # Development roadmap
│   ├── coqui_requirements.md     # Technology research
│   ├── design_patterns.md        # GoF patterns documentation
│   └── technical_specifications.md # This file
├── src/                          # Source code
│   ├── core/                     # Core system components
│   │   ├── __init__.py
│   │   ├── facade.py             # VoiceAssistantFacade
│   │   ├── event_bus.py          # EventBusService
│   │   └── config_manager.py     # ConfigurationManager
│   ├── audio/                    # Audio processing
│   │   ├── __init__.py
│   │   ├── capture_service.py    # AudioCaptureService
│   │   ├── player_service.py     # AudioPlayerService
│   │   └── preprocessing.py      # AudioPreprocessor
│   ├── speech/                   # Speech processing
│   │   ├── __init__.py
│   │   ├── recognition_service.py # SpeechRecognitionService
│   │   ├── synthesis_service.py  # SpeechSynthesisService
│   │   └── strategies/           # Strategy implementations
│   │       ├── __init__.py
│   │       ├── whisper_stt.py    # Whisper STT strategy
│   │       └── coqui_tts.py      # Coqui TTS strategy
│   ├── ai/                       # AI integration
│   │   ├── __init__.py
│   │   ├── conversation_service.py # AIConversationService
│   │   └── claude_client.py      # Claude API wrapper
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── exceptions.py         # Custom exceptions
│       ├── logging_config.py     # Logging setup
│       └── validators.py         # Input validation
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── bdd/                      # BDD scenario tests
├── config/                       # Configuration files
│   ├── default.yaml              # Default configuration
│   └── production.yaml           # Production settings
├── models/                       # Model files (gitignored)
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup
├── .env.example                  # Environment template
└── main.py                       # Application entry point
```

## API Design

### VoiceAssistantFacade Interface
```python
class VoiceAssistantFacade:
    def __init__(self, config_path: str = None)
    async def initialize(self) -> bool
    async def start_conversation(self) -> None
    async def stop_conversation(self) -> None
    async def handle_voice_input(self) -> None
    def get_status(self) -> Dict[str, Any]
    async def shutdown(self) -> None
```

### EventBusService Interface
```python
class EventBusService:
    def subscribe(self, event_type: str, handler: Callable) -> None
    def unsubscribe(self, event_type: str, handler: Callable) -> None
    def publish(self, event_type: str, data: Dict = None) -> None
    def publish_async(self, event_type: str, data: Dict = None) -> Awaitable
```

### Event Types
```python
class EventTypes:
    # Audio events
    AUDIO_DATA_RECEIVED = "audio.data.received"
    AUDIO_PLAYBACK_START = "audio.playback.start"
    AUDIO_PLAYBACK_COMPLETE = "audio.playback.complete"
    
    # Speech events
    SPEECH_DETECTED = "speech.detected"
    SPEECH_RECOGNIZED = "speech.recognized"
    SPEECH_SYNTHESIS_START = "speech.synthesis.start"
    SPEECH_SYNTHESIS_COMPLETE = "speech.synthesis.complete"
    
    # AI events
    AI_REQUEST_SENT = "ai.request.sent"
    AI_RESPONSE_RECEIVED = "ai.response.received"
    
    # System events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_ENDED = "conversation.ended"
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS_CHANGED = "system.status.changed"
```

## Configuration Schema
```yaml
# config/default.yaml
audio:
  input:
    device_id: null  # Auto-detect
    sample_rate: 16000
    channels: 1
    chunk_size: 1024
    buffer_size: 4096
    continuous_listening: true     # Always listening, no activation
  output:
    device_id: null  # Auto-detect
    sample_rate: 22050
    volume: 0.8

speech:
  recognition:
    provider: "whisper"  # whisper | faster-whisper
    model: "base"       # tiny, base, small, medium, large
    language: "en"
    confidence_threshold: 0.6    # Lower for natural speech
    continuous_mode: true        # Continuous VAD processing
  synthesis:
    provider: "coqui"   # coqui | edge
    model: "tts_models/multilingual/multi-dataset/xtts_v2"
    voice: null         # Path to reference voice file
    speaking_rate: 1.1  # Slightly faster for natural conversation
    pitch: 0.0
    naturalness: "high" # Focus on natural-sounding speech

ai:
  provider: "claude"
  api_key_env: "ANTHROPIC_API_KEY"
  model: "claude-3-5-haiku-20241022"
  max_tokens: 150               # Shorter responses for conversation
  temperature: 0.8              # More natural variability
  timeout: 20                   # Faster timeout for conversation
  conversation_mode: true       # Enable conversational prompting
  system_prompt: |
    You are having a natural spoken conversation. Be concise and 
    conversational. Respond like a knowledgeable person, not an AI assistant.
    Use natural speech patterns, contractions, and keep responses brief 
    unless asked for details. Handle interruptions and corrections gracefully 
    without acknowledging them explicitly.

conversation:
  mode: "natural"               # Natural conversation vs command mode
  activation_method: "none"     # No wake words or activation
  max_context_length: 15        # More context for natural flow
  context_window_seconds: 600   # Longer memory for conversation
  interrupt_detection: true     # Immediate interruption handling
  interruption_response_ms: 50  # Very fast interruption response
  voice_activity_threshold: 0.015  # Sensitive VAD
  turn_taking_pause_ms: 1500    # Natural pause detection
  response_style: "concise"     # Brief, conversational responses
  formality: "casual"           # Natural speech patterns

system:
  log_level: "INFO"
  log_file: "voice_assistant.log"
  performance_monitoring: true
  health_check_interval: 60
  silent_startup: true          # No startup announcements
```

## Performance Requirements

### Latency Targets
- **Audio Capture to STT**: < 100ms
- **STT Processing**: < 500ms (Whisper base model)
- **AI Response**: < 2000ms (network dependent)
- **TTS Processing**: < 300ms
- **Total End-to-End**: < 3000ms

### Quality Metrics
- **STT Accuracy**: > 95% for clear English speech
- **TTS Naturalness**: MOS score > 4.0
- **System Uptime**: > 99.5%
- **Memory Usage**: < 2GB during operation
- **CPU Usage**: < 50% on average

### Scalability
- **Concurrent Conversations**: 1 (single-user system)
- **Session Duration**: Unlimited (with memory management)
- **Model Loading**: Lazy loading, < 30s startup
- **Configuration Changes**: Hot-reload without restart

## Error Handling Strategy

### Error Categories
1. **Audio Errors**: Device not found, permission denied, buffer overflow
2. **Network Errors**: API timeout, connection lost, rate limiting
3. **Model Errors**: Model loading failed, inference timeout, out of memory
4. **Configuration Errors**: Invalid config, missing files, permission issues

### Recovery Mechanisms
- **Graceful Degradation**: Fallback to simpler models if advanced ones fail
- **Retry Logic**: Exponential backoff for network requests
- **State Recovery**: Save conversation state for crash recovery
- **Health Monitoring**: Automatic restart of failed components

## Security Considerations

### Data Privacy
- **Audio Data**: Never stored permanently, processed in memory only
- **Conversation History**: Local storage only, user-controlled retention
- **API Keys**: Environment variables, never logged or committed
- **Network Traffic**: HTTPS only for all external calls

### Access Control
- **File Permissions**: Restrict model and config file access
- **Network Permissions**: Minimal required network access
- **Process Isolation**: Run with minimal system privileges

## Testing Strategy

### Unit Tests (>90% coverage)
- Individual service classes
- Strategy pattern implementations
- Event bus functionality
- Configuration management

### Integration Tests
- End-to-end conversation flow
- Error handling scenarios
- Performance under load
- Hardware compatibility

### BDD Tests
- All Gherkin scenarios implemented
- User acceptance criteria validation
- Real-world usage patterns

### Performance Tests
- Memory leak detection
- CPU usage monitoring
- Response time measurement
- Stress testing with extended conversations

## Deployment Requirements

### Development Environment
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
python main.py --config config/default.yaml --debug
```

### Production Environment
- **OS**: Ubuntu 20.04+ or compatible Linux distribution
- **Python**: Managed via virtual environment
- **Models**: Pre-downloaded and cached locally
- **Configuration**: Production YAML with optimized settings
- **Monitoring**: Integrated logging and health checks

This technical specification provides the foundation for implementing a robust, well-architected voice assistant system following best practices and design patterns.