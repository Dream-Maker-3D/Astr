# 🎤 Astir Voice Assistant

A natural conversation voice assistant built with Python, OpenRouter AI, Whisper STT, and Coqui TTS using Gang of Four design patterns and Behavior-Driven Development (BDD) methodology.

## 🚀 **Current Status: Phase 1 - Core Foundation**

### ✅ **Completed Components:**
- **Event Bus Service** - Observer pattern implementation for decoupled communication
- **Exception Hierarchy** - Comprehensive error handling system
- **Project Structure** - Clean architecture following SOLID principles
- **Unit Tests** - Comprehensive test coverage for Event Bus Service
- **BDD Planning** - Complete feature specifications and design documents

### 🔄 **In Progress:**
- Configuration Manager (Singleton pattern)
- Audio Capture Service (Strategy pattern)

### 📋 **Next Phase:**
- Audio Processing Pipeline
- Speech Recognition (Whisper STT)
- AI Integration (OpenRouter)

## 🏗️ **Architecture Overview**

### **Design Patterns Used:**
- **Observer Pattern** - Event Bus for loose coupling
- **Strategy Pattern** - Pluggable STT, TTS, and AI services
- **Facade Pattern** - Simplified system interface
- **Factory Pattern** - Service instantiation
- **Singleton Pattern** - Configuration management

### **SOLID Principles:**
- **Single Responsibility** - Each service has one clear purpose
- **Open/Closed** - Extensible without modification
- **Liskov Substitution** - Consistent service interfaces
- **Interface Segregation** - Focused, minimal interfaces
- **Dependency Inversion** - Depend on abstractions

## 🛠️ **Technology Stack**

### **Core Technologies:**
- **Python 3.9+** - Main programming language
- **OpenRouter API** - AI conversation (Claude, GPT, Llama)
- **OpenAI Whisper** - Speech-to-text recognition
- **Coqui TTS** - Text-to-speech synthesis
- **PyAudio** - Audio input/output

### **Development Tools:**
- **pytest** - Unit testing framework
- **behave** - BDD testing framework
- **black** - Code formatting
- **mypy** - Type checking

## 🚀 **Quick Start**

### **Prerequisites:**
```bash
# Python 3.9 or higher
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### **Installation:**
```bash
# Clone the repository
git clone https://github.com/Dream-Maker-3D/Astr.git
cd Astr

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### **Configuration:**
```bash
# Required environment variables
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

### **Run the Application:**
```bash
python main.py
```

## 🧪 **Testing**

### **Run All Tests:**
```bash
# Unit tests
pytest tests/unit/ -v

# BDD tests (when implemented)
behave tests/bdd/

# Test coverage
pytest --cov=src tests/
```

### **Run Specific Tests:**
```bash
# Event Bus tests only
pytest tests/unit/core/test_event_bus.py -v

# Performance tests
pytest tests/unit/core/test_event_bus.py::TestEventBusPerformance -v
```

## 📊 **Current Implementation Status**

### **✅ Phase 1: Core Foundation (COMPLETED)**
- [x] **Event Bus Service** - Observer pattern for component communication
- [x] **Exception Hierarchy** - Structured error handling
- [x] **Project Structure** - Clean architecture setup
- [x] **Unit Tests** - Comprehensive test coverage
- [x] **Main Entry Point** - Basic application startup

### **🔄 Phase 2: Configuration & Audio (IN PROGRESS)**
- [ ] **Configuration Manager** - YAML-based configuration system
- [ ] **Audio Capture Service** - Microphone input with VAD
- [ ] **Audio Output Service** - Speaker output management

### **📋 Phase 3: Speech Processing (PLANNED)**
- [ ] **Speech Recognition Service** - Whisper STT integration
- [ ] **Speech Synthesis Service** - Coqui TTS integration
- [ ] **Natural Speech Processing** - Conversation flow management

### **🎯 Phase 4: AI Integration (PLANNED)**
- [ ] **OpenRouter Client** - AI conversation service
- [ ] **Conversation Manager** - Context and history management
- [ ] **Response Processing** - Natural conversation handling

## 🎯 **BDD Methodology**

This project follows a strict **Behavior-Driven Development** approach:

### **Development Cycle:**
1. **PLAN** - Write Gherkin scenarios and requirements
2. **DESIGN** - Create UML diagrams and API specifications  
3. **VERIFY** - Design review and test planning
4. **IMPLEMENT** - Test-first development

### **Feature Documentation:**
- **Gherkin Scenarios**: `.ai/features/*.feature`
- **UML Diagrams**: `.ai/diagrams/*.puml`
- **API Specifications**: `.ai/specifications/api/*.md`
- **Test Plans**: `.ai/validation/test_plans/*.md`

## 📁 **Project Structure**

```
Astir/
├── .ai/                          # BDD planning and design documents
│   ├── methodology/              # BDD methodology documentation
│   ├── features/                 # Gherkin feature specifications
│   ├── diagrams/                 # UML system and sequence diagrams
│   ├── specifications/           # API and technical specifications
│   └── validation/               # Test plans and validation docs
├── src/                          # Source code
│   ├── core/                     # Core services (Event Bus, Config)
│   ├── audio/                    # Audio capture and playback
│   ├── speech/                   # STT and TTS services
│   ├── ai/                       # AI conversation services
│   └── utils/                    # Utilities and exceptions
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── bdd/                      # BDD scenario tests
├── config/                       # Configuration files
├── main.py                       # Application entry point
├── requirements.txt              # Production dependencies
└── requirements-dev.txt          # Development dependencies
```

## 🔧 **Development Guidelines**

### **Code Quality:**
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maintain 90%+ test coverage
- Document all public APIs

### **BDD Process:**
1. Write Gherkin scenarios first
2. Create UML diagrams for design
3. Implement comprehensive tests
4. Write minimal code to pass tests
5. Refactor and optimize

### **Git Workflow:**
- Feature branches for new components
- Commit after each BDD phase completion
- Update README.md with progress
- Push to GitHub after each milestone

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch** following BDD methodology
3. **Write Gherkin scenarios** for new features
4. **Implement tests first** (TDD approach)
5. **Write minimal code** to pass tests
6. **Submit a pull request** with comprehensive documentation

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **OpenRouter** - AI model access and API
- **OpenAI** - Whisper speech recognition
- **Coqui** - Text-to-speech synthesis
- **Python Community** - Excellent libraries and tools

---

**Built with ❤️ using Behavior-Driven Development and Gang of Four Design Patterns**
