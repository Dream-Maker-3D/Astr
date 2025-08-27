# Coqui STT/TTS Implementation Requirements

## Executive Summary
Based on research conducted in 2024, **Coqui STT is deprecated** and no longer actively maintained. The recommendation is to use **OpenAI Whisper for STT** and **Coqui TTS for voice synthesis**. Coqui TTS remains actively maintained with excellent voice cloning capabilities.

## Coqui STT Status (⚠️ DEPRECATED)
- **Status**: No longer actively maintained as of December 2023
- **Last Update**: Community maintains models but no active development
- **Python Support**: 3.6, 3.7, 3.8, 3.9 only
- **Recommendation**: Use OpenAI Whisper instead for better accuracy and active maintenance

### Legacy Installation (if still needed):
```bash
python -m pip install coqui-stt-model-manager
```

## Coqui TTS (✅ RECOMMENDED)

### System Requirements
- **Python Version**: >= 3.9, < 3.12 (Python 3.12 not yet supported)
- **OS**: Tested on Ubuntu 18.04+ (works on Windows/Mac with prebuilt wheels)
- **Hardware**: GPU highly recommended for advanced models and voice cloning
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for voice cloning

### Installation Options

#### Option 1: PyPI Installation (Recommended)
```bash
# Install the maintained fork
pip install coqui-tts

# Or install specific version
pip install coqui-tts==0.22.0
```

#### Option 2: Development Installation
```bash
# Clone repository
git clone https://github.com/coqui-ai/TTS
cd TTS

# Install with development dependencies
pip install -e .[all,dev,notebooks]

# Ubuntu/Debian system dependencies
make system-deps
make install
```

#### Option 3: Docker Installation
```bash
# Run without installation
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu

# List available models
python3 TTS/server/server.py --list_models

# Start server with specific model
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
```

### Best Models for Voice Assistant (2024)

#### 1. XTTS-v2 (Highest Quality Voice Cloning)
- **Best for**: Real-time voice cloning, multilingual support
- **Languages**: 17 languages supported
- **Latency**: <200ms streaming capability
- **Voice Clone Requirements**: 6-second audio clip minimum
- **Model ID**: `tts_models/multilingual/multi-dataset/xtts_v2`

#### 2. VITS Models
- **Best for**: High-quality single-language synthesis
- **Languages**: Various individual language models
- **Model ID**: `tts_models/en/ljspeech/vits`

#### 3. YourTTS
- **Best for**: Multi-speaker scenarios
- **Features**: Voice conversion, speaker adaptation

### Python Implementation Example
```python
from TTS.api import TTS

# Initialize XTTS-v2 for voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Generate speech with voice cloning
tts.tts_to_file(
    text="Hello, this is a voice clone demonstration.",
    file_path="output.wav",
    speaker_wav="/path/to/reference_voice.wav",
    language="en"
)

# For real-time streaming
import sounddevice as sd
import numpy as np

def stream_tts(text, reference_voice_path):
    wav = tts.tts(text=text, speaker_wav=reference_voice_path, language="en")
    sd.play(wav, samplerate=22050)
    sd.wait()
```

### Voice Cloning Quality Guidelines

#### Training Data Requirements:
- **Minimum**: 6-second audio clip for basic cloning
- **Good Quality**: 1-2 minutes of varied speech
- **Excellent Quality**: 20+ minutes in 3-10 second clips
- **Professional Quality**: 100+ voice samples across different contexts

#### Audio Quality Requirements:
- **Sample Rate**: 22050 Hz (optimal for most models)
- **Format**: WAV, 16-bit recommended
- **Environment**: Low background noise
- **Content**: Natural speech patterns, varied intonation

### Dependencies and System Setup

#### Python Dependencies:
```bash
# Core dependencies (automatically installed)
pip install torch torchaudio
pip install numpy scipy
pip install librosa soundfile
pip install pyyaml
pip install flask  # for server mode
```

#### System Dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev
```

#### Audio System Dependencies:
```bash
# For audio playback
sudo apt-get install -y pulseaudio alsa-utils

# For microphone input
sudo apt-get install -y libasound2-dev
```

## Alternative STT Recommendation: OpenAI Whisper

Since Coqui STT is deprecated, we recommend using OpenAI Whisper:

```bash
# Install Whisper
pip install openai-whisper

# Or faster-whisper for better performance
pip install faster-whisper
```

### Whisper Implementation Example:
```python
import whisper
import numpy as np

# Load model
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe("audio_file.wav")
print(result["text"])
```

## Integration Architecture Recommendations

### Recommended Stack:
1. **STT**: OpenAI Whisper (faster-whisper for production)
2. **TTS**: Coqui TTS with XTTS-v2
3. **Audio I/O**: PyAudio or SoundDevice
4. **AI**: Claude API (Anthropic)

### Performance Considerations:
- **GPU**: Essential for real-time voice cloning
- **CPU**: Sufficient for basic TTS models
- **Memory**: 8GB+ RAM recommended for voice cloning
- **Storage**: Models require 1-4GB disk space each

## Migration Path from JavaScript
The previous JavaScript implementation failed primarily due to:
1. Whisper.js limitations in Node.js environment
2. Complex audio processing pipeline
3. Browser-based transformers.js compatibility issues

The Python ecosystem provides:
- Mature audio processing libraries
- Native Whisper implementation
- Stable Coqui TTS integration
- Better hardware utilization

This research forms the foundation for implementing a robust Python-based voice assistant.