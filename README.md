# Dudoxx Whisper STT

**Ultra-low-latency, self-hosted speech-to-text for healthcare applications**

[![Based on WhisperLiveKit](https://img.shields.io/badge/Based%20on-WhisperLiveKit-blue)](https://github.com/QuentinFuxa/WhisperLiveKit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

Dudoxx Whisper STT is a real-time speech-to-text service optimized for healthcare applications. Based on the excellent [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) project, it provides:

- **Ultra-low latency** transcription using state-of-the-art SimulStreaming
- **Speaker diarization** for multi-speaker scenarios
- **Multi-language support** with translation capabilities
- **Apple Silicon optimization** via MLX-Whisper
- **HIPAA-compliant** self-hosted deployment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Dudoxx/dudoxx-whisper-stt.git
cd dudoxx-whisper-stt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# For macOS Apple Silicon optimization
pip install mlx-whisper
```

### Running the Server

```bash
# Start on default port 8000
dudoxx-stt --model base --language en

# Start on custom port (e.g., 4300 for Dudoxx integration)
dudoxx-stt --model base --language en --port 4300

# With speaker diarization
dudoxx-stt --model medium --language en --port 4300 --diarization
```

### Access the Demo

Open your browser and navigate to:
- **Default:** http://localhost:8000
- **Dudoxx port:** http://localhost:4300

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `small` |
| `--language` | Source language (`en`, `de`, `fr`, `auto`) | `auto` |
| `--port` | Server port | `8000` |
| `--host` | Server host | `localhost` |
| `--diarization` | Enable speaker identification | `False` |
| `--backend` | Whisper backend (`auto`, `mlx-whisper`, `faster-whisper`) | `auto` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dudoxx Whisper STT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Browser/Client                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Audio Capture (MediaRecorder/AudioWorklet)             │   │
│   │              ↓ WebSocket                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│   FastAPI Server (Port 4300)                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  WebSocket Handler → Audio Processor                    │   │
│   │                          ↓                              │   │
│   │  Voice Activity Detection (Silero VAD)                  │   │
│   │                          ↓                              │   │
│   │  Transcription Engine                                   │   │
│   │  ├── SimulStreaming (AlignAtt policy)                  │   │
│   │  ├── MLX-Whisper (Apple Silicon)                       │   │
│   │  └── Faster-Whisper (GPU/CPU)                          │   │
│   │                          ↓                              │   │
│   │  Speaker Diarization (Optional)                         │   │
│   │  ├── Streaming Sortformer (SOTA 2025)                  │   │
│   │  └── Diart                                             │   │
│   │                          ↓                              │   │
│   │  JSON Response → WebSocket                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Dudoxx Clinic Platform

This service integrates with the Dudoxx FHIR Clinic Platform for:

- **Medical transcription** during patient consultations
- **Real-time captioning** for video calls (Jitsi)
- **Clinical documentation** automation
- **Voice-to-FHIR** data extraction (future)

### Port Allocation

| Service | Port |
|---------|------|
| Next.js Frontend | 4000 |
| NestJS Backend | 4100 |
| Calendar Microservice | 4200 |
| **Dudoxx Whisper STT** | **4300** |
| HAPI FHIR Server | 8080 |
| Jitsi Meet | 8543 |

## Development

### Project Structure

```
dudoxx-whisper-stt/
├── whisperlivekit/           # Core library
│   ├── basic_server.py       # FastAPI server
│   ├── audio_processor.py    # Audio processing
│   ├── transcription.py      # Transcription engine
│   ├── diarization/          # Speaker identification
│   ├── simul_whisper/        # SimulStreaming implementation
│   ├── whisper/              # Whisper model handling
│   └── web/                  # Web demo UI
├── chrome-extension/         # Browser extension
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

### Running in Development

```bash
# Install in editable mode
pip install -e .

# Run with auto-reload (development)
uvicorn whisperlivekit.basic_server:app --reload --port 4300
```

## Docker Deployment

```bash
# Build image (CPU)
docker build -f Dockerfile.cpu -t dudoxx-whisper-stt .

# Run container
docker run -p 4300:8000 --name dudoxx-stt dudoxx-whisper-stt --model base --language en

# With GPU support
docker build -t dudoxx-whisper-stt .
docker run --gpus all -p 4300:8000 --name dudoxx-stt dudoxx-whisper-stt
```

## Credits

This project is based on [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) by Quentin Fuxa, which implements state-of-the-art research:

- **SimulStreaming** (SOTA 2025) - Ultra-low latency transcription
- **WhisperStreaming** (SOTA 2023) - LocalAgreement policy
- **Streaming Sortformer** (SOTA 2025) - Real-time speaker diarization
- **Silero VAD** (2024) - Voice Activity Detection

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Original WhisperLiveKit is also licensed under Apache 2.0.

## Support

- **Dudoxx Support:** support@dudoxx.com
- **Original Project:** [WhisperLiveKit Issues](https://github.com/QuentinFuxa/WhisperLiveKit/issues)

---

**Dudoxx UG** - Healthcare AI Solutions  
**Acceleate Consulting** - Enterprise Technology
