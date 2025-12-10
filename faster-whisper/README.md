# Faster-Whisper Streaming ASR

A lightweight, production-ready streaming ASR server using faster-whisper with:
- **Multi-language support**: EN, FR, DE + 96 more languages
- **Low GPU memory**: ~4-5GB total (vs 18GB for Voxtral)
- **MIT License**: Commercial use allowed

## Features

- **Real-time Streaming**: WebSocket-based streaming transcription
- **Voice Activity Detection**: Silero VAD (ONNX) for efficient audio processing
- **Speaker Diarization**: Pyannote 3.1 for speaker identification
- **Built-in Punctuation**: Automatic punctuation and capitalization
- **Word Timestamps**: Precise word-level timing
- **INT8 Quantization**: 3GB VRAM instead of 10GB

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client (Browser/App)                        │
│                  WebSocket: wss://host/asr                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │ PCM Audio (Int16, 16kHz)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│           Faster-Whisper Streaming Server (Port 4400)           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Silero   │  │ faster-      │  │ Pyannote (optional)        │ │
│  │ VAD      │  │ whisper      │  │ Diarization                │ │
│  │ ~100MB   │  │ large-v3     │  │ ~1.2GB                     │ │
│  │ ONNX     │  │ INT8: ~3GB   │  │                            │ │
│  └──────────┘  └──────────────┘  └────────────────────────────┘ │
│                                                                 │
│  Total GPU Memory: ~4-5GB                                       │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Memory Comparison

| Solution | GPU Memory | Relative |
|----------|------------|----------|
| Voxtral Mini 3B | ~18-20GB | 100% |
| **faster-whisper INT8** | **~4-5GB** | **25%** |
| faster-whisper FP16 | ~7-8GB | 40% |

## Installation

### 1. Create Installation Directory

```bash
sudo mkdir -p /opt/faster-whisper
sudo chown $USER:$USER /opt/faster-whisper
```

### 2. Copy Files

```bash
cp streaming_server.py /opt/faster-whisper/
cp config/.env.example /opt/faster-whisper/.env
```

### 3. Create Virtual Environment

```bash
cd /opt/faster-whisper
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install faster-whisper (includes CTranslate2)
pip install faster-whisper

# Install other dependencies
pip install fastapi uvicorn python-multipart silero-vad onnxruntime-gpu

# Optional: For speaker diarization
pip install pyannote.audio
```

### 4. Configure Environment

Edit `/opt/faster-whisper/.env`:

```bash
# Model settings
WHISPER_MODEL=large-v3
WHISPER_COMPUTE_TYPE=int8  # Use int8 for ~3GB VRAM

# Diarization (optional)
ENABLE_DIARIZATION=true
HF_TOKEN=your_huggingface_token
```

### 5. Run Server

```bash
# Development
cd /opt/faster-whisper
source venv/bin/activate
source .env
python streaming_server.py --port 4400

# Production (systemd)
sudo cp systemd/faster-whisper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable faster-whisper
sudo systemctl start faster-whisper
```

### 6. Setup Apache2 Proxy

```bash
# Copy configs
sudo cp ../apache2/canary.example.com*.conf /etc/apache2/sites-available/

# Replace domain
sudo sed -i 's/canary.example.com/canary.yourdomain.com/g' /etc/apache2/sites-available/canary.*.conf

# Enable modules and sites
sudo a2enmod ssl proxy proxy_http proxy_wstunnel rewrite headers
sudo certbot certonly --apache -d canary.yourdomain.com
sudo a2ensite canary.yourdomain.com.conf canary.yourdomain.com-ssl.conf
sudo systemctl reload apache2
```

## Usage

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://canary.yourdomain.com/asr?language=auto');

// Send PCM audio (Int16, 16kHz, mono)
ws.send(audioBuffer);

// Receive transcription
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);           // Transcribed text
  console.log(data.language);       // Detected language
  console.log(data.diarization);    // Speaker info
};
```

### HTTP API

```bash
curl -X POST "http://localhost:4400/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=auto"
```

### Response Format

```json
{
  "type": "partial",
  "text": "Hello, my name is Dr. Schmidt from Berlin.",
  "full_transcript": "Hello, my name is Dr. Schmidt from Berlin.",
  "timing": {"start": 0.0, "end": 2.5, "duration": 2.5},
  "language": {
    "detected": "en",
    "probability": 0.98,
    "requested": "auto"
  },
  "diarization": {
    "current_speaker": "Speaker 1",
    "speakers": [{"id": "SPEAKER_00", "label": "Speaker 1"}]
  },
  "vad": {
    "is_speech": true,
    "speech_ratio": 0.85
  },
  "processing": {
    "latency_ms": 180,
    "rtf": 0.12
  }
}
```

## Supported Languages

Primary targets: **English (en)**, **German (de)**, **French (fr)**

Also supports: Spanish, Italian, Portuguese, Dutch, Arabic, Chinese, Japanese, Korean, and 90+ more languages.

## Model Options

| Model | Size | VRAM (INT8) | VRAM (FP16) | Speed | Accuracy |
|-------|------|-------------|-------------|-------|----------|
| tiny | 39M | ~0.5GB | ~1GB | Fastest | Lower |
| base | 74M | ~0.7GB | ~1.5GB | Fast | Good |
| small | 244M | ~1GB | ~2GB | Medium | Better |
| medium | 769M | ~2GB | ~4GB | Slower | High |
| large-v3 | 1.5B | ~3GB | ~6GB | Slowest | Best |

## Performance Tuning

### Reduce Memory Further

```bash
# Use medium model instead of large-v3
WHISPER_MODEL=medium  # ~2GB instead of 3GB

# Disable diarization
ENABLE_DIARIZATION=false  # Save ~1.2GB
```

### Improve Speed

```bash
# Use smaller model
WHISPER_MODEL=small  # 3x faster than large-v3

# Reduce beam size in code
beam_size=1  # Faster but less accurate
```

## Files

| File | Description |
|------|-------------|
| `streaming_server.py` | Main FastAPI server with WebSocket streaming |
| `config/.env.example` | Environment configuration template |
| `systemd/faster-whisper.service` | Systemd service file |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### CUDA Out of Memory

```bash
# Use INT8 quantization
WHISPER_COMPUTE_TYPE=int8

# Or use smaller model
WHISPER_MODEL=medium
```

### Diarization Not Working

1. Set `HF_TOKEN` in `.env`
2. Accept pyannote license at HuggingFace
3. Check logs: `journalctl -u faster-whisper -f`

### High Latency

- Reduce `CHUNK_DURATION_SECONDS` in code (default: 2.0)
- Use smaller model for real-time applications
- Ensure GPU is being used (check `nvidia-smi`)

## License

- **faster-whisper**: MIT License
- **Whisper models**: MIT License (OpenAI)
- **This code**: MIT License
- **Pyannote**: MIT License (check model-specific licenses)

## References

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
