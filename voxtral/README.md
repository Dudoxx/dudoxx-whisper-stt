# Voxtral Mini 3B - Streaming ASR Server

Voxtral Mini 3B is Mistral AI's open-weight speech understanding model (Apache 2.0 license).
This module provides a production-ready streaming ASR server with enhanced features.

## Features

- **Real-time Streaming**: WebSocket-based streaming transcription with VAD
- **Speaker Diarization**: Identify different speakers using pyannote.audio 3.1
- **Named Entity Recognition**: Detect names, dates, locations, organizations (GLiNER)
- **Punctuation Restoration**: Automatic punctuation with fullstop-multilang-large
- **Semantic Paragraphs**: Topic-based paragraph detection
- **Multi-language**: English, German, French, Spanish, Arabic, and more

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client (Browser/App)                        │
│                  WebSocket: wss://host/asr                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │ PCM Audio (Int16, 16kHz)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Voxtral Streaming Server (Port 4302)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Silero   │  │ Pyannote │  │ GLiNER   │  │ Punctuation      │ │
│  │ VAD      │  │ Diarize  │  │ NER      │  │ Fullstop-Large   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP POST /v1/audio/transcriptions
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 vLLM Backend (Port 4310)                        │
│                 Voxtral Mini 3B (3.3B params)                   │
│                 ~14GB GPU Memory (60% utilization)              │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Memory Requirements

| Component | GPU Memory | Notes |
|-----------|-----------|-------|
| Voxtral Mini 3B (vLLM) | ~14GB | 60% utilization |
| Pyannote Diarization | ~1.2GB | Speaker identification |
| GLiNER NER | ~750MB | Entity recognition |
| Punctuation Model | ~1.5GB | fullstop-multilang-large |
| Sentence Embeddings | ~500MB | Paragraph detection |
| **Total** | **~18GB** | Requires 24GB GPU |

For smaller GPUs, disable features in the streaming server.

## Installation

### 1. Create Installation Directory

```bash
sudo mkdir -p /opt/voxtral
sudo chown $USER:$USER /opt/voxtral
```

### 2. Copy Files

```bash
# Copy Python files
cp voxtral_streaming_server.py /opt/voxtral/
cp asr_enhancers.py /opt/voxtral/
cp text_processor.py /opt/voxtral/

# Copy configuration
cp config/.env.example /opt/voxtral/.env
```

### 3. Create Virtual Environment

```bash
cd /opt/voxtral
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

Edit `/opt/voxtral/.env`:

```bash
# Set your HuggingFace token (required for diarization)
HF_TOKEN=your_token_here

# Adjust GPU memory based on your hardware
VOXTRAL_GPU_MEMORY_UTILIZATION=0.60
```

### 5. Accept Pyannote License

Visit and accept the license:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 6. Install Systemd Services

```bash
sudo cp systemd/voxtral-vllm.service /etc/systemd/system/
sudo cp systemd/voxtral-proxy.service /etc/systemd/system/

# Edit services to match your user/paths
sudo nano /etc/systemd/system/voxtral-vllm.service

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable voxtral-vllm voxtral-proxy
sudo systemctl start voxtral-vllm voxtral-proxy
```

### 7. Setup Apache2 Proxy (Optional)

See `apache2/` directory for reverse proxy configuration with WebSocket support.

## Usage

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://your-domain/asr?language=auto');

// Send PCM audio (Int16, 16kHz, mono)
ws.send(audioBuffer);

// Receive transcription
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);           // Punctuated text
  console.log(data.entities);       // Named entities
  console.log(data.diarization);    // Speaker info
};
```

### HTTP API

```bash
# Transcribe audio file
curl -X POST "http://localhost:4302/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=auto"
```

### Response Format

```json
{
  "type": "partial",
  "text": "Hello, my name is Dr. Schmidt from Berlin.",
  "raw_text": "hello my name is dr schmidt from berlin",
  "timing": {"start": 0.0, "end": 2.5, "duration": 2.5},
  "entities": [
    {"text": "Dr. Schmidt", "label": "PERSON", "confidence": 0.89},
    {"text": "Berlin", "label": "LOCATION", "confidence": 0.94}
  ],
  "diarization": {
    "current_speaker": "Speaker 1",
    "speakers": [{"label": "Speaker 1", "total_duration": 2.5}]
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

## Files

| File | Description |
|------|-------------|
| `voxtral_streaming_server.py` | Main FastAPI server with WebSocket streaming |
| `asr_enhancers.py` | VAD (Silero) and Speaker Diarization (Pyannote) |
| `text_processor.py` | Punctuation, NER (GLiNER), Paragraph detection |
| `config/.env.example` | Environment configuration template |
| `systemd/*.service` | Systemd service files |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### CUDA Out of Memory

Reduce GPU memory utilization:
```bash
# In .env
VOXTRAL_GPU_MEMORY_UTILIZATION=0.50
```

Or disable features:
```python
# In voxtral_streaming_server.py
ENABLE_NEURAL_VAD = True
ENABLE_DIARIZATION = False  # Disable to save ~1.2GB
```

### Diarization Not Working

1. Check HF_TOKEN is set in `.env`
2. Accept pyannote license at HuggingFace
3. Check logs: `journalctl -u voxtral-proxy -f`

### Slow First Request

Models are loaded lazily. First request triggers model loading (~30s).
Use preloading at startup (already enabled in the streaming server).

## License

- **Voxtral Mini 3B**: Apache 2.0 (Commercial use allowed)
- **This code**: Apache 2.0
- **Pyannote**: MIT License (Check model-specific licenses)
- **GLiNER**: Apache 2.0

## References

- [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [GLiNER](https://github.com/urchade/GLiNER)
- [Silero VAD](https://github.com/snakers4/silero-vad)
