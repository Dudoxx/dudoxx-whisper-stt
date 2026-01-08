# IMPORTANT.md - Dudoxx Whisper STT Critical Paths

**Version:** 6.0.0 | **Date:** December 12, 2025  
**Author:** Walid Boudabbous, Founder and CTO of Dudoxx UG, CEO of Acceleate.com

---

## Critical System Information

**Port:** 4300  
**Status:** EXPERIMENTAL  
**Stack:** FastAPI + Whisper (SimulStreaming) + MLX/Faster-Whisper  
**Purpose:** Ultra-low-latency self-hosted speech-to-text for healthcare

---

## Critical File Paths

| Path | Purpose |
|------|---------|
| `whisperlivekit/basic_server.py` | FastAPI WebSocket server |
| `whisperlivekit/audio_processor.py` | Audio processing pipeline |
| `whisperlivekit/core.py` | Core transcription logic |
| `whisperlivekit/simul_whisper/` | SimulStreaming implementation |
| `whisperlivekit/simul_whisper/mlx_encoder.py` | Apple Silicon encoder |
| `whisperlivekit/diarization/` | Speaker identification |
| `faster-whisper/streaming_server.py` | Alternative lightweight server |

---

## Streaming Server Architecture

```
Browser (MediaRecorder/AudioWorklet)
    │
    ▼ WebSocket (ws://localhost:4300/ws?language=en)
    │
FastAPI Server
    │
    ├── Voice Activity Detection (Silero VAD)
    │
    ├── Transcription Engine
    │   ├── SimulStreaming (AlignAtt) ← SOTA, ultra-low latency
    │   ├── MLX-Whisper (Apple Silicon) ← 5-6x faster
    │   └── Faster-Whisper (NVIDIA GPU)
    │
    ├── Speaker Diarization (Optional)
    │   ├── Streaming Sortformer (NVIDIA)
    │   └── Diart/Pyannote (CPU/GPU)
    │
    └── JSON Response → WebSocket → Browser
```

---

## Model Selection Guide

| Model | Size | VRAM | Speed | Quality | Use Case |
|-------|------|------|-------|---------|----------|
| `tiny` | 39M | ~150MB | Fastest | Basic | Quick testing |
| `base` | 74M | ~200MB | Very fast | Good | Real-time mobile |
| `small` | 244M | ~500MB | Fast | Better | Desktop apps |
| `medium` | 769M | ~1.5GB | Moderate | High | Medical transcription |
| `large-v3-turbo` | 809M | ~1.5GB | **Fast** | **Excellent** | **RECOMMENDED** |
| `large-v3` | 1550M | ~3GB | Slower | Excellent | Production archive |

**Recommendation:** Use `large-v3-turbo` for best balance of speed and quality.

---

## Backend Selection

| Backend | Hardware | Performance | When to Use |
|---------|----------|-------------|-------------|
| `mlx-whisper` | Apple Silicon (M1/M2/M3/M4) | **5-6x faster** | Always on Mac |
| `faster-whisper` | NVIDIA GPU | Fast with CUDA | Linux servers |
| `whisper` | CPU | Slowest | Fallback only |

### Apple Silicon (MLX-Whisper)

```bash
# Install MLX backend
pip install mlx-whisper

# Configure
WHISPER_BACKEND=mlx-whisper
```

**Performance on M4:**
- `base.en`: 0.07s encoder time (vs. 0.35s standard)
- `small`: 0.20s encoder time (vs. 1.09s standard)

---

## Performance Tuning

### Latency Optimization

| Parameter | Impact | Recommended Value |
|-----------|--------|-------------------|
| `BACKEND_POLICY` | Streaming strategy | `simulstreaming` |
| `MIN_CHUNK_SIZE` | Processing granularity | `0.1` (100ms) |
| `VAD_ENABLED` | Filter silence | `true` |
| `BUFFER_TRIMMING` | Memory management | `segment` |
| `BUFFER_TRIMMING_SEC` | Buffer threshold | `15` |

### Memory Optimization

| Configuration | VRAM | RAM |
|--------------|------|-----|
| `tiny + mlx-whisper` | ~150MB | ~500MB |
| `base + mlx-whisper` | ~200MB | ~800MB |
| `large-v3-turbo + mlx-whisper` | ~1.5GB | ~3GB |

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:4300/ws?language=en');
```

### Response Format

```json
{
  "type": "transcript_update",
  "status": "active_transcription",
  "lines": [
    {
      "speaker": 1,
      "text": "Transcribed text here",
      "start": 0.0,
      "end": 2.5,
      "detected_language": "en"
    }
  ],
  "buffer_transcription": "Pending text...",
  "remaining_time_transcription": 0.5
}
```

### Client Implementation

```typescript
// 1. Connect WebSocket
const ws = new WebSocket('ws://localhost:4300/ws?language=auto');

// 2. Capture audio
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

// 3. Send audio chunks
mediaRecorder.ondataavailable = (e) => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(e.data);
  }
};

// 4. Receive transcripts
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Transcript:', data.lines);
};

// 5. Start recording (100ms chunks)
mediaRecorder.start(100);
```

---

## Speaker Diarization

### Enable Diarization

```bash
DIARIZATION_ENABLED=true
DIARIZATION_BACKEND=diart  # or sortformer
HF_TOKEN=your_huggingface_token
```

### Required Setup for Pyannote

1. Accept model terms on HuggingFace:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding

2. Get HuggingFace token:
   - https://huggingface.co/settings/tokens

3. Set token in environment:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```

---

## Integration with Dudoxx Platform

### Port Allocation

| Service | Port |
|---------|------|
| Next.js Frontend | 4000 |
| NestJS Backend | 4100 |
| Calendar Microservice | 4200 |
| **Whisper STT** | **4300** |
| Patient Intake | 4400 |
| LiveKit Voice | 4500 |
| Kokoro TTS | 4600 |

### Use Cases

1. **Medical Transcription** - Doctor-patient consultations
2. **Video Call Captioning** - Live captions for Jitsi
3. **Clinical Documentation** - SOAP notes automation
4. **Voice-to-FHIR** - Structured data extraction (future)

---

## Common Issues & Quick Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| WebSocket "Connection refused" | Server not running | Start `dudoxx-stt` |
| "Model download failed" | Network timeout | Pre-download models |
| High latency (>2s) | Wrong backend | Use MLX on Mac, Faster on GPU |
| Poor quality | Small model | Use `large-v3-turbo` |
| OOM error | Insufficient memory | Use smaller model or disable diarization |
| No speaker labels | Diarization disabled | Enable + set HF_TOKEN |

---

## Quick Start Commands

### WhisperLiveKit Server

```bash
# Install
pip install -e .
pip install mlx-whisper  # Mac only

# Run
dudoxx-stt --model large-v3-turbo --language en --port 4300

# With diarization
dudoxx-stt --model large-v3-turbo --port 4300 --diarization
```

### Faster-Whisper Server (Alternative)

```bash
cd faster-whisper
pip install -r requirements.txt
./start.sh
```

### Docker

```bash
# CPU
docker build -f Dockerfile.cpu -t ddx-whisper .
docker run -p 4300:8000 ddx-whisper

# GPU
docker run --gpus all -p 4300:8000 ddx-whisper
```

---

## Security (HIPAA Compliance)

- **Self-Hosted:** All processing on-premises
- **No Cloud API:** Audio never leaves network
- **No Storage:** Audio not retained by default
- **WSS in Production:** Use HTTPS/WSS for encrypted transport

---

## Related Documentation

- `CLAUDE.md` - Complete operational manual
- `ENV_VARIABLES.md` - Environment configuration
- `docs/API.md` - WebSocket API reference
- `docs/supported_languages.md` - 99 supported languages
- `docs/troubleshooting.md` - Common issues
