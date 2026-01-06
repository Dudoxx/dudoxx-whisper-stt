# ddx-ai-whisper - Speech-to-Text Service

**Version:** 6.0.0 | **Port:** 4300 | **Status:** Experimental

---

## Quick Reference

| Item | Value |
|------|-------|
| Tech | FastAPI, MLX-Whisper, Silero VAD |
| Models | Whisper (tiny -> large-v3-turbo) |
| Backend (macOS) | MLX-Whisper (5-6x faster) |
| Backend (GPU) | Faster-Whisper |
| Integration | WebSocket streaming |

---

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e . && pip install mlx-whisper

# Configure
cp .env.example .env

# Run
dudoxx-stt --model large-v3-turbo --language en --port 4300
```

---

## Architecture

```
Browser (MediaRecorder)
    |
    v
WebSocket (ws://localhost:4300/ws)
    |
    v
FastAPI Server
    |
    ├── Silero VAD (noise filtering)
    └── Whisper (MLX/Faster)
         |
         v
    JSON Response (transcript + timestamps)
```

---

## Configuration

```env
# Server
STT_PORT=4300
LOG_LEVEL=INFO

# Model
WHISPER_MODEL=large-v3-turbo    # Recommended
WHISPER_BACKEND=mlx-whisper     # macOS: mlx-whisper, GPU: faster-whisper
DEFAULT_LANGUAGE=auto

# Performance
BACKEND_POLICY=simulstreaming   # Ultra-low latency
VAD_ENABLED=true
```

---

## Model Selection

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Testing |
| base | 74M | Fast | Real-time apps |
| large-v3-turbo | 809M | Fast | **Production (recommended)** |

---

## API

**WebSocket**: `ws://localhost:4300/ws?language=en`

**Response:**
```json
{
  "type": "transcript_update",
  "status": "active_transcription",
  "lines": [
    {"speaker": 1, "text": "Hello", "start": 0.0, "end": 1.5}
  ],
  "buffer_transcription": "pending text..."
}
```

---

## Client Integration

```typescript
const ws = new WebSocket('ws://localhost:4300/ws?language=en');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const text = data.lines?.map(l => l.text).join(' ');
  console.log('Transcript:', text);
};

// Send audio chunks
const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
recorder.ondataavailable = (e) => ws.send(e.data);
recorder.start(100);  // 100ms chunks
```

---

## Performance

| Backend | Encoder Time (base) | Memory |
|---------|---------------------|--------|
| MLX-Whisper (M4) | 0.07s | ~150MB |
| Faster-Whisper | 0.40s | ~500MB |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| High latency | Use smaller model or enable MLX |
| Poor quality | Specify language, use larger model |
| OOM | Use smaller model, disable diarization |

---

## Extended Docs

- `docs/API.md` - WebSocket API reference
- `docs/supported_languages.md` - 99 languages
- `INSTALL.md` - Full setup guide

---

**Last Updated:** January 6, 2026
