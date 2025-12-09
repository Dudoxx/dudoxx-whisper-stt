# Voxtral Mini 3B - Production Setup Guide

## Option A: Maximum Accuracy (Commercial Use)

**License**: Apache 2.0 (Commercial use allowed)  
**Languages**: English, German, French, Spanish, Portuguese, Hindi, Dutch, Italian  
**Model Size**: 3B parameters  
**VRAM Required**: ~8GB (FP16) / ~4GB (INT8)

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Server Deployment](#server-deployment)
5. [API Reference](#api-reference)
6. [Frontend Integration](#frontend-integration)
7. [Live Streaming Implementation](#live-streaming-implementation)
8. [Production Configuration](#production-configuration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Voxtral Mini 3B is Mistral AI's open-weight speech understanding model released in July 2025. It combines:

- **Speech-to-Text Transcription**: High-accuracy multilingual transcription
- **Audio Understanding**: Q&A, summarization, and analysis of audio content
- **Long-Form Support**: Up to 30 minutes of audio in a single request
- **Multilingual**: 8 languages with automatic detection

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Voxtral Mini 3B                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Audio Input (WAV/MP3/FLAC/OGG)                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                  │
│   │     Audio Encoder (Whisper-based)       │                  │
│   │     - Mel spectrogram extraction        │                  │
│   │     - Audio feature encoding            │                  │
│   └─────────────────────────────────────────┘                  │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                  │
│   │     Ministral-3B LLM Backbone           │                  │
│   │     - Text generation                   │                  │
│   │     - Audio understanding               │                  │
│   │     - Multilingual support              │                  │
│   └─────────────────────────────────────────┘                  │
│          │                                                      │
│          ▼                                                      │
│   Transcription / Understanding Output                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3080 (10GB) | NVIDIA RTX 4090 (24GB) |
| **VRAM** | 8GB | 16GB+ |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB | 50GB SSD |
| **CUDA** | 12.1+ | 12.4+ |

### Software

| Component | Version |
|-----------|---------|
| **Python** | 3.10+ |
| **vLLM** | 0.10.0+ |
| **CUDA** | 12.1+ |
| **cuDNN** | 8.9+ |

---

## Installation

### Step 1: Create Virtual Environment

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv voxtral-env
source voxtral-env/bin/activate

# Or using standard venv
python3 -m venv voxtral-env
source voxtral-env/bin/activate
```

### Step 2: Install vLLM with Audio Support

```bash
# Install vLLM with audio support
uv pip install -U "vllm[audio]" --torch-backend=auto

# Install Mistral Common for tokenization
uv pip install -U "mistral_common[audio]"

# Install additional dependencies
uv pip install fastapi uvicorn websockets python-multipart aiofiles
```

### Step 3: Verify Installation

```bash
# Check vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Download Model (Optional - Auto-downloads on first use)

```bash
# Pre-download model
huggingface-cli download mistralai/Voxtral-Mini-3B-2507

# Or with specific cache directory
HF_HOME=/path/to/cache huggingface-cli download mistralai/Voxtral-Mini-3B-2507
```

---

## Server Deployment

### Option 1: vLLM OpenAI-Compatible Server

```bash
# Start vLLM server with Voxtral
vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --host 0.0.0.0 \
  --port 4300 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --dtype float16
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile.voxtral
FROM vllm/vllm-openai:latest

# Set environment variables
ENV HF_HOME=/cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Pre-download model (optional)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('mistralai/Voxtral-Mini-3B-2507')"

# Expose port
EXPOSE 4300

# Start server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "mistralai/Voxtral-Mini-3B-2507", \
     "--host", "0.0.0.0", \
     "--port", "4300", \
     "--tokenizer-mode", "mistral", \
     "--config-format", "mistral", \
     "--load-format", "mistral"]
```

```bash
# Build and run
docker build -f Dockerfile.voxtral -t voxtral-server .
docker run --gpus all -p 4300:4300 -v ~/.cache/huggingface:/cache voxtral-server
```

### Option 3: Docker Compose

```yaml
# docker-compose.voxtral.yml
version: '3.8'

services:
  voxtral:
    image: vllm/vllm-openai:latest
    ports:
      - "4300:4300"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
    command: >
      --model mistralai/Voxtral-Mini-3B-2507
      --host 0.0.0.0
      --port 4300
      --tokenizer-mode mistral
      --config-format mistral
      --load-format mistral
      --max-model-len 32768
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4300/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose -f docker-compose.voxtral.yml up -d
```

---

## API Reference

### Transcription Endpoint

**POST** `/v1/audio/transcriptions`

```bash
# Basic transcription
curl -X POST http://localhost:4300/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=mistralai/Voxtral-Mini-3B-2507" \
  -F "language=de"

# With word-level timestamps
curl -X POST http://localhost:4300/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=mistralai/Voxtral-Mini-3B-2507" \
  -F "timestamp_granularities=word"
```

**Response**:
```json
{
  "text": "Guten Tag, wie kann ich Ihnen helfen?",
  "language": "de",
  "duration": 3.5,
  "segments": [
    {
      "start": 0.0,
      "end": 1.2,
      "text": "Guten Tag,",
      "confidence": 0.98
    },
    {
      "start": 1.2,
      "end": 3.5,
      "text": "wie kann ich Ihnen helfen?",
      "confidence": 0.97
    }
  ]
}
```

### Chat Completion with Audio

**POST** `/v1/chat/completions`

```bash
# Audio Q&A
curl -X POST http://localhost:4300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-Mini-3B-2507",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "audio_url",
            "audio_url": {
              "url": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10..."
            }
          },
          {
            "type": "text",
            "text": "Transcribe this audio in German"
          }
        ]
      }
    ],
    "temperature": 0.0
  }'
```

---

## Frontend Integration

### React/Next.js Client

```typescript
// lib/voxtral-client.ts
import { useState, useCallback, useRef } from 'react';

interface TranscriptionResult {
  text: string;
  language: string;
  duration: number;
  segments: Array<{
    start: number;
    end: number;
    text: string;
    confidence: number;
  }>;
}

interface VoxtralConfig {
  serverUrl: string;
  language?: string;
  model?: string;
}

export function useVoxtralTranscription(config: VoxtralConfig) {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const transcribeAudio = useCallback(async (audioBlob: Blob): Promise<TranscriptionResult> => {
    setIsTranscribing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');
      formData.append('model', config.model || 'mistralai/Voxtral-Mini-3B-2507');
      
      if (config.language) {
        formData.append('language', config.language);
      }

      const response = await fetch(`${config.serverUrl}/v1/audio/transcriptions`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.statusText}`);
      }

      const data: TranscriptionResult = await response.json();
      setResult(data);
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setIsTranscribing(false);
    }
  }, [config]);

  return {
    transcribeAudio,
    isTranscribing,
    result,
    error,
  };
}
```

### Audio Recording Hook

```typescript
// hooks/use-audio-recorder.ts
import { useState, useRef, useCallback } from 'react';

interface AudioRecorderOptions {
  sampleRate?: number;
  channelCount?: number;
  mimeType?: string;
}

export function useAudioRecorder(options: AudioRecorderOptions = {}) {
  const {
    sampleRate = 16000,
    channelCount = 1,
    mimeType = 'audio/webm;codecs=opus',
  } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate,
          channelCount,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
      throw err;
    }
  }, [sampleRate, channelCount, mimeType]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  return {
    isRecording,
    audioBlob,
    startRecording,
    stopRecording,
  };
}
```

### Complete React Component

```tsx
// components/VoxtralTranscriber.tsx
'use client';

import { useState } from 'react';
import { useAudioRecorder } from '@/hooks/use-audio-recorder';
import { useVoxtralTranscription } from '@/lib/voxtral-client';

interface VoxtralTranscriberProps {
  serverUrl: string;
  language?: 'en' | 'de' | 'fr' | 'es' | 'pt' | 'hi' | 'nl' | 'it' | 'auto';
}

export function VoxtralTranscriber({ 
  serverUrl, 
  language = 'auto' 
}: VoxtralTranscriberProps) {
  const [transcript, setTranscript] = useState<string>('');
  
  const { isRecording, audioBlob, startRecording, stopRecording } = useAudioRecorder();
  const { transcribeAudio, isTranscribing, error } = useVoxtralTranscription({
    serverUrl,
    language: language === 'auto' ? undefined : language,
  });

  const handleStopRecording = async () => {
    stopRecording();
    
    // Wait for audioBlob to be available
    setTimeout(async () => {
      if (audioBlob) {
        try {
          const result = await transcribeAudio(audioBlob);
          setTranscript(result.text);
        } catch (err) {
          console.error('Transcription failed:', err);
        }
      }
    }, 100);
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex gap-2">
        {!isRecording ? (
          <button
            onClick={startRecording}
            disabled={isTranscribing}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            Start Recording
          </button>
        ) : (
          <button
            onClick={handleStopRecording}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Stop Recording
          </button>
        )}
      </div>

      {isTranscribing && (
        <div className="text-gray-500">Transcribing...</div>
      )}

      {error && (
        <div className="text-red-500">Error: {error}</div>
      )}

      {transcript && (
        <div className="p-4 bg-gray-100 rounded">
          <h3 className="font-bold mb-2">Transcript:</h3>
          <p>{transcript}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Live Streaming Implementation

> **Note**: As of July 2025, Voxtral does not natively support streaming audio input via vLLM. 
> The following implementation uses a chunked approach for near-real-time transcription.

### Chunked Streaming Server

```python
# voxtral_streaming_server.py
import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional
import tempfile

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voxtral Streaming Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VOXTRAL_SERVER_URL = "http://localhost:4300"
CHUNK_DURATION_SECONDS = 3.0  # Process every 3 seconds of audio
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = int(CHUNK_DURATION_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)


class VoxtralStreamingProcessor:
    """Handles chunked audio processing for near-real-time transcription."""
    
    def __init__(self, language: str = "auto"):
        self.language = language
        self.audio_buffer = bytearray()
        self.full_transcript = []
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process an audio chunk and return transcription."""
        self.audio_buffer.extend(audio_data)
        
        # Only process when we have enough audio
        if len(self.audio_buffer) < CHUNK_SIZE_BYTES:
            return None
        
        # Extract chunk for processing
        chunk = bytes(self.audio_buffer[:CHUNK_SIZE_BYTES])
        self.audio_buffer = self.audio_buffer[CHUNK_SIZE_BYTES:]
        
        # Create WAV file
        wav_data = self._create_wav(chunk)
        
        # Send to Voxtral
        try:
            files = {"file": ("chunk.wav", wav_data, "audio/wav")}
            data = {"model": "mistralai/Voxtral-Mini-3B-2507"}
            
            if self.language != "auto":
                data["language"] = self.language
            
            response = await self.client.post(
                f"{VOXTRAL_SERVER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    self.full_transcript.append(text)
                    return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        
        return None
    
    async def flush(self) -> Optional[str]:
        """Process remaining audio in buffer."""
        if len(self.audio_buffer) < SAMPLE_RATE * BYTES_PER_SAMPLE * 0.5:  # < 0.5s
            return None
        
        chunk = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        
        wav_data = self._create_wav(chunk)
        
        try:
            files = {"file": ("final.wav", wav_data, "audio/wav")}
            data = {"model": "mistralai/Voxtral-Mini-3B-2507"}
            
            if self.language != "auto":
                data["language"] = self.language
            
            response = await self.client.post(
                f"{VOXTRAL_SERVER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    self.full_transcript.append(text)
                    return text
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
        
        return None
    
    def _create_wav(self, pcm_data: bytes) -> bytes:
        """Create WAV file from PCM data."""
        import struct
        
        # WAV header
        channels = 1
        sample_width = 2
        frame_rate = SAMPLE_RATE
        
        data_size = len(pcm_data)
        file_size = data_size + 36
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            file_size,
            b'WAVE',
            b'fmt ',
            16,  # Subchunk1Size
            1,   # AudioFormat (PCM)
            channels,
            frame_rate,
            frame_rate * channels * sample_width,  # ByteRate
            channels * sample_width,  # BlockAlign
            sample_width * 8,  # BitsPerSample
            b'data',
            data_size,
        )
        
        return header + pcm_data
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript."""
        return " ".join(self.full_transcript)
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket, language: str = "auto"):
    """WebSocket endpoint for streaming ASR."""
    await websocket.accept()
    logger.info(f"WebSocket connected, language: {language}")
    
    processor = VoxtralStreamingProcessor(language=language)
    
    try:
        # Send initial config
        await websocket.send_json({
            "type": "config",
            "language": language,
            "chunk_duration": CHUNK_DURATION_SECONDS,
        })
        
        while True:
            # Receive audio data
            message = await websocket.receive_bytes()
            
            if not message:
                # Empty message signals end of stream
                final_text = await processor.flush()
                if final_text:
                    await websocket.send_json({
                        "type": "final",
                        "text": final_text,
                        "full_transcript": processor.get_full_transcript(),
                    })
                break
            
            # Process audio chunk
            text = await processor.process_chunk(message)
            
            if text:
                await websocket.send_json({
                    "type": "partial",
                    "text": text,
                    "full_transcript": processor.get_full_transcript(),
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await processor.close()
        logger.info("WebSocket cleanup complete")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "voxtral-mini-3b"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4301)
```

### Frontend WebSocket Client

```typescript
// lib/voxtral-streaming-client.ts
export interface VoxtralStreamingConfig {
  serverUrl: string;
  language?: string;
  onPartialTranscript?: (text: string, fullTranscript: string) => void;
  onFinalTranscript?: (text: string, fullTranscript: string) => void;
  onError?: (error: string) => void;
  onConnectionChange?: (connected: boolean) => void;
}

export class VoxtralStreamingClient {
  private ws: WebSocket | null = null;
  private config: VoxtralStreamingConfig;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private isStreaming = false;

  constructor(config: VoxtralStreamingConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    const wsUrl = this.config.serverUrl.replace(/^http/, 'ws');
    const language = this.config.language || 'auto';
    
    this.ws = new WebSocket(`${wsUrl}/asr?language=${language}`);
    
    return new Promise((resolve, reject) => {
      if (!this.ws) return reject(new Error('WebSocket not initialized'));

      this.ws.onopen = () => {
        this.config.onConnectionChange?.(true);
        resolve();
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'partial':
            this.config.onPartialTranscript?.(data.text, data.full_transcript);
            break;
          case 'final':
            this.config.onFinalTranscript?.(data.text, data.full_transcript);
            break;
          case 'error':
            this.config.onError?.(data.message);
            break;
        }
      };

      this.ws.onerror = (error) => {
        this.config.onError?.('WebSocket error');
        reject(error);
      };

      this.ws.onclose = () => {
        this.config.onConnectionChange?.(false);
      };
    });
  }

  async startStreaming(): Promise<void> {
    if (this.isStreaming) return;

    // Get microphone access
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // Create audio context
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    // Create script processor for raw PCM access
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    
    this.processor.onaudioprocess = (event) => {
      if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const inputData = event.inputBuffer.getChannelData(0);
      
      // Convert Float32 to Int16
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Send PCM data
      this.ws.send(pcmData.buffer);
    };

    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
    
    this.isStreaming = true;
  }

  async stopStreaming(): Promise<void> {
    this.isStreaming = false;

    // Send empty message to signal end
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(new ArrayBuffer(0));
    }

    // Clean up audio
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
  }

  disconnect(): void {
    this.stopStreaming();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
```

### React Streaming Component

```tsx
// components/VoxtralStreamingTranscriber.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { VoxtralStreamingClient } from '@/lib/voxtral-streaming-client';

interface VoxtralStreamingTranscriberProps {
  serverUrl: string;
  language?: 'en' | 'de' | 'fr' | 'es' | 'auto';
}

export function VoxtralStreamingTranscriber({
  serverUrl,
  language = 'auto',
}: VoxtralStreamingTranscriberProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [partialText, setPartialText] = useState('');
  const [fullTranscript, setFullTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  const clientRef = useRef<VoxtralStreamingClient | null>(null);

  useEffect(() => {
    // Initialize client
    clientRef.current = new VoxtralStreamingClient({
      serverUrl,
      language,
      onPartialTranscript: (text, full) => {
        setPartialText(text);
        setFullTranscript(full);
      },
      onFinalTranscript: (text, full) => {
        setPartialText('');
        setFullTranscript(full);
      },
      onError: (err) => setError(err),
      onConnectionChange: (connected) => setIsConnected(connected),
    });

    return () => {
      clientRef.current?.disconnect();
    };
  }, [serverUrl, language]);

  const handleConnect = async () => {
    try {
      await clientRef.current?.connect();
      setError(null);
    } catch (err) {
      setError('Failed to connect');
    }
  };

  const handleStartStreaming = async () => {
    try {
      await clientRef.current?.startStreaming();
      setIsStreaming(true);
      setError(null);
    } catch (err) {
      setError('Failed to start streaming');
    }
  };

  const handleStopStreaming = async () => {
    await clientRef.current?.stopStreaming();
    setIsStreaming(false);
  };

  return (
    <div className="flex flex-col gap-4 p-4 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold">Voxtral Streaming Transcription</h2>
      
      {/* Connection Status */}
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
        <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        {!isConnected ? (
          <button
            onClick={handleConnect}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Connect
          </button>
        ) : !isStreaming ? (
          <button
            onClick={handleStartStreaming}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Start Streaming
          </button>
        ) : (
          <button
            onClick={handleStopStreaming}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 animate-pulse"
          >
            Stop Streaming
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Partial Transcript */}
      {partialText && (
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
          <span className="text-sm text-gray-500">Processing: </span>
          <span className="italic">{partialText}</span>
        </div>
      )}

      {/* Full Transcript */}
      <div className="p-4 bg-gray-50 rounded min-h-32">
        <h3 className="font-semibold mb-2">Transcript:</h3>
        <p className="whitespace-pre-wrap">
          {fullTranscript || 'Start speaking to see transcription...'}
        </p>
      </div>
    </div>
  );
}
```

---

## Production Configuration

### Environment Variables

```bash
# .env.voxtral
# Server Configuration
VOXTRAL_HOST=0.0.0.0
VOXTRAL_PORT=4300
VOXTRAL_WORKERS=1

# Model Configuration
VOXTRAL_MODEL=mistralai/Voxtral-Mini-3B-2507
VOXTRAL_MAX_MODEL_LEN=32768
VOXTRAL_GPU_MEMORY_UTILIZATION=0.9
VOXTRAL_DTYPE=float16

# Quantization (for lower memory usage)
VOXTRAL_QUANTIZATION=none  # Options: none, awq, gptq

# Hugging Face
HF_TOKEN=your_token_here
HF_HOME=/path/to/cache

# Logging
LOG_LEVEL=INFO
```

### Systemd Service

```ini
# /etc/systemd/system/voxtral.service
[Unit]
Description=Voxtral Mini 3B ASR Server
After=network.target

[Service]
Type=simple
User=voxtral
Group=voxtral
WorkingDirectory=/opt/voxtral
EnvironmentFile=/opt/voxtral/.env.voxtral
ExecStart=/opt/voxtral/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model ${VOXTRAL_MODEL} \
    --host ${VOXTRAL_HOST} \
    --port ${VOXTRAL_PORT} \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --max-model-len ${VOXTRAL_MAX_MODEL_LEN} \
    --gpu-memory-utilization ${VOXTRAL_GPU_MEMORY_UTILIZATION} \
    --dtype ${VOXTRAL_DTYPE}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable voxtral
sudo systemctl start voxtral
```

### NGINX Reverse Proxy

```nginx
# /etc/nginx/sites-available/voxtral
upstream voxtral_backend {
    server 127.0.0.1:4300;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name stt.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/stt.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/stt.yourdomain.com/privkey.pem;

    client_max_body_size 100M;  # For large audio files

    location / {
        proxy_pass http://voxtral_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long audio processing
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    location /asr {
        proxy_pass http://127.0.0.1:4301;  # Streaming proxy
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```bash
# Reduce GPU memory utilization
vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 16384

# Or use INT8 quantization
vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --dtype float16 \
  --quantization awq
```

#### 2. Slow First Request

The model needs to be loaded on first request. Pre-warm with:

```bash
curl -X POST http://localhost:4300/v1/audio/transcriptions \
  -F "file=@warmup.wav" \
  -F "model=mistralai/Voxtral-Mini-3B-2507"
```

#### 3. Audio Format Not Supported

Convert audio to WAV format:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav
```

#### 4. WebSocket Connection Drops

Increase timeouts in nginx and check server logs:

```bash
journalctl -u voxtral -f
```

### Performance Tuning

| Setting | Low Memory | Balanced | High Performance |
|---------|------------|----------|------------------|
| `gpu-memory-utilization` | 0.5 | 0.8 | 0.95 |
| `max-model-len` | 8192 | 16384 | 32768 |
| `dtype` | float16 | float16 | bfloat16 |
| `max-num-seqs` | 8 | 16 | 32 |

---

## Next Steps

1. [NVIDIA Canary Setup](./NVIDIA_CANARY_SETUP.md) - For non-commercial use with higher accuracy
2. [Integration Guide](./technical_integration.md) - Integrating with Dudoxx Clinic Platform
3. [API Documentation](./API.md) - Full API reference

---

**License**: Apache 2.0  
**Last Updated**: December 2025  
**Maintainer**: Dudoxx UG
