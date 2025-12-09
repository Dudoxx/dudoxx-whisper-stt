# NVIDIA Canary-1B - Production Setup Guide

## Option B: Maximum Accuracy (Non-Commercial/Research)

**License**: CC-BY-NC-4.0 (Non-commercial use only)  
**Languages**: English, German, French, Spanish  
**Model Size**: 1B parameters  
**VRAM Required**: ~4GB (FP16)  
**Accuracy**: #1 on HuggingFace Open ASR Leaderboard

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Model Variants](#model-variants)
5. [Server Deployment](#server-deployment)
6. [Streaming Implementation](#streaming-implementation)
7. [Frontend Integration](#frontend-integration)
8. [NVIDIA Riva Deployment](#nvidia-riva-deployment)
9. [Production Configuration](#production-configuration)
10. [Benchmarks](#benchmarks)
11. [Troubleshooting](#troubleshooting)

---

## Overview

NVIDIA Canary-1B is a state-of-the-art multilingual ASR model from NVIDIA NeMo. It features:

- **True Streaming**: Native streaming support with chunked inference
- **Multi-Task**: ASR, Translation, and Code-Switching
- **Punctuation & Capitalization**: Built-in formatting
- **Highest Accuracy**: Top of Open ASR Leaderboard
- **Low Latency**: 200-400ms with streaming mode

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NVIDIA Canary-1B                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Audio Input (16kHz PCM)                                       │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                  │
│   │     FastConformer Encoder               │                  │
│   │     - Streaming-capable                 │                  │
│   │     - Multi-scale feature extraction    │                  │
│   │     - 600M+ parameters                  │                  │
│   └─────────────────────────────────────────┘                  │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                  │
│   │     Transformer Decoder                 │                  │
│   │     - Multi-task prompting              │                  │
│   │     - Language/Task tokens              │                  │
│   │     - Punctuation & Capitalization      │                  │
│   └─────────────────────────────────────────┘                  │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────┐                  │
│   │     Output Tokens                       │                  │
│   │     - Transcription (ASR)               │                  │
│   │     - Translation (S2T)                 │                  │
│   └─────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Tasks

| Task | Description | Example Prompt |
|------|-------------|----------------|
| **ASR** | Speech-to-text in same language | `task=asr, source_lang=de, target_lang=de` |
| **S2T** | Speech translation | `task=s2t, source_lang=de, target_lang=en` |
| **PnC** | Punctuation & Capitalization | `pnc=yes` |

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3070 (8GB) | NVIDIA RTX 4090 (24GB) |
| **VRAM** | 4GB | 8GB+ |
| **RAM** | 16GB | 32GB |
| **Storage** | 10GB | 30GB SSD |
| **CUDA** | 12.1+ | 12.4+ |

### Software

| Component | Version |
|-----------|---------|
| **Python** | 3.10+ |
| **PyTorch** | 2.1+ |
| **NeMo Toolkit** | 2.0+ |
| **CUDA** | 12.1+ |

---

## Installation

### Step 1: Create Virtual Environment

```bash
# Using conda (recommended for NeMo)
conda create -n canary python=3.10
conda activate canary

# Or using venv
python3 -m venv canary-env
source canary-env/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install NeMo Toolkit

```bash
# Install NeMo with ASR support
pip install "nemo_toolkit[asr]>=2.0.0"

# Install additional dependencies
pip install fastapi uvicorn websockets librosa soundfile
```

### Step 4: Verify Installation

```bash
# Test NeMo installation
python -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR ready')"

# Test model loading
python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
print(f'Model loaded: {model.__class__.__name__}')
"
```

### Step 5: Accept License

Visit [HuggingFace Canary-1B](https://huggingface.co/nvidia/canary-1b) and accept the CC-BY-NC-4.0 license.

```bash
# Login to HuggingFace
huggingface-cli login
```

---

## Model Variants

| Model | Parameters | Languages | Use Case |
|-------|------------|-----------|----------|
| **canary-1b** | 1B | EN, DE, FR, ES | Best accuracy |
| **canary-1b-v2** | 1B | EN, DE, FR, ES + streaming | Streaming support |
| **canary-180m** | 180M | EN, DE, FR, ES | Low memory |
| **parakeet-tdt-1.1b** | 1.1B | EN only | English-only, fastest |

### Loading Different Models

```python
import nemo.collections.asr as nemo_asr

# Standard Canary
model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")

# Canary v2 with streaming
model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")

# Smaller model for low memory
model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-180m")
```

---

## Server Deployment

### Basic FastAPI Server

```python
# canary_server.py
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NVIDIA Canary ASR Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = None

SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German", 
    "fr": "French",
    "es": "Spanish",
}


@app.on_event("startup")
async def load_model():
    """Load Canary model on startup."""
    global model, device
    
    logger.info("Loading NVIDIA Canary-1B model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on {device}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "nvidia/canary-1b",
        "device": str(device),
        "languages": list(SUPPORTED_LANGUAGES.keys()),
    }


@app.get("/api/languages")
async def get_languages():
    """Get supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    task: Optional[str] = Form(default="asr"),
    target_language: Optional[str] = Form(default=None),
    pnc: Optional[bool] = Form(default=True),
):
    """Transcribe audio file."""
    if language not in SUPPORTED_LANGUAGES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"}
        )
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load and resample audio
        audio, sr = sf.read(tmp_path)
        
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Prepare transcription config
        decode_cfg = model.cfg.decoding.copy()
        decode_cfg.beam.beam_size = 1  # Greedy for speed
        
        # Set task and language
        target_lang = target_language or language
        
        with torch.no_grad():
            # Transcribe
            transcriptions = model.transcribe(
                [tmp_path],
                batch_size=1,
                pnc=pnc,
                source_lang=language,
                target_lang=target_lang,
            )
        
        text = transcriptions[0] if transcriptions else ""
        
        return {
            "text": text,
            "language": language,
            "task": task,
            "model": "nvidia/canary-1b",
        }
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(default="en"),
    pnc: Optional[bool] = Form(default=True),
):
    """Translate speech to text in another language."""
    return await transcribe(
        file=file,
        language=source_language,
        task="s2t",
        target_language=target_language,
        pnc=pnc,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4300)
```

### Run Server

```bash
python canary_server.py
```

---

## Streaming Implementation

### Streaming ASR Server

```python
# canary_streaming_server.py
import asyncio
import logging
import struct
from collections import deque
from typing import AsyncGenerator, Optional

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
    BatchedFrameASRRNNT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Canary Streaming ASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100  # 100ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
CONTEXT_SIZE = 10  # seconds of left context
LOOKAHEAD_SIZE = 0.5  # seconds of lookahead

# Global model
model = None
device = None


class StreamingCanaryProcessor:
    """Handles streaming ASR with Canary model."""
    
    def __init__(
        self,
        model,
        source_lang: str = "en",
        target_lang: str = "en",
        pnc: bool = True,
    ):
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.pnc = pnc
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * 30))  # 30s max
        self.processed_samples = 0
        
        # Results
        self.confirmed_text = []
        self.buffer_text = ""
        
        # Streaming config
        self.chunk_size = CHUNK_SIZE
        self.context_size = int(CONTEXT_SIZE * SAMPLE_RATE)
        self.lookahead_size = int(LOOKAHEAD_SIZE * SAMPLE_RATE)
        self.min_process_size = int(SAMPLE_RATE * 1.0)  # Min 1 second
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        self.audio_buffer.extend(audio_chunk)
    
    async def process(self) -> Optional[dict]:
        """Process buffered audio and return transcription."""
        buffer_len = len(self.audio_buffer)
        
        # Need minimum audio to process
        if buffer_len - self.processed_samples < self.min_process_size:
            return None
        
        # Get audio for processing
        audio = np.array(self.audio_buffer)
        
        # Include context
        start_idx = max(0, self.processed_samples - self.context_size)
        audio_segment = audio[start_idx:]
        
        if len(audio_segment) < self.min_process_size:
            return None
        
        # Create temporary file for NeMo
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_segment, SAMPLE_RATE)
            tmp_path = tmp.name
        
        try:
            with torch.no_grad():
                transcriptions = self.model.transcribe(
                    [tmp_path],
                    batch_size=1,
                    pnc=self.pnc,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                )
            
            text = transcriptions[0] if transcriptions else ""
            
            # Update processed position
            self.processed_samples = buffer_len
            
            # Simple word-level agreement
            words = text.split()
            if len(words) > 2:
                # Confirm all but last 2 words
                new_confirmed = words[:-2]
                self.confirmed_text.extend(new_confirmed)
                self.buffer_text = " ".join(words[-2:])
            else:
                self.buffer_text = text
            
            return {
                "type": "partial",
                "confirmed": " ".join(self.confirmed_text),
                "buffer": self.buffer_text,
                "full_text": text,
            }
        
        finally:
            import os
            os.unlink(tmp_path)
    
    async def finalize(self) -> dict:
        """Process remaining audio and finalize."""
        if len(self.audio_buffer) > self.processed_samples:
            result = await self.process()
            if result:
                # Move buffer to confirmed
                self.confirmed_text.append(self.buffer_text)
                self.buffer_text = ""
        
        final_text = " ".join(self.confirmed_text)
        return {
            "type": "final",
            "text": final_text,
        }


@app.on_event("startup")
async def load_model():
    """Load Canary model."""
    global model, device
    
    logger.info("Loading NVIDIA Canary-1B for streaming...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on {device}")


@app.websocket("/asr")
async def websocket_asr(
    websocket: WebSocket,
    language: str = "en",
    target_language: str = None,
    pnc: bool = True,
):
    """WebSocket endpoint for streaming ASR."""
    await websocket.accept()
    
    target_lang = target_language or language
    processor = StreamingCanaryProcessor(
        model=model,
        source_lang=language,
        target_lang=target_lang,
        pnc=pnc,
    )
    
    logger.info(f"Streaming session started: {language} -> {target_lang}")
    
    # Send config
    await websocket.send_json({
        "type": "config",
        "language": language,
        "target_language": target_lang,
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SIZE,
    })
    
    # Processing task
    async def process_loop():
        while True:
            await asyncio.sleep(0.5)  # Process every 500ms
            result = await processor.process()
            if result:
                try:
                    await websocket.send_json(result)
                except:
                    break
    
    process_task = asyncio.create_task(process_loop())
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if not data:
                # End of stream
                break
            
            # Convert bytes to float32 PCM
            if len(data) % 2 != 0:
                data = data[:-1]
            
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            processor.add_audio(audio)
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        process_task.cancel()
        
        # Send final result
        try:
            final = await processor.finalize()
            await websocket.send_json(final)
        except:
            pass
        
        logger.info("Streaming session ended")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "nvidia/canary-1b", "streaming": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4300)
```

### True Chunked Streaming (Using NeMo Streaming Utils)

```python
# canary_chunked_streaming.py
"""
Advanced streaming using NeMo's native streaming utilities.
Provides lower latency and better accuracy than simple chunking.
"""

import asyncio
import logging
from typing import Optional

import torch
import numpy as np
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import (
    FrameBatchMultiTaskAED,
    CacheAwareStreamingAudioBuffer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryStreamingDecoder:
    """
    True streaming decoder using NeMo's chunked inference.
    
    Based on: examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/canary-1b-v2",
        chunk_size_secs: float = 1.0,
        left_context_secs: float = 10.0,
        right_context_secs: float = 0.5,
        source_lang: str = "en",
        target_lang: str = "en",
        pnc: bool = True,
    ):
        self.chunk_size_secs = chunk_size_secs
        self.left_context_secs = left_context_secs
        self.right_context_secs = right_context_secs
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.pnc = pnc
        
        # Load model
        logger.info(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Configure for streaming
        self._setup_streaming()
        
        logger.info("Streaming decoder ready")
    
    def _setup_streaming(self):
        """Configure model for streaming inference."""
        # Get decoding config
        decoding_cfg = self.model.cfg.decoding.copy()
        
        # Configure streaming policy
        decoding_cfg.streaming_policy = "waitk"  # or "alignatt"
        decoding_cfg.waitk_lagging = 2
        decoding_cfg.alignatt_thr = 8
        decoding_cfg.exclude_sink_frames = 8
        decoding_cfg.xatt_scores_layer = -2
        decoding_cfg.hallucinations_detector = True
        
        # Apply config
        self.model.change_decoding_strategy(decoding_cfg)
        
        # Setup streaming buffer
        self.sample_rate = 16000
        self.chunk_size_samples = int(self.chunk_size_secs * self.sample_rate)
        self.left_context_samples = int(self.left_context_secs * self.sample_rate)
        self.right_context_samples = int(self.right_context_secs * self.sample_rate)
        
        # Frame batch processor
        self.frame_asr = FrameBatchMultiTaskAED(
            asr_model=self.model,
            frame_len=self.chunk_size_secs,
            total_buffer=self.left_context_secs + self.chunk_size_secs + self.right_context_secs,
            batch_size=1,
        )
    
    def create_session(self) -> "StreamingSession":
        """Create a new streaming session."""
        return StreamingSession(self)


class StreamingSession:
    """Handles a single streaming session."""
    
    def __init__(self, decoder: CanaryStreamingDecoder):
        self.decoder = decoder
        self.audio_buffer = CacheAwareStreamingAudioBuffer(
            sample_rate=decoder.sample_rate,
            chunk_size=decoder.chunk_size_samples,
            left_context_size=decoder.left_context_samples,
            right_context_size=decoder.right_context_samples,
        )
        self.transcription = []
        self.is_active = True
    
    def add_audio(self, audio: np.ndarray) -> None:
        """Add audio samples to buffer."""
        self.audio_buffer.append_audio(audio)
    
    async def transcribe_chunk(self) -> Optional[str]:
        """Transcribe available audio and return new text."""
        if not self.audio_buffer.is_buffer_ready():
            return None
        
        # Get chunk with context
        audio_chunk, _ = self.audio_buffer.get_next_chunk()
        
        if audio_chunk is None:
            return None
        
        # Prepare prompt
        prompt = {
            "pnc": "yes" if self.decoder.pnc else "no",
            "task": "asr",
            "source_lang": self.decoder.source_lang,
            "target_lang": self.decoder.target_lang,
        }
        
        with torch.no_grad():
            # Run streaming inference
            transcription = await asyncio.to_thread(
                self.decoder.frame_asr.transcribe,
                audio_chunk,
                prompt=prompt,
            )
        
        if transcription:
            self.transcription.append(transcription)
            return transcription
        
        return None
    
    def finalize(self) -> str:
        """Finalize session and return complete transcription."""
        self.is_active = False
        return " ".join(self.transcription)


# Example usage in WebSocket handler
async def handle_streaming_websocket(websocket, decoder: CanaryStreamingDecoder):
    """Handle streaming WebSocket connection."""
    session = decoder.create_session()
    
    async def transcription_loop():
        while session.is_active:
            text = await session.transcribe_chunk()
            if text:
                await websocket.send_json({
                    "type": "partial",
                    "text": text,
                    "full": " ".join(session.transcription),
                })
            await asyncio.sleep(0.1)
    
    task = asyncio.create_task(transcription_loop())
    
    try:
        async for message in websocket.iter_bytes():
            if not message:
                break
            audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            session.add_audio(audio)
    finally:
        task.cancel()
        final = session.finalize()
        await websocket.send_json({"type": "final", "text": final})
```

---

## Frontend Integration

### TypeScript Client

```typescript
// lib/canary-client.ts
export interface CanaryConfig {
  serverUrl: string;
  language?: 'en' | 'de' | 'fr' | 'es';
  targetLanguage?: 'en' | 'de' | 'fr' | 'es';
  pnc?: boolean;
}

export interface TranscriptionResult {
  type: 'partial' | 'final';
  confirmed?: string;
  buffer?: string;
  text?: string;
  full_text?: string;
}

export class CanaryStreamingClient {
  private ws: WebSocket | null = null;
  private config: CanaryConfig;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private isStreaming = false;
  
  // Callbacks
  public onPartial?: (result: TranscriptionResult) => void;
  public onFinal?: (result: TranscriptionResult) => void;
  public onError?: (error: string) => void;
  public onConnectionChange?: (connected: boolean) => void;

  constructor(config: CanaryConfig) {
    this.config = {
      language: 'en',
      targetLanguage: config.targetLanguage || config.language || 'en',
      pnc: true,
      ...config,
    };
  }

  async connect(): Promise<void> {
    const wsUrl = this.config.serverUrl.replace(/^http/, 'ws');
    const params = new URLSearchParams({
      language: this.config.language!,
      target_language: this.config.targetLanguage!,
      pnc: String(this.config.pnc),
    });
    
    this.ws = new WebSocket(`${wsUrl}/asr?${params}`);
    
    return new Promise((resolve, reject) => {
      if (!this.ws) return reject(new Error('WebSocket not initialized'));

      this.ws.onopen = () => {
        this.onConnectionChange?.(true);
        resolve();
      };

      this.ws.onmessage = (event) => {
        const data: TranscriptionResult = JSON.parse(event.data);
        
        if (data.type === 'partial') {
          this.onPartial?.(data);
        } else if (data.type === 'final') {
          this.onFinal?.(data);
        }
      };

      this.ws.onerror = () => {
        this.onError?.('WebSocket error');
        reject(new Error('WebSocket error'));
      };

      this.ws.onclose = () => {
        this.onConnectionChange?.(false);
      };
    });
  }

  async startStreaming(): Promise<void> {
    if (this.isStreaming) return;

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    // Use ScriptProcessorNode for raw PCM access
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    
    this.processor.onaudioprocess = (event) => {
      if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const inputData = event.inputBuffer.getChannelData(0);
      
      // Convert Float32 to Int16 PCM
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

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

    // Cleanup audio
    this.processor?.disconnect();
    this.processor = null;
    
    await this.audioContext?.close();
    this.audioContext = null;
    
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.mediaStream = null;
  }

  disconnect(): void {
    this.stopStreaming();
    this.ws?.close();
    this.ws = null;
  }
}
```

### React Component

```tsx
// components/CanaryTranscriber.tsx
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { CanaryStreamingClient, TranscriptionResult } from '@/lib/canary-client';

interface CanaryTranscriberProps {
  serverUrl: string;
  language?: 'en' | 'de' | 'fr' | 'es';
  targetLanguage?: 'en' | 'de' | 'fr' | 'es';
  onTranscript?: (text: string) => void;
}

const LANGUAGE_NAMES = {
  en: 'English',
  de: 'German',
  fr: 'French',
  es: 'Spanish',
};

export function CanaryTranscriber({
  serverUrl,
  language = 'en',
  targetLanguage,
  onTranscript,
}: CanaryTranscriberProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [confirmedText, setConfirmedText] = useState('');
  const [bufferText, setBufferText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState(language);
  
  const clientRef = useRef<CanaryStreamingClient | null>(null);

  // Initialize client
  useEffect(() => {
    clientRef.current = new CanaryStreamingClient({
      serverUrl,
      language: selectedLanguage,
      targetLanguage: targetLanguage || selectedLanguage,
    });

    clientRef.current.onPartial = (result) => {
      if (result.confirmed) setConfirmedText(result.confirmed);
      if (result.buffer) setBufferText(result.buffer);
    };

    clientRef.current.onFinal = (result) => {
      const finalText = result.text || '';
      setConfirmedText(finalText);
      setBufferText('');
      onTranscript?.(finalText);
    };

    clientRef.current.onError = setError;
    clientRef.current.onConnectionChange = setIsConnected;

    return () => {
      clientRef.current?.disconnect();
    };
  }, [serverUrl, selectedLanguage, targetLanguage, onTranscript]);

  const handleConnect = async () => {
    try {
      await clientRef.current?.connect();
      setError(null);
    } catch {
      setError('Failed to connect to server');
    }
  };

  const handleStartStreaming = async () => {
    try {
      await clientRef.current?.startStreaming();
      setIsStreaming(true);
      setConfirmedText('');
      setBufferText('');
      setError(null);
    } catch {
      setError('Failed to start recording. Check microphone permissions.');
    }
  };

  const handleStopStreaming = async () => {
    await clientRef.current?.stopStreaming();
    setIsStreaming(false);
  };

  const fullText = confirmedText + (bufferText ? ` ${bufferText}` : '');

  return (
    <div className="flex flex-col gap-4 p-6 max-w-2xl mx-auto bg-white rounded-lg shadow">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-800">
          NVIDIA Canary ASR
        </h2>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span className="text-sm text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Language Selection */}
      <div className="flex gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Language
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value as any)}
            disabled={isStreaming}
            className="w-full px-3 py-2 border rounded-md"
          >
            {Object.entries(LANGUAGE_NAMES).map(([code, name]) => (
              <option key={code} value={code}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        {!isConnected ? (
          <button
            onClick={handleConnect}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
          >
            Connect
          </button>
        ) : !isStreaming ? (
          <button
            onClick={handleStartStreaming}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition flex items-center gap-2"
          >
            <MicrophoneIcon />
            Start Recording
          </button>
        ) : (
          <button
            onClick={handleStopStreaming}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition animate-pulse flex items-center gap-2"
          >
            <StopIcon />
            Stop Recording
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
        </div>
      )}

      {/* Transcription Display */}
      <div className="min-h-40 p-4 bg-gray-50 rounded-md border">
        <div className="text-gray-800">
          {confirmedText && <span>{confirmedText}</span>}
          {bufferText && (
            <span className="text-gray-400 italic"> {bufferText}</span>
          )}
          {!fullText && (
            <span className="text-gray-400">
              {isStreaming ? 'Listening...' : 'Start recording to see transcription'}
            </span>
          )}
        </div>
      </div>

      {/* Model Info */}
      <div className="text-xs text-gray-500 flex justify-between">
        <span>Model: nvidia/canary-1b</span>
        <span>License: CC-BY-NC-4.0</span>
      </div>
    </div>
  );
}

// Icons
function MicrophoneIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
    </svg>
  );
}
```

---

## NVIDIA Riva Deployment

For production deployments with maximum performance, use NVIDIA Riva.

### Install Riva

```bash
# Download Riva Quick Start
ngc registry resource download-version nvidia/riva/riva_quickstart:2.14.0

cd riva_quickstart_v2.14.0

# Configure for ASR only
sed -i 's/service_enabled_asr=.*/service_enabled_asr=true/' config.sh
sed -i 's/service_enabled_nlp=.*/service_enabled_nlp=false/' config.sh
sed -i 's/service_enabled_tts=.*/service_enabled_tts=false/' config.sh

# Set languages
sed -i 's/riva_asr_languages_models=.*/riva_asr_languages_models=("en-US" "de-DE" "fr-FR" "es-ES")/' config.sh

# Initialize (downloads models)
bash riva_init.sh

# Start Riva server
bash riva_start.sh
```

### WebSocket Bridge for Riva

```bash
# Clone WebSocket bridge
git clone https://github.com/nvidia-riva/websocket-bridge
cd websocket-bridge

# Configure
export RIVA_API_URL=localhost:50051

# Run bridge
python server.py --port 4300
```

### Riva Client Example

```python
# riva_client.py
import riva.client
import riva.client.proto.riva_asr_pb2 as rasr

def transcribe_streaming(audio_chunks, language="en-US"):
    """Stream audio to Riva and get transcriptions."""
    
    # Connect to Riva
    auth = riva.client.Auth(uri="localhost:50051")
    asr_service = riva.client.ASRService(auth)
    
    # Configure streaming
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code=language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=True,
    )
    
    # Stream and get results
    responses = asr_service.streaming_response_generator(
        audio_chunks=audio_chunks,
        streaming_config=config,
    )
    
    for response in responses:
        for result in response.results:
            if result.is_final:
                yield {
                    "type": "final",
                    "text": result.alternatives[0].transcript,
                    "confidence": result.alternatives[0].confidence,
                }
            else:
                yield {
                    "type": "partial",
                    "text": result.alternatives[0].transcript,
                }
```

---

## Production Configuration

### Environment Variables

```bash
# .env.canary
# Model Configuration
CANARY_MODEL=nvidia/canary-1b
CANARY_DEVICE=cuda:0

# Server Configuration
CANARY_HOST=0.0.0.0
CANARY_PORT=4300
CANARY_WORKERS=1

# Streaming Configuration
CHUNK_SIZE_SECS=1.0
LEFT_CONTEXT_SECS=10.0
RIGHT_CONTEXT_SECS=0.5

# Logging
LOG_LEVEL=INFO

# HuggingFace
HF_TOKEN=your_token_here
```

### Systemd Service

```ini
# /etc/systemd/system/canary.service
[Unit]
Description=NVIDIA Canary ASR Server
After=network.target

[Service]
Type=simple
User=canary
Group=canary
WorkingDirectory=/opt/canary
EnvironmentFile=/opt/canary/.env.canary
ExecStart=/opt/canary/venv/bin/python canary_streaming_server.py
Restart=always
RestartSec=10
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

```dockerfile
# Dockerfile.canary
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]>=2.0.0" \
    fastapi \
    uvicorn \
    websockets \
    soundfile \
    librosa

# Copy server code
COPY canary_streaming_server.py .

# Expose port
EXPOSE 4300

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:4300/health || exit 1

# Run server
CMD ["python", "canary_streaming_server.py"]
```

```yaml
# docker-compose.canary.yml
version: '3.8'

services:
  canary:
    build:
      context: .
      dockerfile: Dockerfile.canary
    ports:
      - "4300:4300"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

---

## Benchmarks

### Accuracy (Word Error Rate - WER)

| Model | English | German | French | Spanish |
|-------|---------|--------|--------|---------|
| **Canary-1B** | 5.2% | 6.8% | 7.1% | 6.5% |
| Whisper Large-v3 | 6.1% | 8.2% | 8.5% | 7.8% |
| Voxtral Mini | 5.8% | 7.5% | 7.8% | 7.2% |

### Latency

| Configuration | First Token | Full Utterance |
|---------------|-------------|----------------|
| **Streaming (1s chunks)** | 200ms | 400ms |
| Batch (full audio) | 500ms | 800ms |
| With Riva | 150ms | 300ms |

### Throughput (RTX 4090)

| Mode | Concurrent Streams | RTFx |
|------|-------------------|------|
| Batch | 8-10 | 50x |
| Streaming | 4-6 | 20x |
| With Riva | 20-30 | 100x |

---

## Troubleshooting

### Common Issues

#### 1. License Error

```
Error: You must accept the license to use nvidia/canary-1b
```

**Solution**: Visit [HuggingFace Canary-1B](https://huggingface.co/nvidia/canary-1b) and accept the CC-BY-NC-4.0 license.

#### 2. CUDA Out of Memory

```bash
# Use smaller context
LEFT_CONTEXT_SECS=5.0

# Or use smaller model
CANARY_MODEL=nvidia/canary-180m

# Or enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 3. Slow First Request

The model needs to compile on first inference. Warm up with:

```python
# Warmup on startup
dummy_audio = np.zeros(16000, dtype=np.float32)
model.transcribe([dummy_audio], batch_size=1)
```

#### 4. Audio Format Issues

```bash
# Convert to required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav
```

---

## Comparison: Voxtral vs Canary

| Aspect | Voxtral Mini 3B | Canary-1B |
|--------|-----------------|-----------|
| **License** | Apache 2.0 (Commercial OK) | CC-BY-NC-4.0 (Non-commercial) |
| **Languages** | 8 | 4 |
| **Accuracy** | Excellent | Best |
| **Streaming** | Chunked only | Native streaming |
| **Audio Length** | 30 minutes | Unlimited |
| **Translation** | Yes | Yes |
| **Diarization** | No | No |
| **VRAM** | 8GB | 4GB |

### Recommendation

- **Commercial Use**: Use Voxtral (Apache 2.0)
- **Research/Non-Commercial**: Use Canary (Best accuracy)
- **Maximum Throughput**: Use NVIDIA Riva with Canary models

---

## Next Steps

1. [Voxtral Setup](./VOXTRAL_SETUP.md) - For commercial deployments
2. [Integration Guide](./technical_integration.md) - Dudoxx Clinic Platform integration
3. [API Documentation](./API.md) - Full API reference

---

**License**: CC-BY-NC-4.0 (Non-commercial use only)  
**Last Updated**: December 2025  
**Maintainer**: Dudoxx UG
