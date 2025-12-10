# Faster-Whisper Streaming ASR

A lightweight, production-ready streaming ASR server using faster-whisper with AI-powered enhancements.

**GPU Memory**: ~2.5GB (88% less than Voxtral)
**Languages**: EN, FR, DE + 96 more
**License**: MIT (commercial use allowed)

## Features

- **Real-time Streaming**: WebSocket-based streaming transcription
- **Low GPU Memory**: ~2.5GB total (INT8 quantization)
- **Speaker Diarization**: Pyannote 3.1 for speaker identification
- **Entity Recognition**: GLiNER for multilingual NER (persons, companies, locations, dates)
- **Smart Corrections**: 32 vocabulary patterns + fuzzy matching
- **Hotwords Support**: Custom vocabulary ("Dudoxx", "Walid Boudabbous")
- **Multi-language**: English, German, French + 96 more

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
│  │ Silero   │  │ faster-      │  │ GLiNER NER                 │ │
│  │ VAD      │  │ whisper      │  │ Entity                     │ │
│  │ ~100MB   │  │ large-v3     │  │ Recognition                │ │
│  │ ONNX     │  │ INT8: ~2GB   │  │ ~300MB                     │ │
│  └──────────┘  └──────────────┘  └────────────────────────────┘ │
│                                                                 │
│  Total GPU Memory: ~2.5GB                                       │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Memory Comparison

| Solution | GPU Memory | Relative |
|----------|------------|----------|
| Voxtral Mini 3B | ~18-20GB | 100% |
| **faster-whisper** | **~2.5GB** | **12%** |

## Installation

### Prerequisites

- CUDA 12.8+ (compatible with CUDA 13 drivers)
- Python 3.11+
- 24GB GPU recommended (works with 8GB)

### Quick Start

```bash
# 1. Create installation directory
sudo mkdir -p /opt/faster-whisper
sudo chown $USER:$USER /opt/faster-whisper

# 2. Copy files
cp streaming_server.py /opt/faster-whisper/
cp config/.env.example /opt/faster-whisper/.env
cp start.sh /opt/faster-whisper/

# 3. Create virtual environment
cd /opt/faster-whisper
python3.11 -m venv venv
source venv/bin/activate

# 4. Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Install dependencies
pip install faster-whisper fastapi 'uvicorn[standard]' python-multipart websockets httptools silero-vad pyannote.audio gliner transformers

# 6. Configure environment
nano .env
# Set HF_TOKEN for diarization
# Adjust HOTWORDS for your use case

# 7. Run server
./start.sh
```

### Systemd Service (Production)

```bash
# Copy service file
sudo cp systemd/faster-whisper.service /etc/systemd/system/

# Edit paths to match your installation
sudo nano /etc/systemd/system/faster-whisper.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable faster-whisper
sudo systemctl start faster-whisper

# Check status
sudo systemctl status faster-whisper
```

## Configuration

### Environment Variables (.env)

```bash
# Model Configuration
WHISPER_MODEL=large-v3              # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cuda                 # cuda or cpu
WHISPER_COMPUTE_TYPE=int8           # int8 (~2GB) or float16 (~6GB)

# Features
ENABLE_DIARIZATION=true             # Speaker identification
ENABLE_NER=true                     # Entity recognition
ENABLE_SMART_FORMAT=false           # Smart formatting (adds latency)

# HuggingFace Token (required for diarization)
HF_TOKEN=your_token_here

# Custom Vocabulary
HOTWORDS=Dudoxx,Walid Boudabbous,Hamburg

# Audio Processing
CHUNK_DURATION=5.0                  # Chunk size in seconds
```

### Vocabulary Corrections

The server includes 32 built-in corrections for common Whisper errors:

```python
"passion" → "patient"              # Medical context
"didak" → "Dudoxx"                # Company name
"g-docs" → "Dudoxx"               # Common misrecognition
"d-dox" → "Dudoxx"                # Phonetic variation
```

Add custom corrections via environment variable:
```bash
VOCAB_CORRECTIONS='{"mycompany":"MyCompany","produc":"product"}'
```

## API Usage

### WebSocket Streaming

```javascript
const ws = new WebSocket('wss://your-domain/asr?language=auto');

// Connection opened
ws.onopen = () => {
  console.log('Connected to ASR server');
};

// Send PCM audio (Int16, 16kHz, mono)
ws.send(audioBuffer);  // ArrayBuffer

// Receive transcription
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'partial') {
    console.log('Partial:', data.text);
  } else if (data.type === 'final') {
    console.log('Final:', data.text);
    console.log('Entities:', data.entities);
    console.log('Speaker:', data.diarization?.current_speaker);
  }
};
```

### Response Format

```json
{
  "type": "partial",
  "text": "Hello, my name is Dr. Walid Boudabbous from Dudoxx.",
  "full_transcript": "...",
  "timing": {
    "start": 0.0,
    "end": 5.0,
    "duration": 5.0
  },
  "segments": [
    {
      "start": 0.44,
      "end": 4.76,
      "text": "Hello, my name is Dr. Walid Boudabbous from Dudoxx."
    }
  ],
  "words": [
    {"word": "Hello,", "start": 0.44, "end": 0.94, "probability": 0.699},
    {"word": "Dr.", "start": 0.98, "end": 1.66, "probability": 0.858}
  ],
  "language": {
    "detected": "en",
    "probability": 0.919
  },
  "diarization": {
    "current_speaker": "Speaker 1",
    "speakers": [
      {"id": "SPEAKER_00", "label": "Speaker 1"}
    ]
  },
  "entities": {
    "chunk_entities": [
      {"text": "Dr. Walid Boudabbous", "label": "person name", "score": 0.893},
      {"text": "Dudoxx", "label": "organization", "score": 0.687}
    ]
  },
  "processing": {
    "latency_ms": 448.4,
    "rtf": 0.09,
    "vocab_corrections": [
      {"original": "Dudok", "corrected": "Dudoxx"}
    ]
  },
  "stats": {
    "word_count": 9,
    "wpm": 71.7,
    "total_speakers": 1,
    "total_entities": 2
  }
}
```

## Next.js Integration

### 1. Install Dependencies

```bash
npm install --save @types/node
```

### 2. Create ASR Hook

```typescript
// hooks/use-faster-whisper.ts
'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

interface TranscriptionSegment {
  text: string;
  speaker: string;
  timestamp: number;
}

interface Entity {
  text: string;
  label: string;
  score: number;
}

interface TranscriptionResponse {
  type: 'partial' | 'final' | 'config';
  text?: string;
  full_transcript?: string;
  timing?: {
    start: number;
    end: number;
    duration: number;
  };
  diarization?: {
    current_speaker: string;
    speakers: Array<{id: string; label: string}>;
  };
  entities?: {
    chunk_entities: Entity[];
    session_entities: Entity[];
  };
  processing?: {
    vocab_corrections?: Array<{original: string; corrected: string}>;
  };
}

interface UseFasterWhisperOptions {
  serverUrl: string;
  language?: string;
  onTranscript?: (data: TranscriptionResponse) => void;
  onError?: (error: Error) => void;
}

export function useFasterWhisper({
  serverUrl,
  language = 'auto',
  onTranscript,
  onError,
}: UseFasterWhisperOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${serverUrl}/asr?language=${language}`;

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setIsConnected(true);
    };

    wsRef.current.onmessage = (event) => {
      const data: TranscriptionResponse = JSON.parse(event.data);

      if (data.type === 'final') {
        // Store final segments
        if (data.text && data.timing) {
          setSegments(prev => [...prev, {
            text: data.text!,
            speaker: data.diarization?.current_speaker || 'Speaker 1',
            timestamp: data.timing!.start
          }]);
        }

        // Update entities
        if (data.entities?.session_entities) {
          setEntities(data.entities.session_entities);
        }
      }

      onTranscript?.(data);
    };

    wsRef.current.onerror = () => {
      onError?.(new Error('WebSocket connection failed'));
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      setIsRecording(false);
    };
  }, [serverUrl, language, onTranscript, onError]);

  const startRecording = useCallback(async () => {
    if (!wsRef.current || isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(2048, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        wsRef.current.send(pcm.buffer);
      };

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      processorRef.current = processor;

      setIsRecording(true);
      setSegments([]);  // Reset
      setEntities([]);

    } catch (err) {
      onError?.(err as Error);
    }
  }, [isRecording, onError]);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;

    // Send empty buffer to signal end
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(new ArrayBuffer(0));
    }

    // Cleanup audio
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setIsRecording(false);
  }, [isRecording]);

  const disconnect = useCallback(() => {
    stopRecording();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [stopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    startRecording,
    stopRecording,
    isConnected,
    isRecording,
    segments,
    entities,
  };
}
```

### 3. Create Transcription Component

```typescript
// components/faster-whisper-transcriber.tsx
'use client';

import { useState } from 'react';
import { useFasterWhisper } from '@/hooks/use-faster-whisper';

interface Props {
  serverUrl: string;  // e.g., "voxtral.dudoxx.com"
  language?: string;
}

export function FasterWhisperTranscriber({ serverUrl, language = 'auto' }: Props) {
  const [partialText, setPartialText] = useState('');

  const {
    connect,
    disconnect,
    startRecording,
    stopRecording,
    isConnected,
    isRecording,
    segments,
    entities,
  } = useFasterWhisper({
    serverUrl,
    language,
    onTranscript: (data) => {
      if (data.type === 'partial') {
        setPartialText(data.text || '');
      } else if (data.type === 'final') {
        setPartialText('');
      }
    },
    onError: (error) => {
      console.error('ASR Error:', error);
    },
  });

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex gap-2">
        {!isConnected ? (
          <button onClick={connect} className="px-4 py-2 bg-blue-500 text-white rounded">
            Connect
          </button>
        ) : !isRecording ? (
          <button onClick={startRecording} className="px-4 py-2 bg-green-500 text-white rounded">
            Start Recording
          </button>
        ) : (
          <button onClick={stopRecording} className="px-4 py-2 bg-red-500 text-white rounded">
            Stop Recording
          </button>
        )}
      </div>

      {/* Transcript Display */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 min-h-64 max-h-96 overflow-y-auto">
        {segments.map((seg, idx) => (
          <div key={idx} className="mb-3">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              <strong>{seg.speaker}</strong> • {formatTimestamp(seg.timestamp)}
            </div>
            <div className="text-base">{seg.text}</div>
          </div>
        ))}
        {partialText && (
          <div className="mb-3 text-yellow-600 italic">
            <div className="text-sm">Listening...</div>
            <div>{partialText}</div>
          </div>
        )}
      </div>

      {/* Detected Entities */}
      {entities.length > 0 && (
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="font-semibold mb-2">Detected Entities</h3>
          <div className="flex flex-wrap gap-2">
            {entities.map((entity, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-sm"
              >
                {entity.text} <span className="text-xs">({entity.label})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
```

## Features in Detail

### 1. Hotwords Support

Boosts recognition of custom vocabulary:

```bash
HOTWORDS=Dudoxx,Walid Boudabbous,Hamburg,Tomedo,Odoo
```

### 2. Vocabulary Corrections (32 patterns)

Automatic post-processing corrections:
- Medical: "passion" → "patient"
- Company: "Dudok", "G-Docs", "D-DOX" → "Dudoxx"
- Context-aware replacements

### 3. Fuzzy Hotword Matching

Uses Levenshtein distance (70% similarity):
- "Dudoks" → "Dudoxx"
- "Bodabous" → "Boudabbous"

### 4. Entity Recognition (GLiNER)

Detects:
- Person names
- Company/organization names
- Cities, countries
- Dates, times
- Medical terms
- Phone numbers, emails

### 5. Speaker Diarization (Pyannote 3.1)

- Identifies different speakers
- Automatic speaker labeling
- Per-segment speaker attribution

### 6. Modern UI Features

- Timeline format (MM:SS)
- Auto-scroll
- Fade-in animations
- Pulse effect for partial text
- Client-side renderer log
- Entity & correction panels

## Performance

| Metric | Value |
|--------|-------|
| Latency | 200-350ms |
| RTF | 0.04-0.1 (10-25x faster than real-time) |
| Chunk Size | 5 seconds |
| GPU Memory | ~2.5GB |
| Languages | 99 |

## Troubleshooting

### cuDNN Errors

If you see "Unable to load libcudnn":

```bash
# The systemd service includes the fix:
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/faster-whisper/venv/lib/python3.11/site-packages/ctranslate2.libs"
```

### WebSocket Not Working

Check Apache/Nginx proxy configuration:
```apache
ProxyPass /asr ws://localhost:4400/asr
ProxyPassReverse /asr ws://localhost:4400/asr
```

### High GPU Memory

Use smaller model or CPU:
```bash
WHISPER_MODEL=medium      # ~2GB instead of ~3GB
WHISPER_DEVICE=cpu        # Use CPU instead of GPU
```

## Files

| File | Description |
|------|-------------|
| `streaming_server.py` | Main FastAPI server (1600+ lines) |
| `start.sh` | Startup script with LD_LIBRARY_PATH |
| `config/.env.example` | Environment configuration template |
| `config/.env.production` | Production configuration (from deployment) |
| `systemd/faster-whisper.service` | Systemd service with cuDNN fix |
| `requirements.txt` | Python dependencies |

## License

- **faster-whisper**: MIT
- **Whisper models**: MIT (OpenAI)
- **This code**: MIT
- **Pyannote**: MIT
- **GLiNER**: Apache 2.0

## References

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [GLiNER](https://github.com/urchade/GLiNER)
