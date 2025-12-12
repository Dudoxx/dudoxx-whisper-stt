# ENV_VARIABLES.md - Dudoxx Whisper STT Environment Configuration

**Version:** 6.0.0 | **Date:** December 12, 2025  
**Author:** Walid Boudabbous, Founder and CTO of Dudoxx UG, CEO of Acceleate.com

---

## Quick Reference

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

---

## Required Variables

### HuggingFace Configuration (For Diarization)

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for pyannote models | `hf_your_token_here` |

Get token from: https://huggingface.co/settings/tokens

### Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `STT_PORT` | Server port | `4300` |
| `STT_HOST` | Server bind address | `localhost` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Whisper Model Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `WHISPER_MODEL` | Model size | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_BACKEND` | Inference backend | `mlx-whisper` | `auto`, `mlx-whisper`, `faster-whisper`, `whisper` |
| `DEFAULT_LANGUAGE` | Source language | `auto` | `auto`, `en`, `de`, `fr`, etc. |

### Model Recommendations

| Use Case | Model | Backend |
|----------|-------|---------|
| **Development/Testing** | `base` | `mlx-whisper` |
| **Production (Mac)** | `large-v3-turbo` | `mlx-whisper` |
| **Production (Linux/GPU)** | `large-v3-turbo` | `faster-whisper` |
| **Low Memory** | `small` | `faster-whisper` |

---

## Streaming Configuration

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `BACKEND_POLICY` | Streaming strategy | `simulstreaming` | `simulstreaming` |
| `MIN_CHUNK_SIZE` | Minimum processing unit (seconds) | `0.1` | `0.1` |
| `BUFFER_TRIMMING` | Buffer management | `segment` | `segment` |
| `BUFFER_TRIMMING_SEC` | Buffer threshold (seconds) | `15` | `15` |

---

## Voice Activity Detection (VAD)

| Variable | Description | Default |
|----------|-------------|---------|
| `VAD_ENABLED` | Enable silence filtering | `true` |
| `VAC_ENABLED` | Voice Activity Controller | `true` |
| `VAC_CHUNK_SIZE` | VAC chunk size (seconds) | `0.04` |

---

## Speaker Diarization

| Variable | Description | Default |
|----------|-------------|---------|
| `DIARIZATION_ENABLED` | Enable speaker identification | `false` |
| `DIARIZATION_BACKEND` | Backend engine | `diart` |
| `SEGMENTATION_MODEL` | Pyannote segmentation model | `pyannote/segmentation-3.0` |
| `EMBEDDING_MODEL` | Pyannote embedding model | `pyannote/embedding` |

### Diarization Backends

| Backend | Hardware | Quality | Setup |
|---------|----------|---------|-------|
| `diart` | CPU/GPU | Good | Requires HF_TOKEN |
| `sortformer` | NVIDIA GPU only | Best | Requires NeMo toolkit |

---

## Translation Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TARGET_LANGUAGE` | Translation target (empty = disabled) | `` |
| `DIRECT_ENGLISH_TRANSLATION` | Use Whisper's built-in translation | `false` |
| `NLLB_BACKEND` | Translation backend | `transformers` |
| `NLLB_SIZE` | NLLB model size | `600M` |

---

## Advanced Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FRAME_THRESHOLD` | Attention-guided decoding threshold | `25` |
| `AUDIO_MAX_LEN` | Maximum audio buffer (seconds) | `30.0` |
| `AUDIO_MIN_LEN` | Minimum audio before processing | `0.0` |
| `BEAMS` | Beam search beams (1 = greedy) | `1` |
| `DECODER_TYPE` | Decoder type | `greedy` |
| `INIT_PROMPT` | Initial prompt for vocabulary | `` |
| `STATIC_INIT_PROMPT` | Persistent prompt | `` |

---

## SSL Configuration (Production)

| Variable | Description | Default |
|----------|-------------|---------|
| `SSL_CERTFILE` | SSL certificate path | `` |
| `SSL_KEYFILE` | SSL key path | `` |
| `FORWARDED_ALLOW_IPS` | Reverse proxy IPs | `` |

---

## Audio Input Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PCM_INPUT` | Use raw PCM (AudioWorklet) | `false` |

---

## Example .env

```bash
# =============================================================================
# HuggingFace Configuration (Required for Diarization)
# =============================================================================
HF_TOKEN=hf_your_token_here

# =============================================================================
# Server Configuration
# =============================================================================
STT_PORT=4300
STT_HOST=localhost
LOG_LEVEL=INFO

# =============================================================================
# Whisper Model Configuration
# =============================================================================
WHISPER_MODEL=large-v3-turbo
WHISPER_BACKEND=mlx-whisper
DEFAULT_LANGUAGE=auto

# =============================================================================
# Streaming Configuration
# =============================================================================
BACKEND_POLICY=simulstreaming
MIN_CHUNK_SIZE=0.1
BUFFER_TRIMMING=segment
BUFFER_TRIMMING_SEC=15

# =============================================================================
# Voice Activity Detection
# =============================================================================
VAD_ENABLED=true
VAC_ENABLED=true
VAC_CHUNK_SIZE=0.04

# =============================================================================
# Speaker Diarization (Optional)
# =============================================================================
DIARIZATION_ENABLED=false
DIARIZATION_BACKEND=diart
SEGMENTATION_MODEL=pyannote/segmentation-3.0
EMBEDDING_MODEL=pyannote/embedding

# =============================================================================
# Translation (Optional)
# =============================================================================
TARGET_LANGUAGE=
DIRECT_ENGLISH_TRANSLATION=false
NLLB_BACKEND=transformers
NLLB_SIZE=600M

# =============================================================================
# Advanced Configuration
# =============================================================================
FRAME_THRESHOLD=25
AUDIO_MAX_LEN=30.0
AUDIO_MIN_LEN=0.0
BEAMS=1
DECODER_TYPE=greedy
INIT_PROMPT=
STATIC_INIT_PROMPT=

# =============================================================================
# SSL Configuration (Production)
# =============================================================================
SSL_CERTFILE=
SSL_KEYFILE=
FORWARDED_ALLOW_IPS=

# =============================================================================
# Audio Input
# =============================================================================
PCM_INPUT=false
```

---

## Faster-Whisper Server Configuration

For the alternative `faster-whisper/` server, create `config/.env`:

```bash
# =============================================================================
# Faster-Whisper Server Configuration
# =============================================================================
HF_TOKEN=hf_your_token_here
HOST=0.0.0.0
PORT=4300
LOG_LEVEL=INFO

# Model Configuration
WHISPER_MODEL=large-v3-turbo
WHISPER_COMPUTE_TYPE=float16  # float16 (GPU) or int8 (CPU)
WHISPER_DEVICE=cuda           # cuda, cpu, or auto

# Hotwords (domain-specific vocabulary)
HOTWORDS=Dudoxx,FHIR,Tomedo,Hamburg
```

---

## Environment-Specific Configurations

### Development (Local Mac)

```bash
WHISPER_MODEL=base
WHISPER_BACKEND=mlx-whisper
DEFAULT_LANGUAGE=en
STT_PORT=4300
LOG_LEVEL=DEBUG
DIARIZATION_ENABLED=false
```

### Production (Linux/GPU)

```bash
WHISPER_MODEL=large-v3-turbo
WHISPER_BACKEND=faster-whisper
DEFAULT_LANGUAGE=auto
STT_PORT=4300
LOG_LEVEL=INFO
DIARIZATION_ENABLED=true
DIARIZATION_BACKEND=sortformer
SSL_CERTFILE=/etc/ssl/certs/whisper.crt
SSL_KEYFILE=/etc/ssl/private/whisper.key
```

### Production (Mac M-Series)

```bash
WHISPER_MODEL=large-v3-turbo
WHISPER_BACKEND=mlx-whisper
DEFAULT_LANGUAGE=auto
STT_PORT=4300
LOG_LEVEL=INFO
DIARIZATION_ENABLED=true
DIARIZATION_BACKEND=diart
HF_TOKEN=hf_your_token_here
```

---

## Supported Languages (99 total)

Common languages:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `fr` | French |
| `de` | German | `es` | Spanish |
| `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish |
| `ru` | Russian | `ja` | Japanese |
| `zh` | Chinese | `ko` | Korean |
| `ar` | Arabic | `hi` | Hindi |

Use `auto` for automatic language detection.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "HF_TOKEN required" | Missing token | Set HF_TOKEN in .env |
| "Model not found" | Download failed | Check network, pre-download |
| "CUDA out of memory" | Insufficient VRAM | Use smaller model or `int8` |
| "MLX not available" | Not on Apple Silicon | Use `faster-whisper` |
| "Diarization failed" | Missing model access | Accept terms on HuggingFace |

---

## Related Documentation

- `IMPORTANT.md` - Critical paths and patterns
- `CLAUDE.md` - Complete operational manual
- `docs/API.md` - WebSocket API reference
- `docs/supported_languages.md` - All 99 languages
- `docs/troubleshooting.md` - Common issues
