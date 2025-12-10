#!/bin/bash
cd /opt/faster-whisper
source .env
export WHISPER_MODEL WHISPER_DEVICE WHISPER_COMPUTE_TYPE ENABLE_DIARIZATION HF_TOKEN

# CRITICAL: Add ctranslate2's bundled cuDNN to library path
export LD_LIBRARY_PATH="/opt/faster-whisper/venv/lib/python3.11/site-packages/ctranslate2.libs:$LD_LIBRARY_PATH"

exec /opt/faster-whisper/venv/bin/python streaming_server.py --port 4400
