# Dudoxx Whisper STT - Installation Guide

**Version:** 6.0.0  

## Repo Policy

- Default branch: `main`
- This repository is used as a submodule of `dudoxx-hapifihr` (see `../.gitmodules`)
**Date:** December 12, 2025  
**Author:** Walid Boudabbous, Founder and CTO of Dudoxx UG, CEO of Acceleate.com

---

## 📋 Prerequisites

### **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8GB | 16GB+ |
| **GPU** | None (CPU mode) | NVIDIA GPU (4GB+ VRAM) or Apple Silicon (M1/M2/M3/M4) |
| **Storage** | 10GB free | 50GB+ (for multiple models) |

### **Software Requirements**

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10 - 3.13 | Core runtime |
| **pip** | 21.0+ | Package management |
| **ffmpeg** | 4.0+ | Audio processing |
| **CUDA** | 12.0+ (optional) | NVIDIA GPU acceleration |
| **Git** | 2.0+ | Version control |

### **Operating System Support**

✅ **macOS** (10.15+) - Recommended for Apple Silicon  
✅ **Linux** (Ubuntu 20.04+, Debian 11+, RHEL 8+)  
✅ **Windows** (10/11 with WSL2 recommended)

---

## 🚀 Quick Start Installation

### **Option 1: WhisperLiveKit (Recommended for Development)**

```bash
# 1. Clone the repository
git clone https://github.com/Dudoxx/dudoxx-hapifihr.git
cd dudoxx-hapifihr/ddx-ai-whisper

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install -e .

# 4. Install backend-specific dependencies
# For Apple Silicon (M1/M2/M3/M4):
pip install mlx-whisper

# For NVIDIA GPU:
pip install faster-whisper

# For diarization support:
pip install pyannote.audio diart rx

# 5. Copy and configure environment
cp .env.example .env
nano .env  # Edit configuration

# 6. Start the server
dudoxx-stt --model large-v3-turbo --language en --port 4300

# Or with custom settings:
dudoxx-stt \
  --model large-v3-turbo \
  --language auto \
  --port 4300 \
  --host localhost \
  --backend mlx-whisper \
  --diarization
```

### **Option 2: Faster-Whisper Server (Recommended for Production)**

```bash
# 1. Navigate to faster-whisper directory
cd dudoxx-hapifihr/ddx-ai-whisper/faster-whisper

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA support (NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Or for CPU-only:
pip install torch torchaudio

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp config/.env.example .env
nano .env  # Set HF_TOKEN and other settings

# 6. Start the server
./start.sh
```

---

## 📦 Detailed Installation Steps

### **Step 1: Install System Dependencies**

#### **macOS**

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11 ffmpeg git

# Verify installation
python3 --version  # Should be 3.10+
ffmpeg -version
```

#### **Ubuntu/Debian**

```bash
# Update package list
sudo apt update

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg git build-essential

# Verify installation
python3.11 --version
ffmpeg -version
```

#### **Windows (WSL2)**

```bash
# Install WSL2 (in PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Inside WSL2, install dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg git build-essential
```

### **Step 2: Clone Repository**

```bash
# Clone main repository
git clone https://github.com/Dudoxx/dudoxx-hapifihr.git
cd dudoxx-hapifihr/ddx-ai-whisper

# Verify structure
ls -la
# Expected: whisperlivekit/, faster-whisper/, docs/, .env.example, pyproject.toml
```

### **Step 3: Create Virtual Environment**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### **Step 4: Install Python Dependencies**

#### **Core Installation (WhisperLiveKit)**

```bash
# Install in editable mode with all dependencies
pip install -e .

# Verify installation
dudoxx-stt --help
```

#### **Apple Silicon Optimization (macOS M1/M2/M3/M4)**

```bash
# Install MLX-Whisper for 5-6x speedup
pip install mlx-whisper

# Verify MLX installation
python -c "import mlx_whisper; print('MLX-Whisper installed successfully')"
```

#### **NVIDIA GPU Acceleration**

```bash
# Install CUDA-enabled PyTorch (if not already installed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install faster-whisper with CUDA support
pip install faster-whisper

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### **Speaker Diarization Support**

```bash
# Install diarization dependencies
pip install pyannote.audio diart rx

# For Streaming Sortformer (NVIDIA GPU only)
# Follow instructions in docs/NVIDIA_CANARY_SETUP.md
```

#### **Translation Support**

```bash
# Install NLLB translation model
pip install nllw

# Or install transformers for NLLB
pip install transformers sentencepiece
```

### **Step 5: Download Whisper Models**

Models are automatically downloaded on first use, but you can pre-download:

```bash
# Pre-download models
python -c "import whisper; whisper.load_model('tiny')"
python -c "import whisper; whisper.load_model('base')"
python -c "import whisper; whisper.load_model('small')"
python -c "import whisper; whisper.load_model('medium')"
python -c "import whisper; whisper.load_model('large-v3')"
python -c "import whisper; whisper.load_model('large-v3-turbo')"

# Or set custom cache directory
export HF_HOME=/path/to/models
export TRANSFORMERS_CACHE=/path/to/models
```

**Model Sizes:**

| Model | Download Size | Disk Space |
|-------|--------------|------------|
| tiny | 75 MB | 150 MB |
| base | 142 MB | 290 MB |
| small | 466 MB | 950 MB |
| medium | 1.5 GB | 3 GB |
| large-v3 | 2.9 GB | 6 GB |
| large-v3-turbo | 1.6 GB | 3.2 GB |

### **Step 6: Configuration**

#### **Copy Environment Template**

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit configuration
nano .env  # Or use your preferred editor
```

#### **Essential Configuration (.env)**

```bash
# =============================================================================
# Minimal Configuration (Quick Start)
# =============================================================================

# Server
STT_PORT=4300
STT_HOST=localhost
LOG_LEVEL=INFO

# Model
WHISPER_MODEL=large-v3-turbo
WHISPER_BACKEND=mlx-whisper  # or faster-whisper
DEFAULT_LANGUAGE=auto

# Performance
BACKEND_POLICY=simulstreaming
VAD_ENABLED=true
VAC_ENABLED=true

# Diarization (Optional - requires HuggingFace token)
DIARIZATION_ENABLED=false
# HF_TOKEN=your_huggingface_token_here
```

#### **Get HuggingFace Token (for Diarization)**

1. Go to https://huggingface.co/settings/tokens
2. Create new token with "Read" access
3. Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Add token to `.env`:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### **Step 7: Verify Installation**

```bash
# Test command-line help
dudoxx-stt --help

# Expected output:
# usage: dudoxx-stt [-h] [--model {tiny,base,small,medium,large-v3,large-v3-turbo}]
#                   [--language LANGUAGE] [--port PORT] [--host HOST]
#                   [--backend {auto,mlx-whisper,faster-whisper,whisper}]
#                   [--diarization] ...

# Start test server
dudoxx-stt --model tiny --language en --port 4300 --backend auto

# Expected output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://localhost:4300 (Press CTRL+C to quit)
```

**Test WebSocket Connection:**

```bash
# In another terminal, test WebSocket endpoint
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     http://localhost:4300/ws?language=en

# Expected: HTTP/101 Switching Protocols
```

---

## 🏗️ Production Installation

### **Option A: Systemd Service (Linux)**

```bash
# 1. Install to system location
sudo mkdir -p /opt/dudoxx-whisper
sudo chown $USER:$USER /opt/dudoxx-whisper

# 2. Copy files
cp -r ddx-ai-whisper/* /opt/dudoxx-whisper/
cd /opt/dudoxx-whisper

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
pip install mlx-whisper  # or faster-whisper

# 4. Copy systemd service
sudo cp faster-whisper/systemd/faster-whisper.service /etc/systemd/system/dudoxx-whisper.service

# 5. Edit service file
sudo nano /etc/systemd/system/dudoxx-whisper.service

# Update paths:
# WorkingDirectory=/opt/dudoxx-whisper
# ExecStart=/opt/dudoxx-whisper/venv/bin/python /opt/dudoxx-whisper/faster-whisper/streaming_server.py

# 6. Configure environment
cp .env.example /opt/dudoxx-whisper/.env
nano /opt/dudoxx-whisper/.env

# 7. Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable dudoxx-whisper
sudo systemctl start dudoxx-whisper

# 8. Check status
sudo systemctl status dudoxx-whisper

# 9. View logs
sudo journalctl -u dudoxx-whisper -f
```

**Systemd Service File Example:**

```ini
[Unit]
Description=Dudoxx Whisper STT Service
After=network.target

[Service]
Type=simple
User=dudoxx
WorkingDirectory=/opt/dudoxx-whisper
Environment="PATH=/opt/dudoxx-whisper/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/opt/dudoxx-whisper"
Environment="LD_LIBRARY_PATH=/opt/dudoxx-whisper/venv/lib/python3.11/site-packages/ctranslate2.libs"
ExecStart=/opt/dudoxx-whisper/venv/bin/dudoxx-stt \
  --model large-v3-turbo \
  --language auto \
  --port 4300 \
  --host 0.0.0.0 \
  --backend mlx-whisper
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dudoxx-whisper

[Install]
WantedBy=multi-user.target
```

### **Option B: Docker Deployment**

#### **CPU-Only Container**

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t dudoxx-whisper:latest .

# Run container
docker run -d \
  --name dudoxx-whisper \
  -p 4300:8000 \
  -e WHISPER_MODEL=large-v3-turbo \
  -e DEFAULT_LANGUAGE=auto \
  -e LOG_LEVEL=INFO \
  dudoxx-whisper:latest

# Check logs
docker logs -f dudoxx-whisper
```

#### **GPU-Enabled Container (NVIDIA)**

```bash
# Build GPU image
docker build -t dudoxx-whisper-gpu:latest .

# Run with GPU support
docker run -d \
  --name dudoxx-whisper-gpu \
  --gpus all \
  -p 4300:8000 \
  -e WHISPER_MODEL=large-v3-turbo \
  -e WHISPER_BACKEND=faster-whisper \
  -e DEFAULT_LANGUAGE=auto \
  -e LOG_LEVEL=INFO \
  dudoxx-whisper-gpu:latest

# Verify GPU usage
docker exec dudoxx-whisper-gpu nvidia-smi
```

#### **Docker Compose (with PostgreSQL for transcripts)**

```yaml
# docker-compose.yml
version: '3.8'

services:
  dudoxx-whisper:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    container_name: dudoxx-whisper
    ports:
      - "4300:8000"
    environment:
      - WHISPER_MODEL=large-v3-turbo
      - DEFAULT_LANGUAGE=auto
      - LOG_LEVEL=INFO
      - STT_PORT=8000
      - DIARIZATION_ENABLED=false
    volumes:
      - ./models:/root/.cache/whisper
      - ./.env:/app/.env
    restart: unless-stopped

  # Optional: PostgreSQL for transcript storage
  postgres:
    image: postgres:15-alpine
    container_name: dudoxx-whisper-db
    environment:
      - POSTGRES_DB=whisper_transcripts
      - POSTGRES_USER=whisper
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - whisper-db:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  whisper-db:
```

**Run with Docker Compose:**

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f dudoxx-whisper

# Stop services
docker-compose down
```

### **Option C: Reverse Proxy (Apache/Nginx)**

#### **Apache Configuration**

```apache
# /etc/apache2/sites-available/dudoxx-whisper.conf

<VirtualHost *:443>
    ServerName whisper.dudoxx.com

    # SSL Configuration
    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/dudoxx-whisper.crt
    SSLCertificateKeyFile /etc/ssl/private/dudoxx-whisper.key

    # WebSocket Proxy
    ProxyPreserveHost On
    ProxyPass /ws ws://localhost:4300/ws
    ProxyPassReverse /ws ws://localhost:4300/ws

    # HTTP Proxy (for health checks)
    ProxyPass / http://localhost:4300/
    ProxyPassReverse / http://localhost:4300/

    # WebSocket Upgrade Headers
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://localhost:4300/$1" [P,L]

    # Logging
    ErrorLog ${APACHE_LOG_DIR}/whisper-error.log
    CustomLog ${APACHE_LOG_DIR}/whisper-access.log combined
</VirtualHost>
```

**Enable Apache Modules:**

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite ssl
sudo a2ensite dudoxx-whisper
sudo systemctl restart apache2
```

#### **Nginx Configuration**

```nginx
# /etc/nginx/sites-available/dudoxx-whisper

upstream whisper_backend {
    server localhost:4300;
}

server {
    listen 443 ssl http2;
    server_name whisper.dudoxx.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/dudoxx-whisper.crt;
    ssl_certificate_key /etc/ssl/private/dudoxx-whisper.key;

    # WebSocket Endpoint
    location /ws {
        proxy_pass http://whisper_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    # HTTP Endpoints
    location / {
        proxy_pass http://whisper_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Logging
    access_log /var/log/nginx/whisper-access.log;
    error_log /var/log/nginx/whisper-error.log;
}
```

**Enable Site:**

```bash
sudo ln -s /etc/nginx/sites-available/dudoxx-whisper /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🧪 Testing Installation

### **1. Basic Functionality Test**

```bash
# Start server
dudoxx-stt --model tiny --language en --port 4300

# In another terminal, test with curl
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: test123" \
     http://localhost:4300/ws?language=en

# Expected: HTTP/1.1 101 Switching Protocols
```

### **2. Web Interface Test**

```bash
# Start server
dudoxx-stt --model base --language en --port 4300

# Open browser
open http://localhost:4300  # macOS
xdg-open http://localhost:4300  # Linux

# Expected: Web demo interface with "Start Recording" button
```

### **3. Model Loading Test**

```bash
# Test each backend
dudoxx-stt --model tiny --backend mlx-whisper --port 4300  # Apple Silicon
dudoxx-stt --model tiny --backend faster-whisper --port 4300  # NVIDIA/CPU
dudoxx-stt --model tiny --backend whisper --port 4300  # Standard

# Check logs for successful model loading
# Expected: "Model loaded successfully" or similar message
```

### **4. Performance Benchmark**

```bash
# Install benchmark tool
pip install webrtcvad

# Run performance test
python scripts/benchmark_latency.py \
  --model large-v3-turbo \
  --backend mlx-whisper \
  --language en \
  --audio-file test_audio.wav

# Expected output:
# Model: large-v3-turbo
# Backend: mlx-whisper
# Average latency: 0.45s
# RTF (Real-time Factor): 0.15
```

---

## 🐛 Common Installation Issues

### **Issue 1: `ModuleNotFoundError: No module named 'whisper'`**

**Solution:**

```bash
# Reinstall dependencies
pip install --force-reinstall -e .
```

### **Issue 2: `ImportError: No module named 'mlx_whisper'`**

**Solution:**

```bash
# Install MLX-Whisper (macOS only)
pip install mlx-whisper

# Verify installation
python -c "import mlx_whisper; print('OK')"
```

### **Issue 3: CUDA / GPU Not Detected**

**Symptoms:** `CUDA not available` despite having NVIDIA GPU

**Solution:**

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### **Issue 4: `ffmpeg` Not Found**

**Symptoms:** `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Solution:**

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### **Issue 5: Permission Denied (Port 4300)**

**Symptoms:** `PermissionError: [Errno 13] Permission denied`

**Solution:**

```bash
# Use non-privileged port (>1024)
dudoxx-stt --port 8300

# Or run with sudo (not recommended)
sudo dudoxx-stt --port 4300
```

### **Issue 6: HuggingFace Token Invalid**

**Symptoms:** `401 Unauthorized` when loading diarization models

**Solution:**

```bash
# 1. Generate new token at https://huggingface.co/settings/tokens
# 2. Accept terms at https://huggingface.co/pyannote/segmentation-3.0
# 3. Update .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 4. Test token
python -c "from huggingface_hub import login; login('hf_xxx')"
```

---

## 📦 Uninstallation

### **Remove Virtual Environment**

```bash
# Deactivate virtual environment
deactivate

# Remove directory
rm -rf venv
```

### **Remove Systemd Service**

```bash
# Stop service
sudo systemctl stop dudoxx-whisper
sudo systemctl disable dudoxx-whisper

# Remove service file
sudo rm /etc/systemd/system/dudoxx-whisper.service
sudo systemctl daemon-reload
```

### **Remove Docker Containers**

```bash
# Stop and remove containers
docker-compose down -v

# Remove images
docker rmi dudoxx-whisper:latest
```

---

## 🎯 Next Steps

After successful installation:

1. **Read IMPORTANT.md** - Critical paths and performance optimization
2. **Read CLAUDE.md** - Complete system documentation and API reference
3. **Configure Production Settings** - SSL, authentication, monitoring
4. **Integrate with Dudoxx Platform** - Connect to ddx-api and ddx-web
5. **Test with Real Audio** - Medical consultations, video calls
6. **Monitor Performance** - Latency, memory usage, transcription quality

---

## 📞 Support

**Company:** Dudoxx UG, Hamburg, Germany  
**Email:** support@dudoxx.com  
**Documentation:** https://github.com/Dudoxx/dudoxx-hapifihr/tree/main/ddx-ai-whisper/docs

For installation issues, please check:
- **docs/troubleshooting.md** - Common issues and solutions
- **GitHub Issues** - https://github.com/QuentinFuxa/WhisperLiveKit/issues
- **Dudoxx Support** - support@dudoxx.com

---

**Installation Complete!** 🎉

Your Dudoxx Whisper STT server is now ready to provide real-time, HIPAA-compliant speech-to-text transcription for healthcare applications.
