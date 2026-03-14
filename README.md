<p align="center">
  <img src="static/favicon.ico" alt="Kokoro WebUI logo" width="64" height="64" />
</p>

# Kokoro WebUI

A self-hosted text-to-speech server and web interface powered by [Kokoro](https://github.com/hexgrad/kokoro).

Kokoro is a lightweight, fast TTS model that runs entirely on your machine. No cloud services, no API keys, no data leaving your computer.

## Features

- **Browser-based UI** - Type text, click a button, get audio
- **Multiple voices** - Choose from various voice options
- **Audio formats** - WAV, PCM, or compressed Opus output
- **Pitch control** - Adjust voice pitch up or down
- **Streaming** - Get audio chunks as they're generated (no waiting)
- **OpenAI-compatible API** - Drop-in replacement for `/v1/audio/speech`
- **GPU support** - CUDA acceleration when available (falls back to CPU)

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/yourusername/kokoro-webui.git
cd kokoro-webui

# Install dependencies (pick one):
uv sync                              # CPU only
uv sync --extra server               # Linux, optimized
uv sync --extra gpu                  # NVIDIA GPU support
uv sync --extra server --extra gpu   # Linux + NVIDIA
```

### 2. Download models

Place these files in the `models/` folder:

```
models/kokoro-v1.0.onnx
models/voices-v1.0.bin
```

Download from: [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)

### 3. Run

```bash
./scripts/run-server.sh
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000)

That's it. You now have a local TTS server running.

## Basic Configuration

Create a `.env` file to customize behavior:

```bash
# Change port
KOKORO_PORT=8080

# Require authentication
KOKORO_REQUIRE_AUTH=1
KOKORO_API_KEY=your-secret-key

# Enable GPU (auto-detects if available)
KOKORO_PROVIDER=auto
```

See [CONFIGURATION.md](CONFIGURATION.md) for all options.

## API Usage

### Native API

```bash
# Simple synthesis
curl -X POST http://localhost:8000/api/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_heart"}' \
  -o output.wav

# With options
curl -X POST http://localhost:8000/api/speak \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_heart",
    "format": "opus",
    "pitch": 1.5
  }' \
  -o output.opus
```

### OpenAI-Compatible API

```bash
# Works with any OpenAI-compatible client
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer any-key-works" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello world",
    "voice": "af_heart"
  }' \
  -o output.wav
```

### WebSocket Streaming

For real-time applications, connect to `/ws/speak-stream`:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/speak-stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    text: "Hello world",
    voice: "af_heart"
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    // Play audio chunk
    const audio = new Audio('data:audio/wav;base64,' + data.audio_base64);
    audio.play();
  }
};
```

## Behind a Reverse Proxy

If you want to run behind nginx, see the [Reverse Proxy](#reverse-proxy) section below. You'll need to set:

```bash
KOKORO_TRUST_PROXY_HEADERS=1
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1
```

## Documentation

- [Full configuration reference](CONFIGURATION.md) - All environment variables
- [API reference](docs/API.md) - Detailed endpoint documentation
- [Tuning guide](docs/TUNING.md) - Performance optimization
- [Contributing](CONTRIBUTING.md) - Development setup

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) (package manager)
- Model files (see Quick Start)
- Optional: ffmpeg (for Opus output and pitch shifting)
- Optional: NVIDIA GPU with CUDA 12.x libraries (for GPU acceleration)

## Troubleshooting

Check `/api/health` for runtime status:

```bash
curl http://localhost:8000/api/health
```

Common issues:

- **"Model file not found"** - Download models to `models/` folder
- **"ffmpeg required"** - Install ffmpeg for Opus/pitch features
- **GPU not detected** - Check CUDA 12.x libraries are installed

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

---

**Detailed documentation below...**

## Reverse Proxy

You can run Kokoro WebUI behind nginx or another reverse proxy.

### Environment

Add to your `.env`:

```bash
KOKORO_TRUST_PROXY_HEADERS=1
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name kokoro.local;

    location /static/ {
        alias /path/to/kokoro-webui/static/;
        expires 1d;
    }

    location /ws/speak-stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
    }

    location /api/health/stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Replace `/path/to/kokoro-webui/static/` with your actual project path.

### Setup Steps

```bash
# Add to /etc/hosts
echo "127.0.0.1 kokoro.local" | sudo tee -a /etc/hosts

# Enable nginx site
sudo ln -s /path/to/nginx-config.conf /etc/nginx/sites-enabled/kokoro
sudo nginx -t
sudo nginx -s reload
```
