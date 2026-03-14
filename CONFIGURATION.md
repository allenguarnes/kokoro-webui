# Configuration Reference

Kokoro WebUI reads configuration from environment variables. Create a `.env` file in the project root or export variables directly.

## Quick Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `KOKORO_HOST` | `127.0.0.1` | Server bind address |
| `KOKORO_PORT` | `8000` | Server port |
| `KOKORO_REQUIRE_AUTH` | `0` | Require API key for all requests |
| `KOKORO_API_KEY` | - | API key when auth is enabled |
| `KOKORO_PROVIDER` | `auto` | Runtime: `auto`, `cpu`, or `cuda` |

## Core Settings

### KOKORO_HOST
**Default:** `127.0.0.1`

Network interface to bind to. Use `0.0.0.0` to accept connections from any interface (not recommended without a reverse proxy).

```bash
KOKORO_HOST=127.0.0.1  # Local only (default)
KOKORO_HOST=0.0.0.0    # All interfaces
```

### KOKORO_PORT
**Default:** `8000`

Port to listen on.

```bash
KOKORO_PORT=8000
```

### KOKORO_ENABLE_WEB_UI
**Default:** `1`

Whether to serve the built-in web interface. Set to `0` for API-only mode.

```bash
KOKORO_ENABLE_WEB_UI=1  # Serve web UI (default)
KOKORO_ENABLE_WEB_UI=0  # API only
```

## Authentication

### KOKORO_REQUIRE_AUTH
**Default:** `0`

Require API key authentication for all non-static routes.

```bash
KOKORO_REQUIRE_AUTH=1
```

### KOKORO_API_KEY
**Required when:** `KOKORO_REQUIRE_AUTH=1`

Secret key for authentication. Used for HTTP bearer auth and issuing short-lived WebSocket session tokens.

```bash
KOKORO_API_KEY=your-secure-random-key
```

> [!WARNING]
> Use a strong, randomly generated key. The built-in web UI stores this only in memory (not localStorage), so users must re-enter it after page reload.

### Auth Rate Limiting

Failed authentication attempts are rate-limited per client.

| Variable | Default | Purpose |
|----------|---------|---------|
| `KOKORO_AUTH_FAILURE_LIMIT` | `5` | Max failures before 429 responses |
| `KOKORO_AUTH_FAILURE_WINDOW_SEC` | `60` | Time window for failure counting |
| `KOKORO_AUTH_FAILURE_MAX_BUCKETS` | `4096` | Max in-memory failure trackers |

## Runtime & Performance

### KOKORO_PROVIDER
**Default:** `auto`

ONNX Runtime provider selection.

- `auto` - Try CUDA, fall back to CPU
- `cpu` - CPU only
- `cuda` - NVIDIA GPU only

```bash
KOKORO_PROVIDER=auto
```

### KOKORO_STRICT_PROVIDER
**Default:** `0`

If enabled, fail startup when the requested provider cannot initialize instead of falling back.

```bash
KOKORO_STRICT_PROVIDER=1  # Fail on CUDA init error
```

### Synthesis Workers

Controls concurrent synthesis jobs.

| Variable | Default | Purpose |
|----------|---------|---------|
| `KOKORO_SYNTH_WORKERS` | `2` (CPU), `1` (GPU) | Max concurrent synthesis jobs |
| `KOKORO_SYNTH_QUEUE` | `workers * 4` | Queue depth for waiting jobs |

The defaults are conservative. CPU mode can benefit from `2-3` workers. GPU mode should usually stay at `1` unless you've benchmarked otherwise.

```bash
KOKORO_SYNTH_WORKERS=2
KOKORO_SYNTH_QUEUE=8
```

### KOKORO_RUNTIME_IDLE_UNLOAD_SEC
**Default:** `0` (disabled)

Automatically unload the ONNX runtime session after N seconds of inactivity. This frees GPU VRAM but adds cold-start latency to the next request.

```bash
KOKORO_RUNTIME_IDLE_UNLOAD_SEC=120  # Unload after 2 minutes idle
```

### KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY
**Default:** `0`

Allow `KOKORO_SYNTH_WORKERS > 1` when explicitly forced to `cuda` provider. This enables GPU concurrency experiments but may cause issues. Only enable if you're benchmarking.

```bash
KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY=1
```

## GPU Configuration

### KOKORO_CUDA_LIB_DIR
**Optional:** Path to CUDA 12.x libraries

Only needed if CUDA libraries are in a non-standard location.

```bash
KOKORO_CUDA_LIB_DIR=/opt/cuda-12.9/lib64
```

> [!NOTE]
> You also need compatible CUDA 12.x user-space libraries installed. `nvidia-smi` showing a CUDA version doesn't mean the runtime libraries are present. Check with: `ldconfig -p | grep 'libcublasLt.so.12\|libcudart.so.12\|libcudnn.so.9'`

## Audio Features

### KOKORO_FORMATS
**Default:** All supported formats

Comma-separated list of enabled output formats. Disables unsupported formats in the web UI.

```bash
KOKORO_FORMATS=wav,pcm        # WAV and PCM only
KOKORO_FORMATS=opus           # Opus only (requires ffmpeg)
KOKORO_FORMATS=wav,pcm,opus   # All (default)
```

### KOKORO_FFMPEG_TIMEOUT_SEC
**Default:** `20`

Timeout for ffmpeg operations (Opus encoding, pitch shifting).

```bash
KOKORO_FFMPEG_TIMEOUT_SEC=20
```

> [!NOTE]
> ffmpeg is required for Opus output and pitch shifting. Install with your package manager: `apt install ffmpeg` (Debian/Ubuntu), `brew install ffmpeg` (macOS), etc.

## Proxy Configuration

### KOKORO_TRUST_PROXY_HEADERS
**Default:** `0`

Trust `X-Forwarded-For` and `X-Real-IP` headers for determining client identity. **Required when running behind a reverse proxy.**

```bash
KOKORO_TRUST_PROXY_HEADERS=1
```

### KOKORO_TRUSTED_PROXY_IPS
**Required when:** `KOKORO_TRUST_PROXY_HEADERS=1`

Comma-separated list of trusted proxy IP addresses or CIDR ranges.

```bash
# Single IP
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1

# Multiple IPs
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1,10.0.0.1

# CIDR range
KOKORO_TRUSTED_PROXY_IPS=10.0.0.0/8

# Combined
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1,10.0.0.0/8
```

> [!IMPORTANT]
> For security, the proxy must **overwrite** `X-Real-IP` with the actual client IP. Passing through client-supplied values allows IP spoofing.
>
> IPv4-mapped IPv6 addresses (e.g., `::ffff:127.0.0.1`) are treated as distinct from their IPv4 counterparts for rate limiting.

## WebSocket Configuration

### KOKORO_WS_AUTH_HANDSHAKE_TIMEOUT_SEC
**Default:** `5`

Timeout (seconds) for WebSocket auth handshake before closing connection.

```bash
KOKORO_WS_AUTH_HANDSHAKE_TIMEOUT_SEC=5
```

### KOKORO_WS_SESSION_TOKEN_TTL_SEC
**Default:** `30`

Lifetime (seconds) for one-time WebSocket session tokens.

```bash
KOKORO_WS_SESSION_TOKEN_TTL_SEC=30
```

### KOKORO_WS_SESSION_TOKEN_MAX_TOKENS
**Default:** `1024`

Maximum number of pending WebSocket session tokens stored in memory.

```bash
KOKORO_WS_SESSION_TOKEN_MAX_TOKENS=1024
```

## CORS Configuration

### KOKORO_ALLOWED_ORIGINS
**Default:** Unset (same-origin only)

Comma-separated list of allowed origins for cross-origin requests. Leave unset to restrict to same-origin only (recommended when auth is enabled).

```bash
KOKORO_ALLOWED_ORIGINS=http://example.com,https://example.com
```

## Model Paths

### KOKORO_MODEL_PATH
**Default:** `models/kokoro-v1.0.onnx`

Path to the Kokoro ONNX model file.

```bash
KOKORO_MODEL_PATH=/custom/path/kokoro-v1.0.onnx
```

### KOKORO_VOICES_PATH
**Default:** `models/voices-v1.0.bin`

Path to the voices binary file.

```bash
KOKORO_VOICES_PATH=/custom/path/voices-v1.0.bin
```

## Development

### KOKORO_RELOAD
**Default:** `0`

Enable auto-reload on code changes (development only).

```bash
KOKORO_RELOAD=1  # Development
KOKORO_RELOAD=0  # Production (default)
```

## Example Configurations

### Minimal Local Setup
```bash
KOKORO_HOST=127.0.0.1
KOKORO_PORT=8000
```

### With Authentication
```bash
KOKORO_REQUIRE_AUTH=1
KOKORO_API_KEY=$(openssl rand -hex 32)
```

### Behind Nginx
```bash
KOKORO_TRUST_PROXY_HEADERS=1
KOKORO_TRUSTED_PROXY_IPS=127.0.0.1
KOKORO_ALLOWED_ORIGINS=https://tts.example.com
```

### GPU Server
```bash
KOKORO_PROVIDER=auto
KOKORO_CUDA_LIB_DIR=/opt/cuda-12.9/lib64
KOKORO_SYNTH_WORKERS=1
```

### CPU-Optimized
```bash
KOKORO_PROVIDER=cpu
KOKORO_SYNTH_WORKERS=3
```
