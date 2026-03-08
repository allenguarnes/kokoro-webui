<p align="center">
  <img src="static/favicon.ico" alt="Kokoro WebUI logo" width="64" height="64" />
</p>

# Kokoro WebUI

Local text-to-speech server and browser UI powered by Kokoro ONNX.

Kokoro WebUI provides:
- Native synthesis endpoints (`/api/*`)
- OpenAI-compatible speech endpoints (`/v1/*`)
- Streaming over NDJSON and WebSocket
- Optional backend pitch shifting
- Automatic CPU fallback when GPU runtime is unavailable or incompatible

## Technology Stack

- Python 3.12 / 3.13
- FastAPI + Uvicorn
- `kokoro-onnx`
- NumPy + SoundFile
- Static HTML/CSS/JS frontend

## Project Architecture

```text
Browser UI
  -> FastAPI app
      -> explicit ONNX Runtime session selection
      -> kokoro-onnx runtime (model + voices)
      -> optional ffmpeg pipeline (opus encoding, pitch shifting)
```

Request flow:
1. UI or API client sends synthesis request.
2. Backend validates payload and synthesizes PCM via Kokoro.
3. Optional post-processing is applied (`pitch`, `opus`).
4. Audio is returned directly or streamed chunk-by-chunk.

## Quick Start

### 1) Install dependencies

CPU-only baseline:

```bash
uv sync
```

Recommended Linux/server stack:

```bash
uv sync --extra server
```

Optional NVIDIA GPU runtime:

```bash
uv sync --extra gpu
```

Linux + NVIDIA GPU:

```bash
uv sync --extra server --extra gpu
```

### 2) Add model files

Place these files in `models/`:

```text
models/kokoro-v1.0.onnx
models/voices-v1.0.bin
```

Model source:
[model-files-v1.0 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)

### 3) Run the server

```bash
./scripts/run-server.sh
```

Quiet startup:

```bash
./scripts/run-server.sh --quiet-checks
```

Preflight only:

```bash
./scripts/run-server.sh --check-only
```

Open: `http://127.0.0.1:8000`

`./scripts/run-server.sh` is the Unix-like shell helper. It loads `.env`, prepends `KOKORO_CUDA_LIB_DIR` to `LD_LIBRARY_PATH` when set, prints a prerequisite preflight summary, and then starts the app.

Useful flags:
- `--quiet-checks`: start without printing the preflight summary
- `--check-only`: print the preflight summary and exit without starting the server

## Runtime Requirements

### Required

- Python `3.12` or `3.13`
- `uv`
- Model files listed above

### Optional (feature-dependent)

- `ffmpeg`
  - Required for `opus` output
  - Without it, `wav` still works
- `ffmpeg` with `rubberband` filter
  - Required for non-zero pitch shifting
  - Without it, pitch control is unavailable
- WebSocket runtime (`websockets` or `wsproto`)
  - Required for `/ws/speak-stream`
  - Without it, NDJSON streaming still works
- `onnxruntime-gpu`
  - Required for NVIDIA GPU acceleration
  - Without it, the app runs on CPU
- `uvicorn[standard]` (`uvloop`, `httptools`, and related runtime helpers)
  - Recommended on Linux for lower-overhead request/event-loop handling
  - Without it, the app still works with the pure-Python/cross-platform stack

> [!NOTE]
> `pitch: 0.0` is a no-op and skips pitch-shift post-processing.

> [!WARNING]
> If `ffmpeg` is missing, requests using `format: opus` fail with HTTP 400.

## GPU Runtime

GPU support is optional and best-effort.

Default behavior:
- `KOKORO_PROVIDER=auto`
- try CUDA when `CUDAExecutionProvider` is available
- fall back to `CPUExecutionProvider` if CUDA initialization fails

Recommended NVIDIA setup:
1. Install the base project with `uv sync --extra gpu`
2. Check whether compatible CUDA 12.x user-space libraries are already available:

```bash
nvidia-smi
ldconfig -p | rg 'libcublasLt\.so\.12|libcudart\.so\.12|libcudnn\.so\.9'
```

3. If those CUDA 12.x and cuDNN 9 libraries are already installed on standard linker paths, keep going. You do not need `KOKORO_CUDA_LIB_DIR`.
4. If they are missing, or your machine only has an incompatible CUDA user-space version, install compatible CUDA 12.x user-space libraries and ensure cuDNN 9.x is present.
5. Set `KOKORO_CUDA_LIB_DIR` only when those compatible CUDA libraries live in a private or non-standard location such as `/opt/cuda-12.9/lib64`.

`nvidia-smi` alone is not enough. The driver may report a newer CUDA capability while ONNX Runtime still needs CUDA 12.x user-space libraries to be present on the machine.

Example launch for a private CUDA 12.9 install:

```bash
cat >> .env <<'EOF'
KOKORO_PROVIDER=auto
KOKORO_CUDA_LIB_DIR=/opt/cuda-12.9/lib64
EOF

./scripts/run-server.sh
```

If CUDA is requested explicitly and you want startup to fail instead of falling back:

```bash
KOKORO_PROVIDER=cuda KOKORO_STRICT_PROVIDER=1 ./scripts/run-server.sh
```

## Configuration

The app auto-loads `.env` (if present).

| Variable | Default | Purpose |
| --- | --- | --- |
| `KOKORO_HOST` | `127.0.0.1` | Bind host |
| `KOKORO_PORT` | `8000` | Bind port |
| `KOKORO_RELOAD` | `0` | Development auto-reload |
| `KOKORO_FFMPEG_TIMEOUT_SEC` | `20` | Timeout for ffmpeg/rubberband subprocess work |
| `KOKORO_PROVIDER` | `auto` | Runtime selection: `auto`, `cpu`, or `cuda` |
| `KOKORO_STRICT_PROVIDER` | `0` | Fail startup instead of falling back when requested provider cannot initialize |
| `KOKORO_SYNTH_WORKERS` | `2` on CPU, `1` on auto/cuda | Cap concurrent synthesis jobs in the dedicated worker pool |
| `KOKORO_SYNTH_QUEUE` | `workers * 4` | Queue depth for admitted synthesis jobs waiting behind active workers |
| `KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY` | `0` | Allow `KOKORO_SYNTH_WORKERS > 1` when the runtime is explicitly forced to `cuda` |
| `KOKORO_CUDA_LIB_DIR` | unset | Optional. Use only when compatible CUDA libraries are installed outside the normal dynamic linker search paths |
| `KOKORO_MODEL_PATH` | `models/kokoro-v1.0.onnx` | Override model path |
| `KOKORO_VOICES_PATH` | `models/voices-v1.0.bin` | Override voices path |

On Linux, `uv sync --extra server` installs `uvicorn[standard]`, which gives Uvicorn access to `uvloop` and `httptools` automatically. The app remains cross-platform because those packages are optional rather than required.

Example `.env`:

```dotenv
KOKORO_HOST=127.0.0.1
KOKORO_PORT=8000
KOKORO_RELOAD=0
KOKORO_FFMPEG_TIMEOUT_SEC=20
KOKORO_PROVIDER=auto
KOKORO_STRICT_PROVIDER=0
# Optional synthesis concurrency cap. Defaults to 2 on CPU, 1 on auto/cuda.
# KOKORO_SYNTH_WORKERS=2
# Optional queue depth for admitted synthesis jobs waiting behind active workers.
# KOKORO_SYNTH_QUEUE=8
# Only set when intentionally benchmarking shared-session GPU concurrency.
# KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY=1
# Only set when CUDA 12.x libs are in a non-standard location
# KOKORO_CUDA_LIB_DIR=/opt/cuda-12.9/lib64
```

`KOKORO_SYNTH_WORKERS` is intentionally conservative by default. CPU mode can benefit from a small amount of parallelism, but GPU mode should usually stay at `1` unless you have measured that additional concurrent jobs improve throughput on your hardware. If you expect concurrent CPU-only requests on a higher-core machine, `KOKORO_SYNTH_WORKERS=3` is a reasonable first tuning step to benchmark.

`KOKORO_SYNTH_QUEUE` controls how many additional synthesis jobs can wait behind the active workers before the server starts rejecting overload with `503`. That is useful once the app is serving multiple users instead of only local, single-user traffic.

`/api/health` now also reports queue wait-time metrics so you can tell the difference between “idle,” “busy but healthy,” and “saturating with growing wait time.”

The scheduler is now provider-aware at the policy level:
- CPU mode allows a small shared-runtime worker pool
- GPU-preferred modes still use a shared runtime/session model, so `workers > 1` should be treated as a benchmarked tuning scenario rather than a default recommendation

If you explicitly force `KOKORO_PROVIDER=cuda`, the server now blocks `KOKORO_SYNTH_WORKERS > 1` by default. Use `KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY=1` only when you are intentionally benchmarking shared-session GPU concurrency.

Development reload:

```bash
KOKORO_RELOAD=1 ./scripts/run-server.sh
```

> [!TIP]
> Keep `KOKORO_RELOAD=0` in release deployments.

## Development and Testing

- Development hot reload:

```bash
KOKORO_RELOAD=1 ./scripts/run-server.sh
```

- Run the integration and runtime tests:

```bash
uv sync --dev
uv run python -m unittest tests.test_api tests.test_runtime tests.test_scheduler
```

- Validate runtime quickly:
  - run `./scripts/run-server.sh --check-only`
  - open `/api/health`
  - check `ok`, `active_provider`, `runtime_error`, and `queue`
  - open `/api/capabilities` for voices, formats, and scheduler/runtime details

## API Overview

Full reference: [docs/API.md](docs/API.md)

### Native Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness and queue status |
| `GET` | `/api/capabilities` | Voices, formats, runtime features, and synthesis limits |
| `POST` | `/api/speak` | Single render (`wav` or `opus`) |
| `POST` | `/api/chunk-plan` | Chunk metadata only |
| `POST` | `/api/speak-stream` | NDJSON chunked streaming |
| `WS` | `/ws/speak-stream` | WebSocket chunked streaming |

### OpenAI-Compatible Endpoints

| Method | Path |
| --- | --- |
| `GET` | `/v1/models` |
| `GET` | `/v1/models/kokoro` |
| `POST` | `/v1/audio/speech` |

### Pitch Behavior

- Native API uses `pitch` field (`-6.0` to `+6.0`).
- OpenAI-compatible API uses voice suffixes:
  - `af_heart+2.0`
  - `af_heart-2.0`
  - `af_heart` (implies `0.0`)

## Health and Troubleshooting

Use `/api/health` first when diagnosing runtime issues.

Key fields:
- `ok`
- `missing`
- `active_provider`
- `provider_fallback`
- `provider_error`
- `runtime_error`
- `queue`

Then use `/api/capabilities` for:
- `voices`
- `formats`
- `requested_provider`
- `attempted_providers`
- `available_providers`
- `pitch_shifting`
- `websocket_streaming`
- `scheduler`

Useful queue fields in `/api/health`:
- `queue.queued_jobs`
- `queue.available_slots`
- `queue.rejected_jobs_total`
- `queue.queue_wait_last_ms`
- `queue.queue_wait_avg_ms`
- `queue.queue_wait_max_ms`

If synthesis fails, verify:
1. Model files exist at expected paths
2. Optional runtime tools are installed for requested features (`opus`, pitch, WebSocket, GPU)
3. CUDA user-space and cuDNN versions are compatible with `onnxruntime-gpu` if GPU is enabled
4. Request fields are within supported ranges

## Project Structure

```text
app/
  api.py
  audio.py
  config.py
  main.py
  openai_compat.py
  runtime.py
  scheduler.py
  schemas.py
static/
  index.html
  app.js
  js/
  styles/
docs/
  API.md
models/
  .gitkeep
README.md
```

For licensing details, see `LICENSE` and `THIRD_PARTY_LICENSES.md`.
