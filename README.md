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

## Technology Stack

- Python 3.12 / 3.13
- FastAPI + Uvicorn
- `kokoro-onnx`
- NumPy + SoundFile
- Static HTML/CSS/JS frontend

## Project Architecture

```text
Browser UI (static/index.html + static/app.js)
  -> FastAPI app (app/main.py)
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

```bash
uv sync
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
uv run python -m app.main
```

Open: `http://127.0.0.1:8000`

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

> [!NOTE]
> `pitch: 0.0` is a no-op and skips pitch-shift post-processing.

> [!WARNING]
> If `ffmpeg` is missing, requests using `format: opus` fail with HTTP 400.

## Configuration

The app auto-loads `.env` (if present).

| Variable | Default | Purpose |
| --- | --- | --- |
| `KOKORO_HOST` | `127.0.0.1` | Bind host |
| `KOKORO_PORT` | `8000` | Bind port |
| `KOKORO_RELOAD` | `0` | Development auto-reload |
| `KOKORO_FFMPEG_TIMEOUT_SEC` | `20` | Timeout for ffmpeg/rubberband subprocess work |
| `KOKORO_MODEL_PATH` | `models/kokoro-v1.0.onnx` | Override model path |
| `KOKORO_VOICES_PATH` | `models/voices-v1.0.bin` | Override voices path |

Example `.env`:

```dotenv
KOKORO_HOST=127.0.0.1
KOKORO_PORT=8000
KOKORO_RELOAD=0
KOKORO_FFMPEG_TIMEOUT_SEC=20
```

Development reload:

```bash
KOKORO_RELOAD=1 uv run python -m app.main
```

> [!TIP]
> Keep `KOKORO_RELOAD=0` in release deployments.

## Development and Testing

- Development hot reload:

```bash
KOKORO_RELOAD=1 uv run python -m app.main
```

- Run the integration tests:

```bash
uv sync --dev
uv run python -m unittest tests.test_api
```

- Validate runtime quickly:
  - open `/api/health`
  - check `ok`, `missing`, `pitch_shifting`, `websocket_streaming`

> [!NOTE]
> This repository currently does not include an automated test suite.

## API Overview

Full reference: [docs/API.md](docs/API.md)

### Native Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Runtime capability and readiness |
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
- `pitch_shifting`
- `websocket_streaming`

If synthesis fails, verify:
1. Model files exist at expected paths
2. Optional runtime tools are installed for requested features (`opus`, pitch, WebSocket)
3. Request fields are within supported ranges

## Project Structure

```text
app/
  main.py
static/
  index.html
  styles.css
  app.js
docs/
  API.md
models/
  .gitkeep
README.md
```

For licensing details, see `LICENSE` and `THIRD_PARTY_LICENSES.md`.
