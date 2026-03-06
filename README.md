# Kokoro WebUI

Small local WebUI for Kokoro TTS with automatic playback after synthesis.

## Stack

- FastAPI backend
- Static HTML/CSS/JS frontend
- `kokoro-onnx` runtime for synthesis

## Files

- `app/main.py`: API and static file server
- `static/`: frontend assets
- `models/`: put Kokoro model files here

## Prerequisites and Setup

### Required

- Python 3.12 or 3.13
- `uv` package manager
- Kokoro model assets:
  - `models/kokoro-v1.0.onnx`
  - `models/voices-v1.0.bin`

### Optional Runtime Tools

- `ffmpeg`
  - Needed for `opus` output encoding.
  - Without it:
    - `wav` synthesis still works.
    - `opus` requests fail with HTTP 400 (`ffmpeg is required for Opus output but is not installed.`).
- `ffmpeg` build with `rubberband` filter
  - Needed for backend pitch shifting (`pitch != 0`).
  - Without it:
    - Pitch shifting is reported unavailable in `/api/health` (`pitch_shifting: false`).
    - Web UI pitch control is disabled automatically.
    - Requests with non-zero pitch fail with HTTP 400.
    - Pitch `0` remains a no-op and works normally.
- WebSocket transport runtime (`websockets` or `wsproto`)
  - Needed for `/ws/speak-stream`.
  - Without it:
    - NDJSON streaming (`/api/speak-stream`) and non-streaming endpoints continue to work.
    - `/api/health` reports `websocket_streaming: false`, and the UI disables WebSocket transport.

### Setup Steps

1. Install dependencies:

```bash
uv sync
```

2. Download the Kokoro ONNX model and voices bundle, then place them here:

   Source: [model-files-v1.0 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)

```text
models/kokoro-v1.0.onnx
models/voices-v1.0.bin
```

3. Start the app:

```bash
uv run python -m app.main
```

4. Open `http://127.0.0.1:8000`.

### Network Binding

Server bind settings come from environment variables:

- `KOKORO_HOST` (default: `127.0.0.1`)
- `KOKORO_PORT` (default: `8000`)
- `KOKORO_RELOAD` (default: disabled; enable only for development with `1/true/yes/on`)

The app also auto-loads a local `.env` file (if present) before reading these values.

Examples:

```bash
# Localhost only (default behavior)
KOKORO_HOST=127.0.0.1 KOKORO_PORT=8000 uv run python -m app.main

# Bind all interfaces (use firewall restrictions in untrusted networks)
KOKORO_HOST=0.0.0.0 KOKORO_PORT=8000 uv run python -m app.main

# Bind directly to your Tailscale interface IP
KOKORO_HOST=100.x.y.z KOKORO_PORT=8000 uv run python -m app.main

# Development hot-reload only
KOKORO_RELOAD=1 uv run python -m app.main
```

Example `.env`:

```dotenv
KOKORO_HOST=100.x.y.z
KOKORO_PORT=8000
KOKORO_RELOAD=0
```

## Notes

- The backend uses `Kokoro(model_path, voices_path)` and `tts.create(text, voice=..., speed=..., lang=...)`, matching the `kokoro-onnx` usage example from the project docs.
- You can override model file locations with `KOKORO_MODEL_PATH` and `KOKORO_VOICES_PATH`.
- If the health badge shows missing runtime files, install dependencies and add the model assets before generating audio.
- The Web UI pitch control is processed in the backend via `ffmpeg`'s `rubberband` filter for the app's `/api/*` synthesis routes.
- The OpenAI-compatible `/v1/audio/speech` endpoint does not expose pitch yet.

## API

Full reference: [docs/API.md](docs/API.md)

- `POST /api/speak`: single audio output (`wav` or `opus`)
- `POST /api/chunk-plan`: metadata-only chunk plan for debugging/planning
- `POST /api/speak-stream`: NDJSON streaming chunks with audio payloads
- `WS /ws/speak-stream`: websocket streaming alternative to NDJSON
- `POST /v1/audio/speech`: OpenAI-compatible speech endpoint
- `GET /v1/models`: OpenAI-compatible model list
- `GET /v1/models/kokoro`: OpenAI-compatible model metadata

### Chunk Plan (metadata-only)

```json
{
  "text": "Long text here.",
  "target_chunk_chars": 360,
  "include_text": false
}
```

### Streaming Request Fields

```json
{
  "text": "Long text here.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": -2.0,
  "lang": "en-us",
  "format": "opus",
  "opus_bitrate": "32k",
  "target_chunk_chars": 360
}
```

- `opus_bitrate` supports: `16k`, `24k`, `32k`, `48k`
- `pitch` is expressed in semitones and currently supports `-6.0` through `+6.0`
- `pitch: 0.0` is a no-op and skips backend pitch-shift processing entirely
- Sentence boundaries are preserved when creating chunks; `target_chunk_chars` is approximate, not a hard cut.
- `GET /api/health` returns `websocket_streaming` and `pitch_shifting` to indicate which runtime features are available.

### OpenAI-Compatible Speech

```json
{
  "model": "kokoro",
  "input": "The future sounds calmer when it is rendered locally.",
  "voice": "af_heart",
  "response_format": "wav",
  "speed": 1.0
}
```

- `model` is accepted for compatibility and is currently ignored by the speech endpoint
- `voice` should be a Kokoro voice id such as `af_heart`, or a suffixed form such as `af_heart+2.0` or `af_heart-2.0`
- Omitting the suffix implies `0.0` pitch shift, so `af_heart` uses the raw voice with no post-processing
- Supported `response_format` values are currently `wav` and `opus`
- `speed` is accepted in the OpenAI-style request, but Kokoro currently supports `0.5` through `1.8`

## License

- Project license: Apache-2.0 ([LICENSE](LICENSE))
- Third-party and model/runtime licensing notes: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
