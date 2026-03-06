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

## Setup

1. Use Python 3.12 or 3.13.
2. Install dependencies:

```bash
uv sync
```

3. Download the Kokoro ONNX model and voices bundle, then place them here:

```text
models/kokoro-v1.0.onnx
models/voices-v1.0.bin
```

4. Start the app:

```bash
uv run uvicorn app.main:app --reload
```

5. Open `http://127.0.0.1:8000`.

## Notes

- The backend uses `Kokoro(model_path, voices_path)` and `tts.create(text, voice=..., speed=..., lang=...)`, matching the `kokoro-onnx` usage example from the project docs.
- You can override model file locations with `KOKORO_MODEL_PATH` and `KOKORO_VOICES_PATH`.
- If the health badge shows missing runtime files, install dependencies and add the model assets before generating audio.

## API

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
  "voice": "af_sarah",
  "speed": 1.0,
  "lang": "en-us",
  "format": "opus",
  "opus_bitrate": "32k",
  "target_chunk_chars": 360
}
```

- `opus_bitrate` supports: `16k`, `24k`, `32k`, `48k`
- Sentence boundaries are preserved when creating chunks; `target_chunk_chars` is approximate, not a hard cut.
- `GET /api/health` returns `websocket_streaming` to indicate whether websocket upgrade support is available in the current runtime.

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
- `voice` should be one of the Kokoro voice ids such as `af_heart`
- Supported `response_format` values are currently `wav` and `opus`
- `speed` is accepted in the OpenAI-style request, but Kokoro currently supports `0.5` through `1.8`
