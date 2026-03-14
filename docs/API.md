# API Reference

Base URL: `http://127.0.0.1:8000`

Kokoro WebUI provides two API groups:

- **Native API** - Full-featured endpoints under `/api/*` and `ws://.../ws/*`
- **OpenAI-compatible API** - Drop-in replacement for OpenAI's `/v1/audio/speech`

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status and queue |
| `/api/capabilities` | GET | List voices, formats, and limits |
| `/api/speak` | POST | Generate single audio file |
| `/api/speak-stream` | POST | Stream audio as NDJSON |
| `/ws/speak-stream` | WS | Stream audio over WebSocket |
| `/v1/audio/speech` | POST | OpenAI-compatible endpoint |

For practical examples, see [examples/README.md](../examples/).

## Authentication

Authentication is optional. When `KOKORO_REQUIRE_AUTH=1` is set:

- HTTP routes accept `Authorization: Bearer <key>` or `X-API-Key: <key>`
- WebSocket accepts the bearer header during handshake
- The built-in Web UI uses short-lived session tokens instead of the API key
- Query string API keys are not supported (security)

Failed auth attempts are rate-limited per client. After too many failures, you'll get `429 Too Many Requests` with a `Retry-After` header.

See [CONFIGURATION.md](../CONFIGURATION.md) for details on setting up authentication.

## Native Endpoints

### Common Fields

These fields work across most native synthesis endpoints:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | string | yes | 1 to 2500 characters |
| `voice` | string | no | Default: `af_heart` |
| `speed` | number | no | 0.5 to 1.8, default 1.0 |
| `pitch` | number | no | -6.0 to 6.0 semitones, default 0.0 |
| `format` | string | no | `pcm`, `wav`, or `opus` (server-dependent) |
| `opus_bitrate` | string | no | `16k`, `24k`, `32k`, or `48k` |
| `wav_sample_rate` | string | no | `native`, `16000`, `22050`, `24000`, `44100`, `48000` |

Notes:

- `pitch: 0.0` skips pitch processing entirely (faster)
- Pitch shifting requires ffmpeg with rubberband filter
- Check `/api/capabilities` to see which formats your server supports

### GET /api/health

Returns server status, queue information, and runtime health.

**Example response:**

```json
{
  "ok": true,
  "missing": [],
  "active_provider": "CPUExecutionProvider",
  "gpu": {
    "available": false,
    "process_vram_used_mb": 0
  },
  "queue": {
    "worker_limit": 2,
    "queue_limit": 8,
    "active_jobs": 0,
    "queued_jobs": 0,
    "available_slots": 10
  }
}
```

Key fields:

- `ok` - `true` when everything is working
- `missing` - List of missing dependencies or assets
- `active_provider` - Current runtime (CPU, CUDA, etc.)
- `queue` - Current load and capacity

### GET /api/capabilities

Returns detailed information about available features.

**Example response:**

```json
{
  "voices": ["af_heart", "af_sarah", "am_michael"],
  "formats": ["wav", "opus", "pcm"],
  "pitch_shifting": true,
  "synthesis_workers": 2,
  "websocket_streaming": true
}
```

Use this to populate UI controls or validate requests.

### POST /api/speak

Generate a single audio file from text.

**Request:**

```json
{
  "text": "Hello world",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": 0.0,
  "format": "wav"
}
```

**Response:**

Returns raw audio bytes:

- `audio/pcm` for `format: pcm`
- `audio/wav` for `format: wav`
- `audio/ogg` for `format: opus`

Response headers include metadata:

| Header | Description |
|--------|-------------|
| `X-Audio-Format` | Format used |
| `X-Sample-Rate` | Sample rate in Hz |
| `X-Audio-Duration` | Duration in seconds |

**Errors:**

- `400` - Invalid request (bad voice, format not supported, etc.)
- `503` - Server overloaded (queue full)

### POST /api/speak-stream

Stream audio chunks as they're generated using NDJSON (Newline Delimited JSON).

**Request:**

Same as `/api/speak`, plus optional:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_chunk_chars` | integer | 360 | Target characters per chunk (80-2000) |

**Response:**

Content-Type: `application/x-ndjson`

Each line is a JSON object with a `type` field:

**Meta message** (first):

```json
{
  "type": "meta",
  "total_chunks": 3,
  "format": "wav"
}
```

**Chunk message** (one per audio chunk):

```json
{
  "type": "chunk",
  "chunk_index": 0,
  "total_chunks": 3,
  "text": "First sentence.",
  "duration_sec": 1.28,
  "audio_base64": "..."
}
```

**Error message** (if a chunk fails):

```json
{
  "type": "error",
  "detail": "Pitch shifting failed",
  "chunk_index": 1
}
```

**Done message** (last):

```json
{
  "type": "done",
  "total_chunks": 3
}
```

### WS /ws/speak-stream

WebSocket endpoint for real-time streaming.

**Usage:**

1. Connect to `ws://127.0.0.1:8000/ws/speak-stream`
2. Send a JSON message with synthesis parameters
3. Receive messages in the same format as NDJSON streaming

**Example client payload:**

```json
{
  "text": "WebSocket streaming example",
  "voice": "af_heart",
  "format": "wav",
  "target_chunk_chars": 360
}
```

WebSocket is ideal for real-time applications where you want to start playing audio before the full text is synthesized.

## OpenAI-Compatible Endpoints

Kokoro WebUI implements the OpenAI audio speech API, making it a drop-in replacement for applications that use OpenAI's TTS.

**Base URL:** `http://127.0.0.1:8000/v1`

### POST /v1/audio/speech

OpenAI-compatible speech generation.

**Request:**

```json
{
  "model": "kokoro",
  "input": "Hello from the compatible endpoint",
  "voice": "af_heart",
  "response_format": "wav",
  "speed": 1.0
}
```

**Fields:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `model` | string | yes | Ignored (accepted for compatibility) |
| `input` | string | yes | Text to synthesize (1-4096 chars) |
| `voice` | string | yes | Voice ID, or with pitch suffix like `af_heart+2.0` |
| `response_format` | string | no | `pcm`, `wav`, or `opus` |
| `speed` | number | no | 0.25-4.0 (but Kokoro only supports 0.5-1.8) |

**Pitch with OpenAI API:**

The OpenAI API doesn't have a `pitch` field, so we use voice suffixes:

- `af_heart` - Normal pitch (0.0)
- `af_heart+2.0` - Raise pitch 2 semitones
- `af_heart-1.5` - Lower pitch 1.5 semitones

Suffix must be between -6.0 and +6.0.

**Response:**

Returns raw audio bytes:

- `wav` and `pcm` stream progressively (chunked transfer encoding)
- `opus` returns after full render

**Headers:**

| Header | Description |
|--------|-------------|
| `X-OpenAI-Compatible` | Always `kokoro` |
| `X-Audio-Format` | Format used |

**Errors:**

Returns OpenAI-compatible error format:

```json
{
  "error": {
    "message": "speed must be between 0.5 and 1.8 for Kokoro",
    "type": "invalid_request_error"
  }
}
```

### GET /v1/models

Returns available models (just `kokoro`).

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "kokoro",
      "object": "model",
      "owned_by": "kokoro-webui"
    }
  ]
}
```

### GET /v1/models/{model_id}

Returns metadata for a specific model.

**Response:**

```json
{
  "id": "kokoro",
  "object": "model",
  "owned_by": "kokoro-webui"
}
```

## Error Handling

All endpoints return consistent error formats:

**Native API errors:**

```json
{
  "detail": "Voice 'invalid_voice' not found"
}
```

**OpenAI-compatible errors:**

```json
{
  "error": {
    "message": "Voice 'invalid_voice' not found",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

**Common status codes:**

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (auth required but missing) |
| 429 | Too many requests (rate limited) |
| 503 | Service unavailable (queue full) |

## Using with OpenAI Clients

### Official Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # Or your API key if auth is enabled
)

response = client.audio.speech.create(
    model="kokoro",
    input="Hello world",
    voice="af_heart",
    response_format="wav"
)

response.stream_to_file("output.wav")
```

### LangChain

```python
from langchain_community.tools import OpenAITTS

tts = OpenAITTS(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

audio = tts.invoke({
    "input": "Hello from LangChain",
    "voice": "af_heart"
})
```

## More Examples

For complete examples in multiple languages (curl, Python, JavaScript), see [examples/README.md](../examples/).

## Differences from OpenAI

While we aim for compatibility, there are some differences:

1. **No SSE streaming** - `stream_format: "sse"` returns an error
2. **Pitch via suffix** - Use `voice+2.0` instead of a separate pitch field
3. **Speed limits** - Kokoro only supports 0.5x to 1.8x speed
4. **Model ignored** - The `model` field is accepted but ignored
5. **Native features missing** - No access to `lang`, `opus_bitrate`, or `target_chunk_chars` via OpenAI endpoints

For full access to all features, use the native `/api/*` endpoints.
