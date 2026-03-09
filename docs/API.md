# API Reference

Base URL: `http://127.0.0.1:8000`

This server exposes two API groups:

- Native Kokoro WebUI routes under `/api/*` plus `ws://.../ws/*`
- OpenAI-compatible routes under `/v1/*`

## Authentication

- Auth is optional and controlled by server configuration.
- When `KOKORO_REQUIRE_AUTH=1`, all non-static API routes require an API key.
- HTTP routes accept either `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- `WS /ws/speak-stream` accepts the bearer header during handshake and also supports an `api_key` field in the first WebSocket message for browser clients that cannot set custom headers.
- `GET /api/public-config` is intentionally public so the built-in Web UI can discover whether auth is required before calling protected routes.

## Quick Links

### Native Endpoints

- [Authentication](#authentication)
- [`GET /api/public-config`](#get-apipublic-config)
- [Common Native Fields](#common-native-fields)
- [`GET /api/health`](#get-apihealth)
- [`GET /api/capabilities`](#get-apicapabilities)
- [`POST /api/speak`](#post-apispeak)
- [`POST /api/chunk-plan`](#post-apichunk-plan)
- [`POST /api/speak-stream`](#post-apispeak-stream)
- [`WS /ws/speak-stream`](#ws-wsspeak-stream)

### OpenAI-Compatible Endpoints

- [OpenAI Compatibility](#openai-compatibility)
- [`GET /v1/models`](#get-v1models)
- [`GET /v1/models/{model_id}`](#get-v1modelsmodel_id)
- [`POST /v1/audio/speech`](#post-v1audiospeech)

### Examples

- [Curl Examples](#curl-examples)

## Native Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/public-config` | Public UI bootstrap config |
| `GET` | `/api/health` | Readiness and queue status |
| `GET` | `/api/capabilities` | Runtime capabilities, voices, formats, and limits |
| `POST` | `/api/speak` | Render one audio response |
| `POST` | `/api/chunk-plan` | Return chunk metadata without audio |
| `POST` | `/api/speak-stream` | Stream chunked audio over NDJSON |
| `WS` | `/ws/speak-stream` | Stream chunked audio over WebSocket |

## Common Native Fields

These fields are shared across the native synthesis routes.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `text` | string | yes | `1..2500` chars |
| `voice` | string | no | Default: `af_heart` |
| `speed` | number | no | Range: `0.5..1.8`, default `1.0` |
| `pitch` | number | no | Range: `-6.0..6.0` semitones, default `0.0` |
| `lang` | string | no | One of `en-us`, `en-gb`, `fr-fr`, `ja`, `ko`, `cmn` |
| `format` | string | no | Server-enabled format such as `pcm`, `wav`, or `opus`. If omitted, the server uses its first enabled format. |
| `opus_bitrate` | string | no | One of `16k`, `24k`, `32k`, `48k` |
| `wav_sample_rate` | string | no | One of `native`, `16000`, `22050`, `24000`, `44100`, `48000` |

Notes:

- `pitch: 0.0` is a no-op and skips backend pitch-shift processing.
- Pitch shifting depends on `ffmpeg` with the `rubberband` filter.
- `opus_bitrate` matters only when `format` is `opus`.
- `wav_sample_rate` matters when `format` is `wav` or `pcm`.
- `/api/capabilities.formats` is the authoritative list of formats enabled on this server.

### `GET /api/public-config`

Returns the minimum public configuration needed by the built-in Web UI before authentication.

### Response

```json
{
  "web_ui_enabled": true,
  "auth_required": true,
  "auth_scheme": "bearer",
  "websocket_auth": "header-or-first-message"
}
```

### `GET /api/health`

Returns lightweight readiness and queue status.

### Response

```json
{
  "ok": true,
  "missing": [],
  "active_provider": "CPUExecutionProvider",
  "active_providers": ["CPUExecutionProvider"],
  "provider_fallback": false,
  "provider_error": null,
  "runtime_error": null,
  "gpu": {
    "process_pid": 41234,
    "available": true,
    "process_vram_used_bytes": 201326592,
    "process_vram_used_mb": 192.0,
    "source": "nvml",
    "error": null
  },
  "queue": {
    "worker_limit": 2,
    "queue_limit": 8,
    "capacity_limit": 10,
    "reserved_jobs": 0,
    "active_jobs": 0,
    "queued_jobs": 0,
    "available_slots": 10,
    "admitted_jobs_total": 24,
    "completed_jobs_total": 24,
    "rejected_jobs_total": 0,
    "queue_wait_last_ms": 0.21,
    "queue_wait_avg_ms": 0.11,
    "queue_wait_max_ms": 1.48,
    "queue_wait_samples": 24
  }
}
```

### Field Notes

| Field | Meaning |
| --- | --- |
| `ok` | `true` when runtime and model assets are available |
| `missing` | Missing dependency or asset names |
| `active_provider` | Currently active provider, if available |
| `provider_fallback` | Whether runtime fell back from the requested provider |
| `runtime_error` | Runtime bootstrap failure, if any |
| `gpu` | Per-process GPU memory for the current server process when NVML is available and the runtime is using a GPU provider |
| `queue` | Current scheduler capacity, rejection counters, and queue-wait timing |

### `GET /api/capabilities`

Returns the full runtime and configuration surface used by the Web UI.

### Response

```json
{
  "model_path": "/abs/path/models/kokoro-v1.0.onnx",
  "voices_path": "/abs/path/models/voices-v1.0.bin",
  "voices": ["af_heart", "af_sarah"],
  "formats": ["wav", "opus", "pcm"],
  "opus_bitrates": ["16k", "24k", "32k", "48k"],
  "wav_sample_rates": ["native", "16000", "22050", "24000", "44100", "48000"],
  "requested_provider": "auto",
  "attempted_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "active_provider": "CPUExecutionProvider",
  "active_providers": ["CPUExecutionProvider"],
  "provider_fallback": true,
  "provider_error": "Failed to load CUDA runtime.",
  "runtime_error": null,
  "pitch_shifting": true,
  "synthesis_workers": 2,
  "synthesis_queue_limit": 8,
  "scheduler": {
    "requested_provider": "auto",
    "active_provider": "CPUExecutionProvider",
    "runtime_kind": "gpu",
    "execution_model": "shared-runtime",
    "supported_execution_models": ["shared-runtime"],
    "planned_execution_models": ["session-pool"],
    "worker_limit": 2,
    "queue_limit": 8,
    "interactive_reserve_slots": 1,
    "prefers_serial_workers": true,
    "experimental_gpu_concurrency": true,
    "concurrency_note": "GPU-preferred mode currently shares one runtime session; keep workers at 1 unless benchmarked.",
    "warning": "GPU-preferred mode currently shares one runtime session. If this resolves to CUDA, KOKORO_SYNTH_WORKERS > 1 is an experimental tuning path."
  },
  "max_pitch_semitones": 6.0,
  "streaming": true,
  "websocket_streaming": true,
  "queue": {
    "worker_limit": 2,
    "queue_limit": 8,
    "capacity_limit": 10,
    "interactive_reserve_slots": 1,
    "stream_capacity_limit": 9,
    "reserved_jobs": 0,
    "active_jobs": 0,
    "queued_jobs": 0,
    "available_slots": 10,
    "admitted_jobs_total": 24,
    "completed_jobs_total": 24,
    "rejected_jobs_total": 0,
    "queue_wait_last_ms": 0.21,
    "queue_wait_avg_ms": 0.11,
    "queue_wait_max_ms": 1.48,
    "queue_wait_samples": 24
  }
}
```

### Field Notes

| Field | Meaning |
| --- | --- |
| `voices` | Available Kokoro voice IDs |
| `pitch_shifting` | Whether backend pitch shift is available |
| `websocket_streaming` | Whether the runtime can serve WebSocket streaming |
| `synthesis_workers` | Number of active synthesis worker threads |
| `synthesis_queue_limit` | Number of queued jobs allowed behind active workers |
| `scheduler` | Provider-aware scheduling policy currently used by the backend |
| `queue.queue_wait_*` | Recent and cumulative wait-time indicators for admitted jobs |

### `POST /api/speak`

Generates one audio response.

### Request

```json
{
  "text": "The future sounds calmer when it is rendered locally.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": -2.0,
  "lang": "en-us",
  "format": "wav",
  "wav_sample_rate": "native"
}
```

### Success Response

Returns audio bytes directly.

- `audio/pcm` when `format` is `pcm`
- `audio/wav` when `format` is `wav`
- `audio/ogg` when `format` is `opus`

### Response Headers

| Header | Meaning |
| --- | --- |
| `Content-Disposition` | Inline filename |
| `X-Audio-Bytes` | Encoded audio byte length |
| `X-Audio-Format` | `pcm`, `wav`, or `opus` |
| `X-Sample-Rate` | Final sample rate |
| `X-Audio-Duration` | Duration in seconds |
| `X-Opus-Bitrate` | Populated for Opus responses |
| `X-Wav-Sample-Rate` | Populated for WAV responses |

### Error Response

Status: `400`

```json
{
  "detail": "Pitch shifting failed: ..."
}
```

### `POST /api/chunk-plan`

Returns how the server would split text before synthesis.

### Request

```json
{
  "text": "Good evening. This is a longer passage. It will be split by sentence boundaries.",
  "target_chunk_chars": 360,
  "include_text": false
}
```

### Request Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `text` | string | yes | `1..2500` chars |
| `target_chunk_chars` | integer | no | Range: `80..2000`, default `360` |
| `include_text` | boolean | no | Include chunk text in the response |

### Response

```json
{
  "chunks": [
    {
      "index": 0,
      "char_count": 67,
      "word_count": 12,
      "sentence_count": 2
    }
  ],
  "count": 1,
  "lengths": [67],
  "target_chunk_chars": 360
}
```

If `include_text` is `true`, each chunk object also includes `text`.

### `POST /api/speak-stream`

Streams chunked synthesis over NDJSON.

### Request

Uses all native synthesis fields plus `target_chunk_chars`.

```json
{
  "text": "Long text here.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": 2.0,
  "lang": "en-us",
  "format": "opus",
  "opus_bitrate": "32k",
  "target_chunk_chars": 360
}
```

### Additional Field

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `target_chunk_chars` | integer | no | Range: `80..2000`, default `360` |

### Response Type

`application/x-ndjson`

Each line is one JSON object with a `type`.

### `meta` Message

```json
{
  "type": "meta",
  "total_chunks": 3,
  "format": "opus",
  "opus_bitrate": "32k",
  "wav_sample_rate": null,
  "pitch": 2.0,
  "target_chunk_chars": 360
}
```

### `chunk` Message

```json
{
  "type": "chunk",
  "chunk_index": 0,
  "total_chunks": 3,
  "text": "First chunk.",
  "pitch": 2.0,
  "bytes": 12234,
  "sample_rate": 24000,
  "duration_sec": 1.28,
  "synth_ms": 85.41,
  "format": "opus",
  "opus_bitrate": "32k",
  "wav_sample_rate": null,
  "mime_type": "audio/ogg",
  "audio_base64": "..."
}
```

### `error` Message

```json
{
  "type": "error",
  "detail": "Pitch shifting failed: ...",
  "chunk_index": 1
}
```

### `done` Message

```json
{
  "type": "done",
  "total_chunks": 3
}
```

### `WS /ws/speak-stream`

Streams the same message types as `/api/speak-stream`, but over WebSocket.

### Client Flow

1. Connect to `ws://127.0.0.1:8000/ws/speak-stream`
2. Send one JSON text message matching the `/api/speak-stream` request body
3. Read `meta`, `chunk`, `error`, and `done` messages as JSON text frames

### Example Client Payload

```json
{
  "text": "Long text here.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": -1.5,
  "lang": "en-us",
  "format": "wav",
  "wav_sample_rate": "native",
  "target_chunk_chars": 360
}
```

### Message Types

The server sends the same `meta`, `chunk`, `error`, and `done` shapes documented for `/api/speak-stream`.

## OpenAI Compatibility

Most OpenAI-compatible clients expect the base URL to already include `/v1`, for example:

- `http://127.0.0.1:8000/v1`

That is usually the correct setting for external integrations. If you point those clients at `http://127.0.0.1:8000` instead, many of them will append `/models` or `/audio/speech` on their own and miss the compatibility routes.

If auth is enabled, use the same bearer key you use for the native API.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/v1/models/{model_id}` | OpenAI-compatible model metadata |
| `POST` | `/v1/audio/speech` | OpenAI-compatible speech generation |

### `GET /v1/models`

OpenAI-compatible model list.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "kokoro",
      "object": "model",
      "created": 0,
      "owned_by": "kokoro-webui"
    }
  ]
}
```

### `GET /v1/models/{model_id}`

Returns metadata for `kokoro`.

### Success Response

```json
{
  "id": "kokoro",
  "object": "model",
  "created": 0,
  "owned_by": "kokoro-webui"
}
```

### Not Found Response

Status: `404`

```json
{
  "error": {
    "message": "The model 'other-model' does not exist.",
    "type": "invalid_request_error",
    "param": null,
    "code": "model_not_found"
  }
}
```

### `POST /v1/audio/speech`

OpenAI-compatible speech generation.

### Request

```json
{
  "model": "kokoro",
  "input": "The future sounds calmer when it is rendered locally.",
  "voice": "af_heart",
  "response_format": "wav",
  "speed": 1.0
}
```

### Request Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Accepted for compatibility, currently ignored |
| `input` | string | yes | `1..4096` chars |
| `voice` | string or object | yes | String voice ID, optional suffixed form like `af_heart+2.0`, or object form like `{ "id": "af_heart-2.0" }` |
| `response_format` | string | no | Server-enabled format such as `pcm`, `wav`, or `opus`. If omitted, the server uses its first enabled format. |
| `speed` | number | no | Accepted range `0.25..4.0`, but values outside `0.5..1.8` are rejected by the Kokoro adapter |
| `instructions` | string or null | no | Accepted but currently unused |
| `stream_format` | string or null | no | `sse` is not supported |

### Success Response

Returns audio bytes directly.

- `audio/pcm` when `response_format` is `pcm`
- `audio/wav` when `response_format` is `wav`
- `audio/ogg` when `response_format` is `opus`

For `wav` and `pcm`, the server writes the response progressively with HTTP chunked transfer encoding so clients can start playback before the full render finishes. That matches the common OpenAI `/v1/audio/speech` streaming pattern more closely. `opus` still returns after the full render is ready.

### Response Headers

| Header | Meaning |
| --- | --- |
| `X-OpenAI-Compatible` | Always `kokoro` |
| `X-Audio-Format` | `pcm`, `wav`, or `opus` |
| `X-Sample-Rate` | Final sample rate |

### Error Response

Status: `400`

```json
{
  "error": {
    "message": "speed must be between 0.5 and 1.8 for Kokoro.",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

Notes:

- This compatibility route does not expose a separate `pitch` field. Use the `voice` suffix forms `af_heart+2.0` or `af_heart-2.0` instead.
- The suffix is optional. `af_heart` implies `0.0` pitch shift and skips post-processing.
- The signed voice suffix must stay within `-6.0..+6.0` semitones.
- This compatibility route does not expose native `lang`, `opus_bitrate`, `wav_sample_rate`, or `target_chunk_chars`.
- `stream_format: "sse"` returns an error.

## Curl Examples

### Health

```bash
curl http://127.0.0.1:8000/api/health
```

### Single WAV Render

```bash
curl -X POST http://127.0.0.1:8000/api/speak \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Local TTS stays fast when the interface stays honest.",
    "voice": "af_heart",
    "speed": 1.0,
    "pitch": -1.0,
    "lang": "en-us",
    "format": "wav",
    "wav_sample_rate": "native"
  }' \
  --output output.wav
```

### NDJSON Stream

```bash
curl -N -X POST http://127.0.0.1:8000/api/speak-stream \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "First sentence. Second sentence. Third sentence.",
    "voice": "af_heart",
    "speed": 1.0,
    "pitch": 1.5,
    "lang": "en-us",
    "format": "opus",
    "opus_bitrate": "32k",
    "target_chunk_chars": 120
  }'
```

### OpenAI-Compatible Speech

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "This request uses the OpenAI-compatible endpoint.",
    "voice": "af_heart+1.5",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output openai-output.wav
```

### OpenAI-Compatible Streaming WAV

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer 1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Good evening, and welcome. Tonight I want to speak about calm systems, patient work, and the quiet value of doing things well.",
    "voice": "af_heart",
    "response_format": "wav",
    "speed": 1.0
  }' | ffplay -i -
```

### OpenAI-Compatible Streaming PCM

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer 1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Good evening, and welcome. Tonight I want to speak about calm systems, patient work, and the quiet value of doing things well.",
    "voice": "af_heart",
    "response_format": "pcm",
    "speed": 1.0
  }' | ffplay -f s16le -ar 24000 -ch_layout mono -i -
```

### WebSocket Stream

This example uses [`websocat`](https://github.com/vi/websocat).

```bash
printf '%s\n' '{
  "text": "First sentence. Second sentence. Third sentence.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": 0.0,
  "lang": "en-us",
  "format": "wav",
  "wav_sample_rate": "native",
  "target_chunk_chars": 120
}' | websocat ws://127.0.0.1:8000/ws/speak-stream
```
