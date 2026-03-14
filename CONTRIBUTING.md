# Contributing to Kokoro WebUI

Thanks for your interest in contributing! This document covers development setup, project structure, and guidelines.

## Development Setup

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Model files in `models/` (see main README)

### Install Development Dependencies

```bash
uv sync --dev
```

This installs all dependencies plus testing tools.

### Run Development Server

```bash
KOKORO_RELOAD=1 ./scripts/run-server.sh
```

Auto-reload watches for code changes and restarts the server.

## Project Structure

```
app/
  api.py           # FastAPI application, routes, middleware
  audio.py         # Audio processing, ffmpeg integration
  config.py        # Environment variable parsing
  main.py          # Entry point
  openai_compat.py # OpenAI API compatibility layer
  runtime.py       # ONNX runtime management
  scheduler.py     # Synthesis job scheduling
  schemas.py       # Pydantic models
  status_stream.py # SSE status broadcast

static/
  index.html       # Main web UI
  js/              # Frontend JavaScript
  styles/          # CSS

docs/
  API.md           # API endpoint documentation
  TUNING.md        # Performance tuning guide

tests/
  test_api.py      # Backend integration tests
  test_frontend.mjs # Frontend regression tests

scripts/
  run-server.sh    # Server launcher
  tune_runtime.py  # Interactive performance tuner
  benchmark_runtime.py # Benchmark harness
```

## Architecture Overview

### Request Flow

1. Request enters through FastAPI (`app/api.py`)
2. Auth middleware validates API key (if required)
3. Request routed to appropriate handler
4. Synthesis job queued via scheduler
5. ONNX runtime generates PCM audio
6. Optional post-processing (pitch, Opus encoding)
7. Response returned (full file or streamed chunks)

### Status Streaming

The built-in UI uses Server-Sent Events (SSE) for live status:

- `GET /api/health` - Bootstrap snapshot endpoint
- `GET /api/health/stream` - Live SSE feed
- Browser tabs share one SSE connection via `BroadcastChannel`
- Single in-process publisher fans out to clients

This replaces the older polling approach for better efficiency.

### Audio Pipeline

```
Text → Kokoro ONNX → PCM
                     ↓
              [optional pitch shift]
                     ↓
              [optional Opus encode]
                     ↓
                  Output
```

- **PCM**: Raw 16-bit, no container (fastest)
- **WAV**: PCM with WAV header
- **Opus**: Compressed via ffmpeg (requires ffmpeg installed)

## Testing

### Backend Tests

```bash
# Run all tests
uv run python -m unittest tests.test_api tests.test_runtime tests.test_scheduler

# Run specific test class
uv run python -m unittest tests.test_api.ApiIntegrationTests

# Run with verbose output
uv run python -m unittest tests.test_api -v
```

### Frontend Tests

```bash
# Requires Node.js
uv run node tests/test_frontend.mjs
```

### Test Coverage Areas

- **Auth**: Key validation, rate limiting, throttling
- **Synthesis**: All formats, streaming vs batch, error handling
- **WebSocket**: Session tokens, auth handshake, streaming
- **Proxy headers**: Trusted/untrusted IPs, chain parsing, malformed headers
- **Status stream**: SSE events, subscriber management

## Code Style

We use Python's standard formatting. Key conventions:

- Type hints on function signatures
- Docstrings for public functions
- Explicit over implicit
- Fail closed on security decisions

Run checks:

```bash
# Type checking
uv run pyright

# Linting (if configured)
uv run ruff check .
```

## Performance Tuning

Use the included tools to find optimal settings for your hardware:

```bash
# Interactive tuner
uv run python scripts/tune_runtime.py --provider cpu

# Raw benchmark
uv run python scripts/benchmark_runtime.py --providers cpu auto --concurrency 4
```

Key variables to tune:
- `KOKORO_SYNTH_WORKERS` - Concurrent synthesis jobs
- `KOKORO_SYNTH_QUEUE` - Queue depth

## Adding Features

### New Endpoints

Add routes in `app/api.py`. Follow existing patterns:

```python
@app.post("/api/new-feature")
async def new_feature(request: NewFeatureRequest) -> Response:
    # Implementation
    pass
```

### New Config Options

Add to `app/config.py`:

1. Parse function
2. Validation
3. Documentation in CONFIGURATION.md

### Frontend Changes

Static files are served from `static/`. The UI is vanilla JS/HTML - no build step required.

Key files:
- `static/index.html` - Main page
- `static/js/api-client.js` - API interactions
- `static/js/playback.js` - Audio playback

## Security Considerations

When modifying auth or proxy handling:

1. **Trust boundaries**: Only trust headers from explicitly configured IPs
2. **Canonicalization**: Normalize IP addresses before comparison
3. **Fail closed**: Reject malformed input rather than trying to parse it
4. **Rate limiting**: Consider per-client impact of changes

See recent proxy header hardening commits for examples.

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure tests pass
5. Submit a pull request

Include:
- Clear description of the change
- Why it's needed
- Test coverage
- Documentation updates

## Questions?

- API questions: See `docs/API.md`
- Config questions: See `CONFIGURATION.md`
- Usage questions: See main README
