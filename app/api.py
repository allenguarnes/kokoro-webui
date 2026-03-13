# pyright: reportUnusedFunction=false, reportUnusedCallResult=false
from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import re
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import ParamSpec, Protocol, TypeVar, cast, runtime_checkable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import audio, config, openai_compat, runtime
from app.scheduler import (
    SynthesisOverloadedError,
    SynthesisScheduler,
    build_scheduler_policy,
)
from app.schemas import (
    ChunkedSynthesisRequest,
    ChunkMetadataRequest,
    ChunkPlanEntry,
    OpenAIModelListResponse,
    OpenAISpeechRequest,
    RenderedChunk,
    SynthesisRequest,
)

P = ParamSpec("P")
T = TypeVar("T")
_OPENAI_STREAM_TARGET_CHARS = 360


@runtime_checkable
class HeaderLookup(Protocol):
    def get(self, key: str, default: str = "") -> str | None: ...


def create_app() -> FastAPI:
    web_ui_enabled = config.get_web_ui_enabled()
    require_auth = config.get_require_auth()
    configured_api_key = config.get_api_key()
    allowed_origins = config.get_allowed_origins()
    auth_failure_limit = config.get_auth_failure_limit()
    auth_failure_window_sec = config.get_auth_failure_window_seconds()
    auth_failure_max_buckets = config.get_auth_failure_max_buckets()
    trust_proxy_headers = config.get_trust_proxy_headers()
    ws_auth_handshake_timeout_sec = (
        config.get_websocket_auth_handshake_timeout_seconds()
    )
    ws_session_token_ttl_sec = config.get_websocket_session_token_ttl_seconds()
    ws_session_token_max_tokens = config.get_websocket_session_token_max_tokens()
    requested_provider = config.get_runtime_provider_mode()
    runtime_status = runtime.get_runtime_status()
    active_runtime_provider = (
        runtime_status.active_providers[0] if runtime_status.active_providers else None
    )
    default_synthesis_workers = 2 if requested_provider == "cpu" else 1
    synthesis_workers = config.get_synthesis_workers(default=default_synthesis_workers)
    default_synthesis_queue = synthesis_workers * 4
    synthesis_queue_limit = config.get_synthesis_queue_limit(
        default=default_synthesis_queue
    )
    allow_experimental_gpu_concurrency = (
        config.get_allow_experimental_cuda_concurrency()
    )
    scheduler_policy = build_scheduler_policy(
        requested_provider=requested_provider,
        active_provider=active_runtime_provider,
        worker_limit=synthesis_workers,
        queue_limit=synthesis_queue_limit,
        allow_experimental_gpu_concurrency=allow_experimental_gpu_concurrency,
    )
    synthesis_scheduler = SynthesisScheduler(
        policy=scheduler_policy,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            synthesis_scheduler.shutdown()

    app = FastAPI(title="Kokoro WebUI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )
    if web_ui_enabled:
        app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

    auth_failures: dict[str, list[float]] = {}
    auth_failures_lock = threading.Lock()
    auth_failures_tick = 0
    ws_session_tokens: dict[str, tuple[float, str]] = {}
    ws_session_tokens_lock = threading.Lock()

    def prune_ws_session_tokens(now: float) -> None:
        if not ws_session_tokens:
            return
        for token, (expires_at, _) in list(ws_session_tokens.items()):
            if expires_at <= now:
                ws_session_tokens.pop(token, None)

    def compact_ws_session_tokens(now: float) -> None:
        prune_ws_session_tokens(now)
        if len(ws_session_tokens) <= ws_session_token_max_tokens:
            return

        overflow = len(ws_session_tokens) - ws_session_token_max_tokens
        oldest = sorted(ws_session_tokens.items(), key=lambda item: item[1][0])[
            :overflow
        ]
        for token, _ in oldest:
            ws_session_tokens.pop(token, None)

    def issue_ws_session_token(client_id: str) -> tuple[str, int]:
        now = time.monotonic()
        token = secrets.token_urlsafe(24)
        expires_in = max(1, int(ws_session_token_ttl_sec))
        with ws_session_tokens_lock:
            ws_session_tokens[token] = (now + ws_session_token_ttl_sec, client_id)
            compact_ws_session_tokens(now)
        return token, expires_in

    def consume_ws_session_token(candidate: str | None, client_id: str) -> bool:
        if not candidate:
            return False
        now = time.monotonic()
        with ws_session_tokens_lock:
            compact_ws_session_tokens(now)
            token_entry = ws_session_tokens.pop(candidate, None)
        if token_entry is None:
            return False
        expires_at, bound_client_id = token_entry
        return expires_at > now and hmac.compare_digest(bound_client_id, client_id)

    def is_valid_api_key(candidate: str | None) -> bool:
        if not require_auth:
            return True
        if candidate is None or configured_api_key is None:
            return False
        return hmac.compare_digest(candidate, configured_api_key)

    def prune_failure_timestamps(timestamps: list[float], now: float) -> list[float]:
        cutoff = now - auth_failure_window_sec
        return [timestamp for timestamp in timestamps if timestamp >= cutoff]

    def auth_identifier_from_host(
        host: str | None, forwarded: str | None = None
    ) -> str:
        selected = forwarded if forwarded else host
        cleaned = (selected or "").strip()
        return cleaned or "unknown-client"

    def get_forwarded_host(headers: object) -> str | None:
        if not trust_proxy_headers:
            return None
        if not isinstance(headers, HeaderLookup):
            return None
        get_header = headers.get
        forwarded_for = (get_header("X-Forwarded-For", "") or "").strip()
        if forwarded_for:
            first = forwarded_for.split(",", 1)[0].strip()
            if first:
                return first
        real_ip = (get_header("X-Real-IP", "") or "").strip()
        if real_ip:
            return real_ip
        return None

    def auth_bucket_id(source_id: str) -> str:
        return source_id

    def compact_auth_failures(now: float) -> None:
        if not auth_failures:
            return
        for client_id in list(auth_failures.keys()):
            timestamps = prune_failure_timestamps(auth_failures[client_id], now)
            if timestamps:
                auth_failures[client_id] = timestamps
            else:
                auth_failures.pop(client_id, None)

        if len(auth_failures) <= auth_failure_max_buckets:
            return

        overflow = len(auth_failures) - auth_failure_max_buckets
        oldest = sorted(auth_failures.items(), key=lambda item: item[1][-1])[:overflow]
        for client_id, _ in oldest:
            auth_failures.pop(client_id, None)

    def record_auth_failure(client_id: str) -> int:
        nonlocal auth_failures_tick
        if auth_failure_limit <= 0:
            return 0
        now = time.monotonic()
        with auth_failures_lock:
            auth_failures_tick += 1
            if (
                auth_failures_tick % 64 == 0
                or len(auth_failures) >= auth_failure_max_buckets
            ):
                compact_auth_failures(now)
            timestamps = prune_failure_timestamps(auth_failures.get(client_id, []), now)
            timestamps.append(now)
            auth_failures[client_id] = timestamps
            return len(timestamps)

    def clear_auth_failures(client_id: str) -> None:
        with auth_failures_lock:
            auth_failures.pop(client_id, None)

    def auth_failure_throttled(failure_count: int) -> bool:
        if auth_failure_limit <= 0:
            return False
        return failure_count > auth_failure_limit

    def extract_http_api_key(request: Request) -> str | None:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            return token or None
        api_key_header = request.headers.get("X-API-Key", "").strip()
        return api_key_header or None

    def extract_http_client_id(request: Request) -> str:
        forwarded_host = get_forwarded_host(request.headers)
        return auth_identifier_from_host(
            request.client.host if request.client else None,
            forwarded_host,
        )

    def http_auth_exception() -> HTTPException:
        return HTTPException(
            status_code=401,
            detail="Authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def http_rate_limit_exception() -> HTTPException:
        return HTTPException(
            status_code=429,
            detail="Too many authentication failures. Try again later.",
            headers={"Retry-After": str(max(1, int(auth_failure_window_sec)))},
        )

    def ensure_http_auth(request: Request) -> None:
        candidate = extract_http_api_key(request)
        client_id = extract_http_client_id(request)
        bucket_id = auth_bucket_id(client_id)
        if is_valid_api_key(candidate):
            clear_auth_failures(bucket_id)
            return
        failure_count = record_auth_failure(bucket_id)
        if auth_failure_throttled(failure_count):
            raise http_rate_limit_exception()
        raise http_auth_exception()

    def openai_auth_error_response(status_code: int = 401) -> JSONResponse:
        if status_code == 429:
            response = openai_compat.openai_error_response(
                429,
                "Too many authentication failures. Try again later.",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            )
            response.headers["Retry-After"] = str(max(1, int(auth_failure_window_sec)))
            return response
        response = openai_compat.openai_error_response(
            401,
            "Authentication failed.",
            error_type="invalid_request_error",
            code="invalid_api_key",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    def openai_auth_error_for_request(request: Request) -> JSONResponse | None:
        candidate = extract_http_api_key(request)
        client_id = extract_http_client_id(request)
        bucket_id = auth_bucket_id(client_id)
        if is_valid_api_key(candidate):
            clear_auth_failures(bucket_id)
            return None
        failure_count = record_auth_failure(bucket_id)
        if auth_failure_throttled(failure_count):
            return openai_auth_error_response(429)
        return openai_auth_error_response(401)

    def extract_websocket_api_key(websocket: WebSocket) -> str | None:
        auth_header = websocket.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                return token
        return None

    def extract_websocket_session_token(message_payload: object) -> str | None:
        if not isinstance(message_payload, dict):
            return None
        message_map = cast(dict[str, object], message_payload)
        message_token = message_map.get("ws_token")
        if not isinstance(message_token, str):
            return None
        cleaned = message_token.strip()
        return cleaned or None

    def extract_websocket_client_id(websocket: WebSocket) -> str:
        forwarded_host = get_forwarded_host(websocket.headers)
        return auth_identifier_from_host(
            websocket.client.host if websocket.client else None,
            forwarded_host,
        )

    def websocket_auth_error_detail(
        websocket: WebSocket, message_payload: object
    ) -> str | None:
        client_id = extract_websocket_client_id(websocket)
        if require_auth:
            ws_token = extract_websocket_session_token(message_payload)
            if consume_ws_session_token(ws_token, client_id):
                return None
        candidate = extract_websocket_api_key(websocket)
        bucket_id = auth_bucket_id(client_id)
        if is_valid_api_key(candidate):
            clear_auth_failures(bucket_id)
            return None
        failure_count = record_auth_failure(bucket_id)
        if auth_failure_throttled(failure_count):
            return "Too many authentication failures. Try again later."
        return "Authentication failed."

    def queue_payload() -> dict[str, int | float]:
        metrics = synthesis_scheduler.snapshot()
        return {
            "worker_limit": metrics.worker_limit,
            "queue_limit": metrics.queue_limit,
            "capacity_limit": metrics.capacity_limit,
            "interactive_reserve_slots": metrics.interactive_reserve_slots,
            "stream_capacity_limit": metrics.stream_capacity_limit,
            "reserved_jobs": metrics.reserved_jobs,
            "active_jobs": metrics.active_jobs,
            "queued_jobs": metrics.queued_jobs,
            "available_slots": metrics.available_slots,
            "admitted_jobs_total": metrics.admitted_jobs_total,
            "completed_jobs_total": metrics.completed_jobs_total,
            "rejected_jobs_total": metrics.rejected_jobs_total,
            "queue_wait_last_ms": metrics.queue_wait_last_ms,
            "queue_wait_avg_ms": metrics.queue_wait_avg_ms,
            "queue_wait_max_ms": metrics.queue_wait_max_ms,
            "queue_wait_samples": metrics.queue_wait_samples,
        }

    def scheduler_payload() -> dict[str, object]:
        return {
            "requested_provider": scheduler_policy.requested_provider,
            "active_provider": scheduler_policy.active_provider,
            "runtime_kind": scheduler_policy.runtime_kind,
            "execution_model": scheduler_policy.execution_model,
            "supported_execution_models": [scheduler_policy.execution_model],
            "planned_execution_models": (
                ["session-pool"]
                if scheduler_policy.execution_model != "session-pool"
                else []
            ),
            "worker_limit": scheduler_policy.worker_limit,
            "queue_limit": scheduler_policy.queue_limit,
            "interactive_reserve_slots": scheduler_policy.interactive_reserve_slots,
            "prefers_serial_workers": scheduler_policy.prefers_serial_workers,
            "experimental_gpu_concurrency": (
                scheduler_policy.experimental_gpu_concurrency
            ),
            "concurrency_note": scheduler_policy.concurrency_note,
            "warning": scheduler_policy.warning,
        }

    async def run_interactive_synthesis_task(
        function: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        return await synthesis_scheduler.run_interactive(function, *args, **kwargs)

    async def run_stream_synthesis_task(
        function: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        return await synthesis_scheduler.run_stream(function, *args, **kwargs)

    async def synthesize_chunk_async(
        payload: SynthesisRequest, text: str
    ) -> RenderedChunk:
        return await run_interactive_synthesis_task(
            audio.synthesize_chunk, payload, text
        )

    async def synthesize_pcm_chunk_async(
        payload: SynthesisRequest, text: str
    ) -> tuple[audio.Float32Array, int, float]:
        return await run_interactive_synthesis_task(
            audio.synthesize_pcm_chunk, payload, text
        )

    async def synthesize_pcm_stream_chunk_async(
        payload: SynthesisRequest, text: str
    ) -> tuple[audio.Float32Array, int, float]:
        return await run_stream_synthesis_task(
            audio.synthesize_pcm_chunk, payload, text
        )

    if web_ui_enabled:

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(config.STATIC_DIR / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> FileResponse:
            return FileResponse(config.STATIC_DIR / "favicon.ico")

    @app.get("/api/public-config")
    async def public_config() -> JSONResponse:
        return JSONResponse(
            {
                "web_ui_enabled": web_ui_enabled,
                "auth_required": require_auth,
                "auth_scheme": "bearer" if require_auth else "none",
                "websocket_auth": (
                    "header-or-session-token" if require_auth else "none"
                ),
            }
        )

    @app.post("/api/ws-token")
    async def issue_websocket_token(request: Request) -> JSONResponse:
        ensure_http_auth(request)
        token, expires_in = issue_ws_session_token(extract_http_client_id(request))
        return JSONResponse(
            {
                "token": token,
                "token_type": "ws",
                "expires_in": expires_in,
            }
        )

    @app.get("/api/health")
    async def health(request: Request) -> JSONResponse:
        ensure_http_auth(request)
        missing: list[str] = []
        if not runtime.kokoro_runtime_available():
            missing.append("kokoro-onnx")
        if not config.MODEL_PATH.exists():
            missing.append(str(config.MODEL_PATH.name))
        if not config.VOICES_PATH.exists():
            missing.append(str(config.VOICES_PATH.name))

        runtime_status = runtime.get_runtime_status()
        gpu_usage = runtime.get_current_process_gpu_usage()

        return JSONResponse(
            {
                "ok": not missing and runtime_status.runtime_error is None,
                "missing": missing,
                "active_provider": runtime_status.active_providers[0]
                if runtime_status.active_providers
                else None,
                "active_providers": runtime_status.active_providers,
                "provider_fallback": runtime_status.provider_fallback,
                "provider_error": runtime_status.provider_error,
                "runtime_error": runtime_status.runtime_error,
                "gpu": {
                    "process_pid": gpu_usage.pid,
                    "process_group_id": gpu_usage.process_group_id,
                    "available": gpu_usage.available,
                    "process_vram_used_bytes": gpu_usage.used_bytes,
                    "process_vram_used_mb": gpu_usage.used_megabytes,
                    "process_group_vram_used_bytes": gpu_usage.group_used_bytes,
                    "process_group_vram_used_mb": gpu_usage.group_used_megabytes,
                    "process_group_member_pids": gpu_usage.group_member_pids,
                    "source": gpu_usage.source,
                    "error": gpu_usage.error,
                },
                "queue": queue_payload(),
            }
        )

    @app.get("/api/capabilities")
    async def capabilities(request: Request) -> JSONResponse:
        ensure_http_auth(request)
        runtime_status = runtime.get_runtime_status()
        available_formats = config.get_available_formats()
        return JSONResponse(
            {
                "model_path": str(config.MODEL_PATH),
                "voices_path": str(config.VOICES_PATH),
                "voices": runtime.load_voice_names(),
                "formats": available_formats,
                "opus_bitrates": config.OPUS_BITRATES,
                "wav_sample_rates": config.WAV_SAMPLE_RATES,
                "requested_provider": runtime_status.requested_provider,
                "attempted_providers": runtime_status.attempted_providers,
                "available_providers": runtime_status.available_providers,
                "active_provider": runtime_status.active_providers[0]
                if runtime_status.active_providers
                else None,
                "active_providers": runtime_status.active_providers,
                "provider_fallback": runtime_status.provider_fallback,
                "provider_error": runtime_status.provider_error,
                "runtime_error": runtime_status.runtime_error,
                "pitch_shifting": audio.ffmpeg_supports_rubberband(),
                "synthesis_workers": synthesis_workers,
                "synthesis_queue_limit": synthesis_queue_limit,
                "scheduler": scheduler_payload(),
                "max_pitch_semitones": config.MAX_PITCH_SHIFT_SEMITONES,
                "streaming": True,
                "websocket_streaming": runtime.websocket_runtime_available(),
                "queue": queue_payload(),
            }
        )

    @app.get("/v1/models", include_in_schema=False)
    async def openai_list_models(request: Request) -> JSONResponse:
        auth_error = openai_auth_error_for_request(request)
        if auth_error is not None:
            return auth_error
        payload: OpenAIModelListResponse = {
            "object": "list",
            "data": [openai_compat.openai_model_object()],
        }
        return JSONResponse(payload)

    @app.get("/v1/models/{model_id}", include_in_schema=False)
    async def openai_retrieve_model(model_id: str, request: Request) -> JSONResponse:
        auth_error = openai_auth_error_for_request(request)
        if auth_error is not None:
            return auth_error
        if model_id != config.OPENAI_COMPAT_MODEL:
            return openai_compat.openai_error_response(
                404,
                f"The model '{model_id}' does not exist.",
                error_type="invalid_request_error",
                code="model_not_found",
            )
        return JSONResponse(openai_compat.openai_model_object())

    @app.post("/v1/audio/speech", include_in_schema=False, response_model=None)
    async def openai_create_speech(
        payload: OpenAISpeechRequest, request: Request
    ) -> Response:
        auth_error = openai_auth_error_for_request(request)
        if auth_error is not None:
            return auth_error
        try:
            synth_request = openai_compat.build_openai_synthesis_request(payload)
        except SynthesisOverloadedError as exc:
            return openai_compat.openai_error_response(503, str(exc))
        except ValueError as exc:
            return openai_compat.openai_error_response(400, str(exc))
        except Exception:
            return openai_compat.openai_error_response(
                500, "Synthesis failed.", error_type="server_error"
            )

        if synth_request.format in {"wav", "pcm"}:
            chunks = runtime.split_text_into_chunks(
                synth_request.text, _OPENAI_STREAM_TARGET_CHARS
            )
            if not chunks:
                return openai_compat.openai_error_response(
                    400, "Enter text before generating."
                )

            stream_sample_rate = (
                int(synth_request.wav_sample_rate)
                if synth_request.wav_sample_rate != "native"
                else 24000
            )
            try:
                (
                    first_samples,
                    first_sample_rate,
                    _,
                ) = await synthesize_pcm_stream_chunk_async(synth_request, chunks[0])
            except SynthesisOverloadedError as exc:
                return openai_compat.openai_error_response(503, str(exc))
            except Exception:
                return openai_compat.openai_error_response(
                    500, "Synthesis failed.", error_type="server_error"
                )

            async def wav_stream() -> AsyncIterator[bytes]:
                if synth_request.format == "wav":
                    yield audio.wav_stream_header(first_sample_rate)
                yield audio.pcm16_bytes(first_samples)

                for chunk in chunks[1:]:
                    if await request.is_disconnected():
                        return
                    try:
                        (
                            samples,
                            _sample_rate,
                            _,
                        ) = await synthesize_pcm_stream_chunk_async(
                            synth_request, chunk
                        )
                    except Exception:
                        return
                    if await request.is_disconnected():
                        return
                    yield audio.pcm16_bytes(samples)

            headers = {
                "X-OpenAI-Compatible": config.OPENAI_COMPAT_MODEL,
                "X-Audio-Format": synth_request.format,
                "X-Sample-Rate": str(stream_sample_rate),
            }
            return StreamingResponse(
                wav_stream(),
                media_type="audio/pcm"
                if synth_request.format == "pcm"
                else "audio/wav",
                headers=headers,
            )

        try:
            rendered = await synthesize_chunk_async(synth_request, synth_request.text)
        except SynthesisOverloadedError as exc:
            return openai_compat.openai_error_response(503, str(exc))
        except Exception:
            return openai_compat.openai_error_response(
                500, "Synthesis failed.", error_type="server_error"
            )

        headers = {
            "X-OpenAI-Compatible": config.OPENAI_COMPAT_MODEL,
            "X-Audio-Format": synth_request.format,
            "X-Sample-Rate": str(rendered["sample_rate"]),
        }
        return StreamingResponse(
            io.BytesIO(rendered["audio_bytes"]),
            media_type=rendered["media_type"],
            headers=headers,
        )

    @app.post("/api/speak")
    async def speak(payload: SynthesisRequest, request: Request) -> StreamingResponse:
        ensure_http_auth(request)
        try:
            rendered = await synthesize_chunk_async(payload, payload.text)
        except SynthesisOverloadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Synthesis failed.") from exc

        headers = {
            "Content-Disposition": f'inline; filename="{rendered["filename"]}"',
            "X-Audio-Bytes": str(len(rendered["audio_bytes"])),
            "X-Audio-Format": payload.format,
            "X-Sample-Rate": str(rendered["sample_rate"]),
            "X-Audio-Duration": f"{rendered['duration_sec']:.6f}",
            "X-Opus-Bitrate": payload.opus_bitrate if payload.format == "opus" else "",
            "X-Wav-Sample-Rate": (
                str(rendered["sample_rate"]) if payload.format == "wav" else ""
            ),
        }
        return StreamingResponse(
            io.BytesIO(rendered["audio_bytes"]),
            media_type=rendered["media_type"],
            headers=headers,
        )

    @app.post("/api/chunk-plan")
    async def chunk_plan(
        payload: ChunkMetadataRequest, request: Request
    ) -> JSONResponse:
        ensure_http_auth(request)
        chunks = runtime.split_text_into_chunks(
            payload.text, payload.target_chunk_chars
        )
        chunk_payload: list[ChunkPlanEntry] = []
        for index, chunk in enumerate(chunks):
            entry: ChunkPlanEntry = {
                "index": index,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "sentence_count": len(re.findall(r"[.!?]+", chunk)),
            }
            if payload.include_text:
                entry["text"] = chunk
            chunk_payload.append(entry)

        return JSONResponse(
            {
                "chunks": chunk_payload,
                "count": len(chunks),
                "lengths": [len(chunk) for chunk in chunks],
                "target_chunk_chars": payload.target_chunk_chars,
            }
        )

    def build_chunk_event(
        payload: ChunkedSynthesisRequest, chunk: str, index: int, total_chunks: int
    ) -> tuple[dict[str, object], bytes]:
        started_at = time.perf_counter()
        rendered = audio.synthesize_chunk(payload, chunk)
        synth_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return (
            {
                "type": "chunk",
                "chunk_index": index,
                "total_chunks": total_chunks,
                "text": chunk,
                "pitch": payload.pitch,
                "bytes": len(rendered["audio_bytes"]),
                "sample_rate": rendered["sample_rate"],
                "duration_sec": rendered["duration_sec"],
                "synth_ms": synth_ms,
                "format": payload.format,
                "opus_bitrate": (
                    payload.opus_bitrate if payload.format == "opus" else None
                ),
                "wav_sample_rate": (
                    rendered["sample_rate"]
                    if payload.format in {"wav", "pcm"}
                    else None
                ),
                "mime_type": rendered["media_type"],
            },
            rendered["audio_bytes"],
        )

    async def build_chunk_event_async(
        payload: ChunkedSynthesisRequest, chunk: str, index: int, total_chunks: int
    ) -> tuple[dict[str, object], bytes]:
        return await run_stream_synthesis_task(
            build_chunk_event,
            payload,
            chunk,
            index,
            total_chunks,
        )

    @app.post("/api/speak-stream")
    async def speak_stream(
        payload: ChunkedSynthesisRequest, request: Request
    ) -> StreamingResponse:
        ensure_http_auth(request)
        chunks = runtime.split_text_into_chunks(
            payload.text, payload.target_chunk_chars
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="Enter text before generating.")

        async def stream() -> AsyncIterator[bytes]:
            total_chunks = len(chunks)
            if await request.is_disconnected():
                return
            yield (
                json.dumps(
                    {
                        "type": "meta",
                        "total_chunks": total_chunks,
                        "format": payload.format,
                        "opus_bitrate": payload.opus_bitrate
                        if payload.format == "opus"
                        else None,
                        "wav_sample_rate": payload.wav_sample_rate
                        if payload.format in {"wav", "pcm"}
                        else None,
                        "pitch": payload.pitch,
                        "target_chunk_chars": payload.target_chunk_chars,
                    }
                )
                + "\n"
            ).encode("utf-8")

            for index, chunk in enumerate(chunks):
                if await request.is_disconnected():
                    return
                try:
                    event, audio_bytes = await build_chunk_event_async(
                        payload, chunk, index, total_chunks
                    )
                except SynthesisOverloadedError:
                    if await request.is_disconnected():
                        return
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "detail": "Streaming synthesis failed.",
                                "chunk_index": index,
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                    return
                except Exception:
                    if await request.is_disconnected():
                        return
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "detail": "Streaming synthesis failed.",
                                "chunk_index": index,
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                    return

                event["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
                yield (json.dumps(event) + "\n").encode("utf-8")

            if await request.is_disconnected():
                return
            yield (
                json.dumps({"type": "done", "total_chunks": total_chunks}) + "\n"
            ).encode("utf-8")

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.websocket("/ws/speak-stream")
    async def ws_speak_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            raw_payload = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=ws_auth_handshake_timeout_sec,
            )
            payload_data = cast(object, json.loads(raw_payload))
            auth_error_detail = websocket_auth_error_detail(websocket, payload_data)
            if auth_error_detail is not None:
                await websocket.close(code=1008, reason=auth_error_detail)
                return
            if isinstance(payload_data, dict):
                payload_map = cast(dict[str, object], payload_data)
                payload_map.pop("api_key", None)
                payload_map.pop("ws_token", None)
            payload = ChunkedSynthesisRequest.model_validate(payload_data)
            chunks = runtime.split_text_into_chunks(
                payload.text, payload.target_chunk_chars
            )
            if not chunks:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "detail": "Enter text before generating."}
                    )
                )
                return

            total_chunks = len(chunks)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "meta",
                        "total_chunks": total_chunks,
                        "format": payload.format,
                        "opus_bitrate": payload.opus_bitrate
                        if payload.format == "opus"
                        else None,
                        "wav_sample_rate": payload.wav_sample_rate
                        if payload.format in {"wav", "pcm"}
                        else None,
                        "pitch": payload.pitch,
                        "target_chunk_chars": payload.target_chunk_chars,
                    }
                )
            )

            for index, chunk in enumerate(chunks):
                try:
                    event, audio_bytes = await build_chunk_event_async(
                        payload, chunk, index, total_chunks
                    )
                except SynthesisOverloadedError as exc:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "detail": str(exc),
                                "chunk_index": index,
                            }
                        )
                    )
                    return
                except Exception:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "detail": "Streaming synthesis failed.",
                                "chunk_index": index,
                            }
                        )
                    )
                    return
                await websocket.send_text(json.dumps(event))
                await websocket.send_bytes(audio_bytes)

            await websocket.send_text(
                json.dumps({"type": "done", "total_chunks": total_chunks})
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1008, reason="Authentication timeout.")
            return
        except WebSocketDisconnect:
            return
        except Exception:
            await websocket.send_text(
                json.dumps({"type": "error", "detail": "Streaming synthesis failed."})
            )
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app
