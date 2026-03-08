# pyright: reportUnusedFunction=false, reportUnusedCallResult=false
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import ParamSpec, TypeVar

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


def create_app() -> FastAPI:
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
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

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

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(config.STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(config.STATIC_DIR / "favicon.ico")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        missing: list[str] = []
        if not runtime.kokoro_runtime_available():
            missing.append("kokoro-onnx")
        if not config.MODEL_PATH.exists():
            missing.append(str(config.MODEL_PATH.name))
        if not config.VOICES_PATH.exists():
            missing.append(str(config.VOICES_PATH.name))

        runtime_status = runtime.get_runtime_status()

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
                "queue": queue_payload(),
            }
        )

    @app.get("/api/capabilities")
    async def capabilities() -> JSONResponse:
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
    async def openai_list_models() -> JSONResponse:
        payload: OpenAIModelListResponse = {
            "object": "list",
            "data": [openai_compat.openai_model_object()],
        }
        return JSONResponse(payload)

    @app.get("/v1/models/{model_id}", include_in_schema=False)
    async def openai_retrieve_model(model_id: str) -> JSONResponse:
        if model_id != config.OPENAI_COMPAT_MODEL:
            return openai_compat.openai_error_response(
                404,
                f"The model '{model_id}' does not exist.",
                error_type="invalid_request_error",
                code="model_not_found",
            )
        return JSONResponse(openai_compat.openai_model_object())

    @app.post("/v1/audio/speech", include_in_schema=False, response_model=None)
    async def openai_create_speech(payload: OpenAISpeechRequest) -> Response:
        try:
            synth_request = openai_compat.build_openai_synthesis_request(payload)
            rendered = await synthesize_chunk_async(synth_request, synth_request.text)
        except SynthesisOverloadedError as exc:
            return openai_compat.openai_error_response(503, str(exc))
        except ValueError as exc:
            return openai_compat.openai_error_response(400, str(exc))
        except Exception as exc:
            return openai_compat.openai_error_response(400, str(exc))

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
    async def speak(payload: SynthesisRequest) -> StreamingResponse:
        try:
            rendered = await synthesize_chunk_async(payload, payload.text)
        except SynthesisOverloadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    async def chunk_plan(payload: ChunkMetadataRequest) -> JSONResponse:
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
                    rendered["sample_rate"] if payload.format == "wav" else None
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

    async def watch_websocket_disconnect(
        websocket: WebSocket, disconnected: asyncio.Event
    ) -> None:
        try:
            while not disconnected.is_set():
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    disconnected.set()
                    return
        except WebSocketDisconnect:
            disconnected.set()
        except RuntimeError:
            disconnected.set()

    @app.post("/api/speak-stream")
    async def speak_stream(
        payload: ChunkedSynthesisRequest, request: Request
    ) -> StreamingResponse:
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
                        if payload.format == "wav"
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
                except SynthesisOverloadedError as exc:
                    if await request.is_disconnected():
                        return
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "detail": str(exc),
                                "chunk_index": index,
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                    return
                except Exception as exc:
                    if await request.is_disconnected():
                        return
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "detail": str(exc),
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
        disconnect_event = asyncio.Event()
        disconnect_task: asyncio.Task[None] | None = None
        try:
            raw_payload = await websocket.receive_text()
            payload = ChunkedSynthesisRequest.model_validate_json(raw_payload)
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

            disconnect_task = asyncio.create_task(
                watch_websocket_disconnect(websocket, disconnect_event)
            )
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
                        if payload.format == "wav"
                        else None,
                        "pitch": payload.pitch,
                        "target_chunk_chars": payload.target_chunk_chars,
                    }
                )
            )

            for index, chunk in enumerate(chunks):
                if disconnect_event.is_set():
                    return
                try:
                    event, audio_bytes = await build_chunk_event_async(
                        payload, chunk, index, total_chunks
                    )
                except SynthesisOverloadedError as exc:
                    if disconnect_event.is_set():
                        return
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
                except Exception as exc:
                    if disconnect_event.is_set():
                        return
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
                if disconnect_event.is_set():
                    return
                await websocket.send_text(json.dumps(event))
                if disconnect_event.is_set():
                    return
                await websocket.send_bytes(audio_bytes)

            if disconnect_event.is_set():
                return
            await websocket.send_text(
                json.dumps({"type": "done", "total_chunks": total_chunks})
            )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            if disconnect_event.is_set():
                return
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                try:
                    await disconnect_task
                except asyncio.CancelledError:
                    pass
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app
