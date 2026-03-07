# pyright: reportUnusedFunction=false, reportUnusedCallResult=false
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import audio, config, openai_compat, runtime
from app.schemas import (
    ChunkedSynthesisRequest,
    ChunkMetadataRequest,
    ChunkPlanEntry,
    OpenAIModelListResponse,
    OpenAISpeechRequest,
    SynthesisRequest,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Kokoro WebUI")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

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
                "model_path": str(config.MODEL_PATH),
                "voices_path": str(config.VOICES_PATH),
                "voices": runtime.load_voice_names(),
                "formats": ["wav", "opus"],
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
                "max_pitch_semitones": config.MAX_PITCH_SHIFT_SEMITONES,
                "streaming": True,
                "websocket_streaming": runtime.websocket_runtime_available(),
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
            rendered = audio.synthesize_chunk(synth_request, synth_request.text)
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
            rendered = audio.synthesize_chunk(payload, payload.text)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'inline; filename="{rendered["filename"]}"',
            "X-Audio-Bytes": str(len(rendered["audio_bytes"])),
            "X-Audio-Format": payload.format,
            "X-Sample-Rate": str(rendered["sample_rate"]),
            "X-Audio-Duration": f"{rendered['duration_sec']:.6f}",
            "X-Opus-Bitrate": payload.opus_bitrate if payload.format == "opus" else "",
            "X-Wav-Sample-Rate": str(rendered["sample_rate"])
            if payload.format == "wav"
            else "",
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
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        rendered = audio.synthesize_chunk(payload, chunk)
        synth_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {
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
            "opus_bitrate": payload.opus_bitrate if payload.format == "opus" else None,
            "wav_sample_rate": rendered["sample_rate"]
            if payload.format == "wav"
            else None,
            "mime_type": rendered["media_type"],
            "audio_base64": base64.b64encode(rendered["audio_bytes"]).decode("ascii"),
        }

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
                    event = build_chunk_event(payload, chunk, index, total_chunks)
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
                    event = build_chunk_event(payload, chunk, index, total_chunks)
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
