from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("soundfile is required to run this app.") from exc

try:
    from kokoro_onnx import Kokoro as KokoroRuntime
except ImportError:  # pragma: no cover
    KokoroRuntime = None


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
MODELS_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("KOKORO_MODEL_PATH", MODELS_DIR / "kokoro-v1.0.onnx"))
VOICES_PATH = Path(os.getenv("KOKORO_VOICES_PATH", MODELS_DIR / "voices-v1.0.bin"))

DEFAULT_VOICES = ["af_heart"]
OPUS_BITRATES: list[str] = ["16k", "24k", "32k", "48k"]
WAV_SAMPLE_RATES: list[str] = ["native", "16000", "22050", "24000", "44100", "48000"]
MAX_PITCH_SHIFT_SEMITONES = 6.0
OPENAI_COMPAT_MODEL = "kokoro"
OPENAI_COMPAT_OWNER = "kokoro-webui"


class RenderedChunk(TypedDict):
    audio_bytes: bytes
    media_type: str
    filename: str
    sample_rate: int
    duration_sec: float


class ChunkPlanEntryBase(TypedDict):
    index: int
    char_count: int
    word_count: int
    sentence_count: int


class ChunkPlanEntry(ChunkPlanEntryBase, total=False):
    text: str


class KokoroEngine(Protocol):
    def create(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        lang: str,
    ) -> tuple[np.ndarray, int]: ...


class VoiceArchive(Protocol):
    files: list[str]


class OpenAIErrorBody(TypedDict):
    message: str
    type: str
    param: str | None
    code: str | None


class OpenAIErrorResponse(TypedDict):
    error: OpenAIErrorBody


class OpenAIModelObject(TypedDict):
    id: str
    object: str
    created: int
    owned_by: str


class OpenAIModelListResponse(TypedDict):
    object: str
    data: list[OpenAIModelObject]


class OpenAIVoiceRef(BaseModel):
    id: str = Field(min_length=1, max_length=64)


class OpenAISpeechRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1, max_length=4096)
    voice: str | OpenAIVoiceRef
    response_format: Literal["wav", "opus"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=4096)
    stream_format: Literal["audio", "sse"] | None = None


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2500)
    voice: str = Field(default="af_heart", min_length=1, max_length=64)
    speed: float = Field(default=1.0, ge=0.5, le=1.8)
    pitch: float = Field(
        default=0.0,
        ge=-MAX_PITCH_SHIFT_SEMITONES,
        le=MAX_PITCH_SHIFT_SEMITONES,
    )
    lang: Literal["en-us", "en-gb", "fr-fr", "ja", "ko", "cmn"] = "en-us"
    format: Literal["wav", "opus"] = "wav"
    opus_bitrate: Literal["16k", "24k", "32k", "48k"] = "32k"
    wav_sample_rate: Literal["native", "16000", "22050", "24000", "44100", "48000"] = (
        "native"
    )


class ChunkedSynthesisRequest(SynthesisRequest):
    target_chunk_chars: int = Field(default=360, ge=80, le=2000)


class ChunkMetadataRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2500)
    target_chunk_chars: int = Field(default=360, ge=80, le=2000)
    include_text: bool = False


@lru_cache(maxsize=1)
def get_tts() -> KokoroEngine:
    if KokoroRuntime is None:
        raise RuntimeError(
            "kokoro-onnx is not installed. Install dependencies with `uv sync`."
        )
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not VOICES_PATH.exists():
        raise RuntimeError(f"Voice file not found: {VOICES_PATH}")
    runtime_factory = cast(Callable[[str, str], KokoroEngine], KokoroRuntime)
    return runtime_factory(str(MODEL_PATH), str(VOICES_PATH))


def _normalize_voice_names(candidates: object) -> list[str]:
    if isinstance(candidates, Mapping):
        names = cast(Iterable[object], candidates.keys())
    elif isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes)):
        names = candidates
    else:
        return []

    normalized_names: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if stripped:
            normalized_names.add(stripped)
    return sorted(normalized_names)


def load_voice_names() -> list[str]:
    try:
        tts = get_tts()
        for attr_name in ("voices", "voice_names", "available_voices"):
            voices = getattr(tts, attr_name, None)
            normalized = _normalize_voice_names(voices)
            if normalized:
                return normalized
    except Exception:
        pass

    if VOICES_PATH.exists():
        try:
            with np.load(VOICES_PATH, allow_pickle=False) as raw_voice_data:  # pyright: ignore[reportAny]
                voice_data = cast(VoiceArchive, raw_voice_data)
                normalized = _normalize_voice_names(voice_data.files)
                if normalized:
                    return normalized
        except Exception:
            pass

    return DEFAULT_VOICES


def split_text_into_chunks(text: str, target_chunk_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentence_split = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [segment.strip() for segment in sentence_split if segment.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > target_chunk_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    wav = io.BytesIO()
    sf_write = cast(Callable[..., None], sf.write)
    sf_write(wav, samples, sample_rate, format="WAV")
    return wav.getvalue()


def encode_opus(samples: np.ndarray, sample_rate: int, bitrate: str) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for Opus output but is not installed.")

    pcm = np.asarray(samples, dtype=np.float32)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-application",
        "voip",
        "-f",
        "ogg",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=pcm.tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Opus encoding failed: {error or 'ffmpeg exited non-zero.'}"
        )
    return result.stdout


@lru_cache(maxsize=1)
def ffmpeg_supports_rubberband() -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    result = subprocess.run(
        ["ffmpeg", "-filters"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0 and " rubberband " in result.stdout


def resample_linear(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise RuntimeError("Sample rate must be positive for resampling.")
    if source_rate == target_rate:
        return samples

    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source.size == 0:
        return source

    target_length = max(1, int(round(source.size * target_rate / source_rate)))
    source_positions = np.linspace(
        0.0, source.size - 1, num=source.size, dtype=np.float64
    )
    target_positions = np.linspace(
        0.0, source.size - 1, num=target_length, dtype=np.float64
    )
    resampled = np.interp(target_positions, source_positions, source)
    return resampled.astype(np.float32, copy=False)


def pitch_shift_samples(
    samples: np.ndarray, sample_rate: int, pitch_semitones: float
) -> np.ndarray:
    if pitch_semitones == 0:
        return np.asarray(samples, dtype=np.float32).reshape(-1)
    if not ffmpeg_supports_rubberband():
        raise RuntimeError(
            "Backend pitch shifting requires ffmpeg with the rubberband filter."
        )

    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    pitch_ratio = 2 ** (pitch_semitones / 12)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-af",
        f"rubberband=pitch={pitch_ratio:.8f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=pcm.tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Pitch shifting failed: {error or 'ffmpeg exited non-zero.'}"
        )

    shifted = np.frombuffer(result.stdout, dtype=np.float32)
    if shifted.size == 0:
        raise RuntimeError("Pitch shifting failed: ffmpeg returned empty audio.")
    return np.clip(shifted, -1.0, 1.0).astype(np.float32, copy=False)


def synthesize_chunk(payload: SynthesisRequest, text: str) -> RenderedChunk:
    tts = get_tts()
    samples, sample_rate = tts.create(
        text.strip(),
        voice=payload.voice.strip(),
        speed=payload.speed,
        lang=payload.lang,
    )
    samples = pitch_shift_samples(samples, sample_rate, payload.pitch)

    if payload.format == "opus":
        audio_bytes = encode_opus(samples, sample_rate, payload.opus_bitrate)
        media_type = "audio/ogg"
        filename = "kokoro-output.ogg"
    else:
        if payload.wav_sample_rate != "native":
            target_rate = int(payload.wav_sample_rate)
            samples = resample_linear(samples, sample_rate, target_rate)
            sample_rate = target_rate
        audio_bytes = encode_wav(samples, sample_rate)
        media_type = "audio/wav"
        filename = "kokoro-output.wav"

    duration_sec = float(len(samples) / sample_rate) if sample_rate else 0.0
    return {
        "audio_bytes": audio_bytes,
        "media_type": media_type,
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
    }


def websocket_runtime_available() -> bool:
    return bool(importlib.util.find_spec("websockets")) or bool(
        importlib.util.find_spec("wsproto")
    )


def openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    payload: OpenAIErrorResponse = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


def openai_model_object() -> OpenAIModelObject:
    return {
        "id": OPENAI_COMPAT_MODEL,
        "object": "model",
        "created": 0,
        "owned_by": OPENAI_COMPAT_OWNER,
    }


def resolve_openai_voice_id(voice: str | OpenAIVoiceRef) -> str:
    if isinstance(voice, OpenAIVoiceRef):
        return voice.id.strip()
    return voice.strip()


def parse_openai_voice_and_pitch(voice: str) -> tuple[str, float]:
    trimmed = voice.strip()
    match = re.fullmatch(
        r"(?P<voice>[A-Za-z0-9_]+?)(?P<pitch>[+-](?:\d+(?:\.\d+)?|\.\d+))?",
        trimmed,
    )
    if not match:
        raise ValueError(
            "voice must be a Kokoro voice id or a voice id suffixed like af_heart+2.0."
        )

    voice_id = match.group("voice")
    pitch_token = match.group("pitch")
    pitch = float(pitch_token) if pitch_token is not None else 0.0
    if pitch < -MAX_PITCH_SHIFT_SEMITONES or pitch > MAX_PITCH_SHIFT_SEMITONES:
        raise ValueError(
            f"voice pitch suffix must be between {-MAX_PITCH_SHIFT_SEMITONES:.1f} and +{MAX_PITCH_SHIFT_SEMITONES:.1f} semitones."
        )
    return voice_id, pitch


def build_openai_synthesis_request(payload: OpenAISpeechRequest) -> SynthesisRequest:
    if payload.stream_format == "sse":
        raise ValueError("stream_format 'sse' is not supported by this server.")
    if payload.speed < 0.5 or payload.speed > 1.8:
        raise ValueError("speed must be between 0.5 and 1.8 for Kokoro.")

    voice, pitch = parse_openai_voice_and_pitch(resolve_openai_voice_id(payload.voice))
    return SynthesisRequest(
        text=payload.input,
        voice=voice,
        speed=payload.speed,
        pitch=pitch,
        format=payload.response_format,
    )


app = FastAPI(title="Kokoro WebUI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/api/health")
async def health() -> JSONResponse:
    missing: list[str] = []
    if KokoroRuntime is None:
        missing.append("kokoro-onnx")
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH.name))
    if not VOICES_PATH.exists():
        missing.append(str(VOICES_PATH.name))

    return JSONResponse(
        {
            "ok": not missing,
            "missing": missing,
            "model_path": str(MODEL_PATH),
            "voices_path": str(VOICES_PATH),
            "voices": load_voice_names(),
            "formats": ["wav", "opus"],
            "opus_bitrates": OPUS_BITRATES,
            "wav_sample_rates": WAV_SAMPLE_RATES,
            "pitch_shifting": ffmpeg_supports_rubberband(),
            "max_pitch_semitones": MAX_PITCH_SHIFT_SEMITONES,
            "streaming": True,
            "websocket_streaming": websocket_runtime_available(),
        }
    )


@app.get("/v1/models", include_in_schema=False)
async def openai_list_models() -> JSONResponse:
    payload: OpenAIModelListResponse = {
        "object": "list",
        "data": [openai_model_object()],
    }
    return JSONResponse(payload)


@app.get("/v1/models/{model_id}", include_in_schema=False)
async def openai_retrieve_model(model_id: str) -> JSONResponse:
    if model_id != OPENAI_COMPAT_MODEL:
        return openai_error_response(
            404,
            f"The model '{model_id}' does not exist.",
            error_type="invalid_request_error",
            code="model_not_found",
        )
    return JSONResponse(openai_model_object())


@app.post("/v1/audio/speech", include_in_schema=False, response_model=None)
async def openai_create_speech(
    payload: OpenAISpeechRequest,
) -> Response:
    try:
        synth_request = build_openai_synthesis_request(payload)
        rendered = synthesize_chunk(synth_request, synth_request.text)
    except ValueError as exc:
        return openai_error_response(400, str(exc))
    except Exception as exc:
        return openai_error_response(400, str(exc))

    headers = {
        "X-OpenAI-Compatible": OPENAI_COMPAT_MODEL,
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
        rendered = synthesize_chunk(payload, payload.text)
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
    chunks = split_text_into_chunks(payload.text, payload.target_chunk_chars)
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
    rendered = synthesize_chunk(payload, chunk)
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
        "wav_sample_rate": rendered["sample_rate"] if payload.format == "wav" else None,
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
    chunks = split_text_into_chunks(payload.text, payload.target_chunk_chars)
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
        chunks = split_text_into_chunks(payload.text, payload.target_chunk_chars)
        if not chunks:
            await websocket.send_text(
                json.dumps({"type": "error", "detail": "Enter text before generating."})
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
