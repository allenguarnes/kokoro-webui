from __future__ import annotations

import re

from fastapi.responses import JSONResponse

from app.config import (
    MAX_PITCH_SHIFT_SEMITONES,
    OPENAI_COMPAT_MODEL,
    OPENAI_COMPAT_OWNER,
)
from app.schemas import (
    OpenAIErrorResponse,
    OpenAIModelObject,
    OpenAISpeechRequest,
    OpenAIVoiceRef,
    SynthesisRequest,
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
