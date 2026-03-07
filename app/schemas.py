from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from pydantic import BaseModel, Field

from app.config import MAX_PITCH_SHIFT_SEMITONES


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
