from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import cast

import numpy as np

from app.config import DEFAULT_VOICES, MODEL_PATH, VOICES_PATH
from app.schemas import KokoroEngine, VoiceArchive

KokoroRuntime: Callable[[str, str], KokoroEngine] | None

try:
    from kokoro_onnx import Kokoro as KokoroRuntime
except ImportError:  # pragma: no cover
    KokoroRuntime = None


def kokoro_runtime_available() -> bool:
    return KokoroRuntime is not None


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
    return KokoroRuntime(str(MODEL_PATH), str(VOICES_PATH))


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


def websocket_runtime_available() -> bool:
    return bool(importlib.util.find_spec("websockets")) or bool(
        importlib.util.find_spec("wsproto")
    )
