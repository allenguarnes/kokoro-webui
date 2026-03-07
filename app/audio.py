from __future__ import annotations

import io
import shutil
import subprocess
from collections.abc import Callable
from functools import lru_cache
from typing import cast

import numpy as np
import numpy.typing as npt

from app.config import get_ffmpeg_timeout_seconds
from app.runtime import get_tts
from app.schemas import RenderedChunk, SynthesisRequest

try:
    import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("soundfile is required to run this app.") from exc


Float32Array = npt.NDArray[np.float32]


def run_ffmpeg_bytes(
    command: list[str], *, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    timeout = get_ffmpeg_timeout_seconds()
    try:
        return subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            check=False,
            text=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout:.1f}s while processing audio."
        ) from exc


def run_ffmpeg_text(command: list[str]) -> subprocess.CompletedProcess[str]:
    timeout = get_ffmpeg_timeout_seconds()
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout:.1f}s while processing audio."
        ) from exc


def encode_wav(samples: Float32Array, sample_rate: int) -> bytes:
    wav = io.BytesIO()
    sf_write = cast(Callable[..., None], sf.write)
    sf_write(wav, samples, sample_rate, format="WAV")
    return wav.getvalue()


def encode_opus(samples: Float32Array, sample_rate: int, bitrate: str) -> bytes:
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
    result = run_ffmpeg_bytes(command, input_bytes=pcm.tobytes())
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

    result = run_ffmpeg_text(["ffmpeg", "-filters"])
    return result.returncode == 0 and " rubberband " in result.stdout


def resample_linear(
    samples: Float32Array, source_rate: int, target_rate: int
) -> Float32Array:
    if source_rate <= 0 or target_rate <= 0:
        raise RuntimeError("Sample rate must be positive for resampling.")
    if source_rate == target_rate:
        return samples

    source = cast(Float32Array, np.asarray(samples, dtype=np.float32).reshape(-1))
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
    samples: Float32Array, sample_rate: int, pitch_semitones: float
) -> Float32Array:
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
    result = run_ffmpeg_bytes(command, input_bytes=pcm.tobytes())
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Pitch shifting failed: {error or 'ffmpeg exited non-zero.'}"
        )

    shifted = np.frombuffer(result.stdout, dtype=np.float32)
    if shifted.size == 0:
        raise RuntimeError("Pitch shifting failed: ffmpeg returned empty audio.")
    clipped = np.clip(shifted, -1.0, 1.0)
    return clipped.astype(np.float32, copy=False)


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
