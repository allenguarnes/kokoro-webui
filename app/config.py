from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = ROOT / ".env"
ProviderMode = Literal["auto", "cpu", "cuda"]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ[key] = value


load_env_file(ENV_FILE_PATH)

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
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_FFMPEG_TIMEOUT_SEC = 20.0
DEFAULT_RUNTIME_PROVIDER: ProviderMode = "auto"


def parse_bool_env(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_server_host() -> str:
    return os.getenv("KOKORO_HOST", DEFAULT_SERVER_HOST).strip() or DEFAULT_SERVER_HOST


def get_server_port() -> int:
    raw_port = os.getenv("KOKORO_PORT", str(DEFAULT_SERVER_PORT)).strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"Invalid KOKORO_PORT value: {raw_port!r}") from exc
    if port < 1 or port > 65535:
        raise RuntimeError(f"KOKORO_PORT must be between 1 and 65535, got {port}.")
    return port


def get_ffmpeg_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "KOKORO_FFMPEG_TIMEOUT_SEC", str(DEFAULT_FFMPEG_TIMEOUT_SEC)
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_FFMPEG_TIMEOUT_SEC must be a positive number, got {raw_timeout!r}."
        ) from exc
    if timeout <= 0:
        raise RuntimeError(
            f"KOKORO_FFMPEG_TIMEOUT_SEC must be greater than 0, got {timeout}."
        )
    return timeout


def get_synthesis_workers(*, default: int) -> int:
    raw_workers = os.getenv("KOKORO_SYNTH_WORKERS", str(default)).strip()
    try:
        workers = int(raw_workers)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_SYNTH_WORKERS must be a positive integer, got {raw_workers!r}."
        ) from exc
    if workers < 1:
        raise RuntimeError(
            f"KOKORO_SYNTH_WORKERS must be greater than 0, got {workers}."
        )
    return workers


def get_runtime_provider_mode() -> ProviderMode:
    raw_provider = (
        os.getenv("KOKORO_PROVIDER", DEFAULT_RUNTIME_PROVIDER).strip().lower()
    )
    if raw_provider in {"auto", "cpu", "cuda"}:
        return cast(ProviderMode, raw_provider)
    raise RuntimeError(
        f"KOKORO_PROVIDER must be one of 'auto', 'cpu', or 'cuda', got {raw_provider!r}."
    )


def get_runtime_provider_strict() -> bool:
    return parse_bool_env(os.getenv("KOKORO_STRICT_PROVIDER"), default=False)


def get_runtime_cuda_lib_dir() -> str | None:
    value = os.getenv("KOKORO_CUDA_LIB_DIR")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
