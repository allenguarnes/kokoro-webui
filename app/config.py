from __future__ import annotations

import ipaddress
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
ALL_AUDIO_FORMATS: tuple[str, ...] = ("wav", "opus", "pcm")
OPUS_BITRATES: list[str] = ["16k", "24k", "32k", "48k"]
WAV_SAMPLE_RATES: list[str] = ["native", "16000", "22050", "24000", "44100", "48000"]
MAX_PITCH_SHIFT_SEMITONES = 6.0
OPENAI_COMPAT_MODEL = "kokoro"
OPENAI_COMPAT_OWNER = "kokoro-webui"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_FFMPEG_TIMEOUT_SEC = 20.0
DEFAULT_RUNTIME_PROVIDER: ProviderMode = "auto"
DEFAULT_ENABLE_WEB_UI = True
DEFAULT_AUTH_FAILURE_LIMIT = 5
DEFAULT_AUTH_FAILURE_WINDOW_SEC = 60.0
DEFAULT_AUTH_FAILURE_MAX_BUCKETS = 4096
DEFAULT_TRUST_PROXY_HEADERS = False
DEFAULT_WS_AUTH_HANDSHAKE_TIMEOUT_SEC = 5.0
DEFAULT_WS_SESSION_TOKEN_TTL_SEC = 30.0
DEFAULT_WS_SESSION_TOKEN_MAX_TOKENS = 1024
DEFAULT_RUNTIME_IDLE_UNLOAD_SEC = 0.0


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


def get_web_ui_enabled() -> bool:
    return parse_bool_env(
        os.getenv("KOKORO_ENABLE_WEB_UI"),
        default=DEFAULT_ENABLE_WEB_UI,
    )


def get_require_auth() -> bool:
    return parse_bool_env(os.getenv("KOKORO_REQUIRE_AUTH"), default=False)


def get_api_key() -> str | None:
    value = os.getenv("KOKORO_API_KEY")
    cleaned = value.strip() if value is not None else ""
    if not cleaned:
        if get_require_auth():
            raise RuntimeError("KOKORO_API_KEY must be set when KOKORO_REQUIRE_AUTH=1.")
        return None
    return cleaned


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv("KOKORO_ALLOWED_ORIGINS")
    if raw_origins is None or not raw_origins.strip():
        return []

    origins: list[str] = []
    for token in raw_origins.split(","):
        value = token.strip()
        if not value:
            continue
        if value not in origins:
            origins.append(value)
    return origins


def get_auth_failure_limit() -> int:
    raw_limit = os.getenv(
        "KOKORO_AUTH_FAILURE_LIMIT", str(DEFAULT_AUTH_FAILURE_LIMIT)
    ).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_LIMIT must be a non-negative integer, got {raw_limit!r}."
        ) from exc
    if limit < 0:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_LIMIT must be 0 or greater, got {limit}."
        )
    return limit


def get_auth_failure_window_seconds() -> float:
    raw_window = os.getenv(
        "KOKORO_AUTH_FAILURE_WINDOW_SEC", str(DEFAULT_AUTH_FAILURE_WINDOW_SEC)
    ).strip()
    try:
        window = float(raw_window)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_WINDOW_SEC must be a positive number, got {raw_window!r}."
        ) from exc
    if window <= 0:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_WINDOW_SEC must be greater than 0, got {window}."
        )
    return window


def get_auth_failure_max_buckets() -> int:
    raw_limit = os.getenv(
        "KOKORO_AUTH_FAILURE_MAX_BUCKETS", str(DEFAULT_AUTH_FAILURE_MAX_BUCKETS)
    ).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_MAX_BUCKETS must be a positive integer, got {raw_limit!r}."
        ) from exc
    if limit < 1:
        raise RuntimeError(
            f"KOKORO_AUTH_FAILURE_MAX_BUCKETS must be greater than 0, got {limit}."
        )
    return limit


def get_trust_proxy_headers() -> bool:
    return parse_bool_env(
        os.getenv("KOKORO_TRUST_PROXY_HEADERS"),
        default=DEFAULT_TRUST_PROXY_HEADERS,
    )


def get_trusted_proxy_ips() -> frozenset[str]:
    raw = os.getenv("KOKORO_TRUSTED_PROXY_IPS", "").strip()
    if not raw:
        return frozenset()
    ips: set[str] = set()
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        try:
            ipaddress.ip_address(cleaned)
            ips.add(cleaned)
        except ValueError:
            try:
                ipaddress.ip_network(cleaned, strict=False)
                ips.add(cleaned)
            except ValueError:
                raise RuntimeError(
                    f"Invalid IP address or network in KOKORO_TRUSTED_PROXY_IPS: {cleaned!r}"
                )
    return frozenset(ips)


def validate_proxy_config() -> None:
    if get_trust_proxy_headers() and not get_trusted_proxy_ips():
        raise RuntimeError(
            "KOKORO_TRUST_PROXY_HEADERS=1 requires KOKORO_TRUSTED_PROXY_IPS to be set. "
            "Configure the trusted proxy IPs (e.g., 127.0.0.1) to prevent header spoofing."
        )


def get_websocket_auth_handshake_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "KOKORO_WS_AUTH_HANDSHAKE_TIMEOUT_SEC",
        str(DEFAULT_WS_AUTH_HANDSHAKE_TIMEOUT_SEC),
    ).strip()
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_WS_AUTH_HANDSHAKE_TIMEOUT_SEC must be a positive number, got {raw_timeout!r}."
        ) from exc
    if timeout <= 0:
        raise RuntimeError(
            f"KOKORO_WS_AUTH_HANDSHAKE_TIMEOUT_SEC must be greater than 0, got {timeout}."
        )
    return timeout


def get_websocket_session_token_ttl_seconds() -> float:
    raw_ttl = os.getenv(
        "KOKORO_WS_SESSION_TOKEN_TTL_SEC",
        str(DEFAULT_WS_SESSION_TOKEN_TTL_SEC),
    ).strip()
    try:
        ttl = float(raw_ttl)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_WS_SESSION_TOKEN_TTL_SEC must be a positive number, got {raw_ttl!r}."
        ) from exc
    if ttl <= 0:
        raise RuntimeError(
            f"KOKORO_WS_SESSION_TOKEN_TTL_SEC must be greater than 0, got {ttl}."
        )
    return ttl


def get_websocket_session_token_max_tokens() -> int:
    raw_limit = os.getenv(
        "KOKORO_WS_SESSION_TOKEN_MAX_TOKENS",
        str(DEFAULT_WS_SESSION_TOKEN_MAX_TOKENS),
    ).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_WS_SESSION_TOKEN_MAX_TOKENS must be a positive integer, got {raw_limit!r}."
        ) from exc
    if limit < 1:
        raise RuntimeError(
            f"KOKORO_WS_SESSION_TOKEN_MAX_TOKENS must be greater than 0, got {limit}."
        )
    return limit


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


def get_synthesis_queue_limit(*, default: int) -> int:
    raw_limit = os.getenv("KOKORO_SYNTH_QUEUE", str(default)).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(
            f"KOKORO_SYNTH_QUEUE must be a non-negative integer, got {raw_limit!r}."
        ) from exc
    if limit < 0:
        raise RuntimeError(f"KOKORO_SYNTH_QUEUE must be 0 or greater, got {limit}.")
    return limit


def get_available_formats() -> list[str]:
    raw_formats = os.getenv("KOKORO_FORMATS")
    if raw_formats is None or not raw_formats.strip():
        return list(ALL_AUDIO_FORMATS)

    parsed_formats: list[str] = []
    invalid_formats: list[str] = []
    for token in raw_formats.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value not in ALL_AUDIO_FORMATS:
            invalid_formats.append(value)
            continue
        if value not in parsed_formats:
            parsed_formats.append(value)

    if invalid_formats:
        allowed = ", ".join(ALL_AUDIO_FORMATS)
        invalid = ", ".join(invalid_formats)
        raise RuntimeError(
            f"KOKORO_FORMATS contains unsupported values: {invalid}. Allowed values: {allowed}."
        )
    if not parsed_formats:
        raise RuntimeError(
            "KOKORO_FORMATS must include at least one supported format when set."
        )
    return parsed_formats


def get_allow_experimental_cuda_concurrency() -> bool:
    return parse_bool_env(
        os.getenv("KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY"),
        default=False,
    )


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


def get_runtime_idle_unload_seconds() -> float:
    raw_timeout = os.getenv(
        "KOKORO_RUNTIME_IDLE_UNLOAD_SEC", str(DEFAULT_RUNTIME_IDLE_UNLOAD_SEC)
    ).strip()
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise RuntimeError(
            "KOKORO_RUNTIME_IDLE_UNLOAD_SEC must be a non-negative number, "
            + f"got {raw_timeout!r}."
        ) from exc
    if timeout < 0:
        raise RuntimeError(
            "KOKORO_RUNTIME_IDLE_UNLOAD_SEC must be 0 or greater, " + f"got {timeout}."
        )
    return timeout


def get_runtime_cuda_lib_dir() -> str | None:
    value = os.getenv("KOKORO_CUDA_LIB_DIR")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
