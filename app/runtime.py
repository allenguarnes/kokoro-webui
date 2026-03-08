from __future__ import annotations

import importlib
import importlib.util
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol, cast

import numpy as np

from app.config import (
    DEFAULT_VOICES,
    MODEL_PATH,
    VOICES_PATH,
    get_runtime_cuda_lib_dir,
    get_runtime_provider_mode,
    get_runtime_provider_strict,
)
from app.schemas import KokoroEngine, VoiceArchive

ProviderMode = Literal["auto", "cpu", "cuda"]


def _prepend_env_path(name: str, path: str) -> None:
    cleaned = path.strip()
    if not cleaned:
        return

    current_parts = [
        part for part in os.environ.get(name, "").split(os.pathsep) if part
    ]
    if cleaned in current_parts:
        return
    os.environ[name] = os.pathsep.join([cleaned, *current_parts])


cuda_lib_dir = get_runtime_cuda_lib_dir()
if cuda_lib_dir:
    _prepend_env_path("LD_LIBRARY_PATH", cuda_lib_dir)

KokoroRuntime: Callable[[str, str], KokoroEngine] | None

try:
    from kokoro_onnx import Kokoro as KokoroRuntime
except ImportError:  # pragma: no cover
    KokoroRuntime = None

try:
    import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
except ImportError:  # pragma: no cover
    ort = None


class OnnxSession(Protocol):
    def get_providers(self) -> list[str]: ...


class OnnxSessionOptions(Protocol):
    log_severity_level: int


class OnnxRuntimeModule(Protocol):
    def get_available_providers(self) -> list[str]: ...

    def set_default_logger_severity(self, severity: int) -> None: ...

    def SessionOptions(self) -> OnnxSessionOptions: ...

    def InferenceSession(
        self,
        model_path: str,
        sess_options: OnnxSessionOptions | None = None,
        providers: list[str] | None = None,
    ) -> OnnxSession: ...


class NvmlProcessInfo(Protocol):
    pid: int
    usedGpuMemory: int


class NvmlModule(Protocol):
    def nvmlInit(self) -> None: ...

    def nvmlShutdown(self) -> None: ...

    def nvmlDeviceGetCount(self) -> int: ...

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...


class KokoroRuntimeFactory(Protocol):
    def __call__(self, model_path: str, voices_path: str) -> KokoroEngine: ...

    @classmethod
    def from_session(
        cls,
        session: OnnxSession,
        voices_path: str,
    ) -> KokoroEngine: ...


class KokoroEngineWithSession(KokoroEngine, Protocol):
    sess: OnnxSession


@dataclass(frozen=True)
class RuntimeStatus:
    requested_provider: str
    attempted_providers: list[str]
    available_providers: list[str]
    active_providers: list[str]
    provider_fallback: bool
    provider_error: str | None
    runtime_error: str | None


@dataclass(frozen=True)
class RuntimeBootstrap:
    tts: KokoroEngine
    status: RuntimeStatus


@dataclass(frozen=True)
class GpuProcessUsage:
    pid: int
    available: bool
    used_bytes: int | None
    used_megabytes: float | None
    source: str | None
    error: str | None


_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_TENSORRT_PROVIDER = "TensorrtExecutionProvider"


def _runtime_factory() -> KokoroRuntimeFactory:
    if KokoroRuntime is None:
        raise RuntimeError(
            "kokoro-onnx is not installed. Install dependencies with `uv sync`."
        )
    return cast(KokoroRuntimeFactory, KokoroRuntime)


def _onnx_runtime() -> OnnxRuntimeModule:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install dependencies with `uv sync`."
        )
    return ort  # pyright: ignore[reportReturnType]


def _load_nvml_module() -> NvmlModule | None:
    try:
        module = importlib.import_module("pynvml")
        return cast(NvmlModule, cast(object, module))
    except ImportError:
        return None


def _sanitize_runtime_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _resolve_attempted_providers(
    requested_provider: ProviderMode, available_providers: list[str]
) -> list[str]:
    if requested_provider == "cpu":
        return [_CPU_PROVIDER]
    if requested_provider == "cuda":
        return [_CUDA_PROVIDER, _CPU_PROVIDER]
    if _CUDA_PROVIDER in available_providers:
        return [_CUDA_PROVIDER, _CPU_PROVIDER]
    return [_CPU_PROVIDER]


def _create_runtime_from_session(providers: list[str]) -> KokoroEngineWithSession:
    onnx_runtime = _onnx_runtime()
    runtime_factory = _runtime_factory()
    onnx_runtime.set_default_logger_severity(4)
    session_options = onnx_runtime.SessionOptions()
    session_options.log_severity_level = 4
    session = onnx_runtime.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=providers,
    )
    tts = runtime_factory.from_session(session, str(VOICES_PATH))
    return cast(KokoroEngineWithSession, tts)


@lru_cache(maxsize=1)
def get_runtime_bootstrap() -> RuntimeBootstrap:
    _ = _runtime_factory()
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not VOICES_PATH.exists():
        raise RuntimeError(f"Voice file not found: {VOICES_PATH}")

    available_providers = get_available_runtime_providers()
    requested_provider = get_runtime_provider_mode()
    attempted_providers = _resolve_attempted_providers(
        requested_provider, available_providers
    )

    try:
        tts = _create_runtime_from_session(attempted_providers)
        return RuntimeBootstrap(
            tts=tts,
            status=RuntimeStatus(
                requested_provider=requested_provider,
                attempted_providers=attempted_providers,
                available_providers=available_providers,
                active_providers=list(tts.sess.get_providers()),
                provider_fallback=False,
                provider_error=None,
                runtime_error=None,
            ),
        )
    except Exception as exc:
        provider_error = _sanitize_runtime_error(exc)
        if attempted_providers == [_CPU_PROVIDER] or get_runtime_provider_strict():
            raise RuntimeError(provider_error) from exc

    fallback_providers = [_CPU_PROVIDER]
    try:
        tts = _create_runtime_from_session(fallback_providers)
    except Exception as fallback_exc:
        raise RuntimeError(_sanitize_runtime_error(fallback_exc)) from fallback_exc

    return RuntimeBootstrap(
        tts=tts,
        status=RuntimeStatus(
            requested_provider=requested_provider,
            attempted_providers=attempted_providers,
            available_providers=available_providers,
            active_providers=list(tts.sess.get_providers()),
            provider_fallback=True,
            provider_error=provider_error,
            runtime_error=None,
        ),
    )


def clear_runtime_caches() -> None:
    get_runtime_bootstrap.cache_clear()


def kokoro_runtime_available() -> bool:
    return KokoroRuntime is not None


def get_available_runtime_providers() -> list[str]:
    if ort is None:
        return []
    try:
        get_available_providers = cast(
            Callable[[], list[str]],
            getattr(ort, "get_available_providers"),
        )
        return list(get_available_providers())
    except Exception:
        return []


def get_runtime_status() -> RuntimeStatus:
    try:
        return get_runtime_bootstrap().status
    except Exception as exc:
        available_providers = get_available_runtime_providers()
        requested_provider = get_runtime_provider_mode()
        attempted_providers = _resolve_attempted_providers(
            requested_provider, available_providers
        )
        return RuntimeStatus(
            requested_provider=requested_provider,
            attempted_providers=attempted_providers,
            available_providers=available_providers,
            active_providers=[],
            provider_fallback=False,
            provider_error=None,
            runtime_error=_sanitize_runtime_error(exc),
        )


def get_active_runtime_providers() -> list[str]:
    return list(get_runtime_status().active_providers)


def get_active_runtime_provider() -> str | None:
    providers = get_active_runtime_providers()
    return providers[0] if providers else None


def _is_gpu_runtime_provider(provider: str | None) -> bool:
    return provider in {_CUDA_PROVIDER, _TENSORRT_PROVIDER}


def _iter_nvml_compute_processes(
    nvml_module: NvmlModule, handle: object
) -> list[NvmlProcessInfo]:
    for attr_name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        getter = cast(
            Callable[[object], object] | None, getattr(nvml_module, attr_name, None)
        )
        if getter is None:
            continue
        raw_processes = getter(handle)
        if not isinstance(raw_processes, Iterable):
            return []
        return [cast(NvmlProcessInfo, process) for process in raw_processes]
    return []


def get_current_process_gpu_usage() -> GpuProcessUsage:
    pid = os.getpid()
    active_provider = get_active_runtime_provider()
    if not _is_gpu_runtime_provider(active_provider):
        return GpuProcessUsage(
            pid=pid,
            available=False,
            used_bytes=None,
            used_megabytes=None,
            source=None,
            error=None,
        )

    nvml_module = _load_nvml_module()
    if nvml_module is None:
        return GpuProcessUsage(
            pid=pid,
            available=False,
            used_bytes=None,
            used_megabytes=None,
            source=None,
            error="pynvml is not installed.",
        )
    try:
        nvml_module.nvmlInit()
    except Exception as exc:
        return GpuProcessUsage(
            pid=pid,
            available=False,
            used_bytes=None,
            used_megabytes=None,
            source=None,
            error=_sanitize_runtime_error(exc),
        )

    try:
        used_bytes = 0
        for index in range(nvml_module.nvmlDeviceGetCount()):
            handle = nvml_module.nvmlDeviceGetHandleByIndex(index)
            try:
                processes = _iter_nvml_compute_processes(nvml_module, handle)
            except Exception:
                continue
            for process in processes:
                process_pid = int(getattr(process, "pid", -1))
                if process_pid != pid:
                    continue
                used_bytes += int(getattr(process, "usedGpuMemory", 0))

        if used_bytes <= 0:
            return GpuProcessUsage(
                pid=pid,
                available=False,
                used_bytes=0,
                used_megabytes=0.0,
                source="nvml",
                error=None,
            )

        return GpuProcessUsage(
            pid=pid,
            available=True,
            used_bytes=used_bytes,
            used_megabytes=round(used_bytes / (1024 * 1024), 2),
            source="nvml",
            error=None,
        )
    except Exception as exc:
        return GpuProcessUsage(
            pid=pid,
            available=False,
            used_bytes=None,
            used_megabytes=None,
            source=None,
            error=_sanitize_runtime_error(exc),
        )
    finally:
        try:
            nvml_module.nvmlShutdown()
        except Exception:
            pass


def get_tts() -> KokoroEngine:
    return get_runtime_bootstrap().tts


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
