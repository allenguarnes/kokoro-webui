from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from httpx import Response
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STREAM_TEXT = (
    "Kokoro WebUI streams sentence-safe chunks for local playback. "
    "This benchmark exercises synthesis, transport framing, and audio encoding. "
    "Pitch shifting is measured separately so the ffmpeg path is visible. "
    "The same text is reused to make CPU and CUDA runs comparable."
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    endpoint: str
    format: str
    pitch: float


@dataclass(frozen=True)
class SampleResult:
    elapsed_ms: float
    wire_bytes: int
    audio_bytes: int
    audio_duration_sec: float
    chunk_count: int
    synth_ms_sum: float


@dataclass(frozen=True)
class CaseSummary:
    name: str
    endpoint: str
    format: str
    pitch: float
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    mean_audio_bytes: int
    mean_wire_bytes: int
    mean_audio_duration_sec: float
    mean_chunk_count: float
    mean_synth_ms_sum: float
    mean_realtime_factor: float | None


@dataclass(frozen=True)
class ProviderRun:
    requested_provider: str
    active_provider: str | None
    active_providers: list[str]
    provider_fallback: bool
    provider_error: str | None
    runtime_error: str | None
    cases: list[CaseSummary]


@dataclass(frozen=True)
class CliArgs:
    child: bool
    iterations: int
    warmup: int
    providers: list[str]
    cuda_lib_dir: str | None


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("speak_wav_pitch0", "/api/speak", "wav", 0.0),
    BenchmarkCase("speak_wav_pitch2", "/api/speak", "wav", 2.0),
    BenchmarkCase("speak_opus_pitch0", "/api/speak", "opus", 0.0),
    BenchmarkCase("stream_ndjson_wav_pitch0", "/api/speak-stream", "wav", 0.0),
    BenchmarkCase("stream_ndjson_wav_pitch2", "/api/speak-stream", "wav", 2.0),
    BenchmarkCase("stream_ndjson_opus_pitch0", "/api/speak-stream", "opus", 0.0),
    BenchmarkCase("stream_ws_wav_pitch0", "/ws/speak-stream", "wav", 0.0),
    BenchmarkCase("stream_ws_wav_pitch2", "/ws/speak-stream", "wav", 2.0),
    BenchmarkCase("stream_ws_opus_pitch0", "/ws/speak-stream", "opus", 0.0),
)


def as_object_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object from benchmark payload.")
    return cast(Mapping[str, object], value)


def as_object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return cast(list[object], value)


def get_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def get_int(mapping: Mapping[str, object], key: str, default: int = 0) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) else default


def get_float(mapping: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def get_bool(mapping: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = mapping.get(key)
    return value if isinstance(value, bool) else default


def get_string_list(mapping: Mapping[str, object], key: str) -> list[str]:
    return [item for item in as_object_list(mapping.get(key)) if isinstance(item, str)]


def build_payload(case: BenchmarkCase) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": DEFAULT_STREAM_TEXT,
        "voice": "af_heart",
        "speed": 1.0,
        "pitch": case.pitch,
        "lang": "en-us",
        "format": case.format,
        "opus_bitrate": "32k",
        "wav_sample_rate": "native",
    }
    if case.endpoint != "/api/speak":
        payload["target_chunk_chars"] = 120
    return payload


def summarize_case(
    case: BenchmarkCase, samples: list[SampleResult], iterations: int
) -> CaseSummary:
    elapsed_values = [sample.elapsed_ms for sample in samples]
    mean_audio_bytes = round(statistics.fmean(sample.audio_bytes for sample in samples))
    mean_wire_bytes = round(statistics.fmean(sample.wire_bytes for sample in samples))
    mean_audio_duration_sec = statistics.fmean(
        sample.audio_duration_sec for sample in samples
    )
    mean_chunk_count = statistics.fmean(sample.chunk_count for sample in samples)
    mean_synth_ms_sum = statistics.fmean(sample.synth_ms_sum for sample in samples)
    realtime_values = [
        sample.audio_duration_sec / (sample.elapsed_ms / 1000)
        for sample in samples
        if sample.elapsed_ms > 0 and sample.audio_duration_sec > 0
    ]
    return CaseSummary(
        name=case.name,
        endpoint=case.endpoint,
        format=case.format,
        pitch=case.pitch,
        iterations=iterations,
        mean_ms=round(statistics.fmean(elapsed_values), 2),
        median_ms=round(statistics.median(elapsed_values), 2),
        min_ms=round(min(elapsed_values), 2),
        max_ms=round(max(elapsed_values), 2),
        mean_audio_bytes=mean_audio_bytes,
        mean_wire_bytes=mean_wire_bytes,
        mean_audio_duration_sec=round(mean_audio_duration_sec, 3),
        mean_chunk_count=round(mean_chunk_count, 2),
        mean_synth_ms_sum=round(mean_synth_ms_sum, 2),
        mean_realtime_factor=round(statistics.fmean(realtime_values), 2)
        if realtime_values
        else None,
    )


def run_case_http_speak(client: TestClient, case: BenchmarkCase) -> SampleResult:
    payload = build_payload(case)
    started = time.perf_counter()
    response: Response = client.post(case.endpoint, json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _ = response.raise_for_status()
    audio_bytes = response.content
    duration_header = cast(str | None, response.headers.get("x-audio-duration"))
    duration_sec = float(duration_header) if duration_header else 0.0
    return SampleResult(
        elapsed_ms=elapsed_ms,
        wire_bytes=len(audio_bytes),
        audio_bytes=len(audio_bytes),
        audio_duration_sec=duration_sec,
        chunk_count=1,
        synth_ms_sum=elapsed_ms,
    )


def run_case_http_stream(client: TestClient, case: BenchmarkCase) -> SampleResult:
    payload = build_payload(case)
    started = time.perf_counter()
    response: Response = client.post(case.endpoint, json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _ = response.raise_for_status()
    wire_bytes = len(response.content)
    audio_bytes = 0
    audio_duration_sec = 0.0
    chunk_count = 0
    synth_ms_sum = 0.0
    for line in response.text.splitlines():
        if not line.strip():
            continue
        event = as_object_mapping(cast(object, json.loads(line)))
        if get_string(event, "type") != "chunk":
            continue
        chunk_count += 1
        audio_bytes += get_int(event, "bytes")
        audio_duration_sec += get_float(event, "duration_sec")
        synth_ms_sum += get_float(event, "synth_ms")
    return SampleResult(
        elapsed_ms=elapsed_ms,
        wire_bytes=wire_bytes,
        audio_bytes=audio_bytes,
        audio_duration_sec=audio_duration_sec,
        chunk_count=chunk_count,
        synth_ms_sum=synth_ms_sum,
    )


def run_case_websocket(client: TestClient, case: BenchmarkCase) -> SampleResult:
    payload = build_payload(case)
    started = time.perf_counter()
    with client.websocket_connect(case.endpoint) as websocket:
        websocket.send_text(json.dumps(payload))
        wire_bytes = 0
        audio_bytes = 0
        audio_duration_sec = 0.0
        chunk_count = 0
        synth_ms_sum = 0.0
        while True:
            frame = websocket.receive_text()
            wire_bytes += len(frame.encode("utf-8"))
            event = as_object_mapping(cast(object, json.loads(frame)))
            event_type = get_string(event, "type")
            if event_type == "chunk":
                chunk_audio = websocket.receive_bytes()
                wire_bytes += len(chunk_audio)
                chunk_count += 1
                audio_bytes += len(chunk_audio)
                audio_duration_sec += get_float(event, "duration_sec")
                synth_ms_sum += get_float(event, "synth_ms")
            if event_type in {"done", "error"}:
                break
    elapsed_ms = (time.perf_counter() - started) * 1000
    return SampleResult(
        elapsed_ms=elapsed_ms,
        wire_bytes=wire_bytes,
        audio_bytes=audio_bytes,
        audio_duration_sec=audio_duration_sec,
        chunk_count=chunk_count,
        synth_ms_sum=synth_ms_sum,
    )


def run_provider_benchmark(iterations: int, warmup: int) -> ProviderRun:
    import app.main as main
    import app.runtime as runtime

    runtime.clear_runtime_caches()
    client = TestClient(main.app)
    try:
        health: Response = client.get("/api/health")
        _ = health.raise_for_status()
        health_payload = as_object_mapping(cast(object, health.json()))
        case_summaries: list[CaseSummary] = []
        for case in CASES:
            runner = (
                run_case_http_speak
                if case.endpoint == "/api/speak"
                else run_case_http_stream
                if case.endpoint == "/api/speak-stream"
                else run_case_websocket
            )
            for _ in range(warmup):
                _ = runner(client, case)
            samples = [runner(client, case) for _ in range(iterations)]
            case_summaries.append(summarize_case(case, samples, iterations))
    finally:
        client.close()
        runtime.clear_runtime_caches()

    return ProviderRun(
        requested_provider=get_string(health_payload, "requested_provider") or "",
        active_provider=get_string(health_payload, "active_provider"),
        active_providers=get_string_list(health_payload, "active_providers"),
        provider_fallback=get_bool(health_payload, "provider_fallback"),
        provider_error=get_string(health_payload, "provider_error"),
        runtime_error=get_string(health_payload, "runtime_error"),
        cases=case_summaries,
    )


def child_main(args: CliArgs) -> int:
    result = run_provider_benchmark(iterations=args.iterations, warmup=args.warmup)
    print(json.dumps(asdict(result)))
    return 0


def build_child_env(
    provider: str, cuda_lib_dir: str | None, inherit_env: Mapping[str, str]
) -> dict[str, str]:
    env = dict(inherit_env)
    env["KOKORO_PROVIDER"] = provider
    if cuda_lib_dir:
        env["KOKORO_CUDA_LIB_DIR"] = cuda_lib_dir
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [part for part in existing.split(os.pathsep) if part]
        if cuda_lib_dir not in parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([cuda_lib_dir, *parts])
    else:
        _ = env.pop("KOKORO_CUDA_LIB_DIR", None)
    return env


def print_report(results: list[ProviderRun]) -> None:
    print("Benchmark summary")
    for result in results:
        provider_label = result.active_provider or "unavailable"
        suffix = " (fallback)" if result.provider_fallback else ""
        print(
            f"\nProvider request={result.requested_provider} active={provider_label}{suffix}"
        )
        if result.provider_error:
            print(f"Provider error: {result.provider_error}")
        if result.runtime_error:
            print(f"Runtime error: {result.runtime_error}")
        print(
            "case".ljust(28),
            "mean_ms".rjust(9),
            "rtf".rjust(6),
            "audio_kb".rjust(10),
            "wire_kb".rjust(9),
            "chunks".rjust(7),
            "synth_ms".rjust(10),
        )
        for case in result.cases:
            rtf = (
                f"{case.mean_realtime_factor:.2f}"
                if case.mean_realtime_factor is not None
                else "--"
            )
            print(
                case.name.ljust(28),
                f"{case.mean_ms:9.2f}",
                f"{rtf:>6}",
                f"{case.mean_audio_bytes / 1024:10.1f}",
                f"{case.mean_wire_bytes / 1024:9.1f}",
                f"{case.mean_chunk_count:7.2f}",
                f"{case.mean_synth_ms_sum:10.2f}",
            )


def case_summary_from_json(payload: Mapping[str, object]) -> CaseSummary:
    return CaseSummary(
        name=get_string(payload, "name") or "",
        endpoint=get_string(payload, "endpoint") or "",
        format=get_string(payload, "format") or "",
        pitch=get_float(payload, "pitch"),
        iterations=get_int(payload, "iterations"),
        mean_ms=get_float(payload, "mean_ms"),
        median_ms=get_float(payload, "median_ms"),
        min_ms=get_float(payload, "min_ms"),
        max_ms=get_float(payload, "max_ms"),
        mean_audio_bytes=get_int(payload, "mean_audio_bytes"),
        mean_wire_bytes=get_int(payload, "mean_wire_bytes"),
        mean_audio_duration_sec=get_float(payload, "mean_audio_duration_sec"),
        mean_chunk_count=get_float(payload, "mean_chunk_count"),
        mean_synth_ms_sum=get_float(payload, "mean_synth_ms_sum"),
        mean_realtime_factor=(
            get_float(payload, "mean_realtime_factor")
            if payload.get("mean_realtime_factor") is not None
            else None
        ),
    )


def provider_run_from_json(payload: Mapping[str, object]) -> ProviderRun:
    return ProviderRun(
        requested_provider=get_string(payload, "requested_provider") or "",
        active_provider=get_string(payload, "active_provider"),
        active_providers=get_string_list(payload, "active_providers"),
        provider_fallback=get_bool(payload, "provider_fallback"),
        provider_error=get_string(payload, "provider_error"),
        runtime_error=get_string(payload, "runtime_error"),
        cases=[
            case_summary_from_json(as_object_mapping(item))
            for item in as_object_list(payload.get("cases"))
        ],
    )


def parent_main(args: CliArgs) -> int:
    results: list[ProviderRun] = []
    for provider in args.providers:
        env = build_child_env(provider, args.cuda_lib_dir, os.environ)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
        ]
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            _ = sys.stderr.write(completed.stderr)
            raise SystemExit(
                f"Benchmark child failed for provider={provider}: {completed.returncode}"
            )
        payload = completed.stdout.strip().splitlines()[-1]
        parsed = as_object_mapping(cast(object, json.loads(payload)))
        results.append(provider_run_from_json(parsed))

    print_report(results)
    print("\nJSON")
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark Kokoro WebUI runtime paths."
    )
    _ = parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    _ = parser.add_argument("--iterations", type=int, default=3)
    _ = parser.add_argument("--warmup", type=int, default=1)
    _ = parser.add_argument(
        "--providers",
        nargs="+",
        default=["cpu", "auto"],
        help="Provider modes to benchmark in isolated subprocesses.",
    )
    _ = parser.add_argument(
        "--cuda-lib-dir",
        default=None,
        help="Optional CUDA 12 library directory for GPU-capable runs.",
    )
    namespace = parser.parse_args()
    providers_obj = cast(object, namespace.providers)
    child_obj = cast(object, namespace.child)
    iterations_obj = cast(object, namespace.iterations)
    warmup_obj = cast(object, namespace.warmup)
    cuda_lib_dir_obj = cast(object, namespace.cuda_lib_dir)
    providers_raw = providers_obj
    if not isinstance(providers_raw, list):
        raise SystemExit("--providers must be a list of strings")
    providers: list[str] = []
    for provider_obj in cast(list[object], providers_raw):
        if not isinstance(provider_obj, str):
            raise SystemExit("--providers must be a list of strings")
        providers.append(provider_obj)
    if not isinstance(child_obj, bool):
        raise SystemExit("--child must resolve to a boolean")
    if not isinstance(iterations_obj, int):
        raise SystemExit("--iterations must resolve to an integer")
    if not isinstance(warmup_obj, int):
        raise SystemExit("--warmup must resolve to an integer")
    cuda_lib_dir = cuda_lib_dir_obj
    if cuda_lib_dir is not None and not isinstance(cuda_lib_dir, str):
        raise SystemExit("--cuda-lib-dir must be a string when provided")
    return CliArgs(
        child=child_obj,
        iterations=iterations_obj,
        warmup=warmup_obj,
        providers=providers,
        cuda_lib_dir=cuda_lib_dir,
    )


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be 0 or greater")
    return child_main(args) if args.child else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
