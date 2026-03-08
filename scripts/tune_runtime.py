from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
    import httpx

    from scripts import benchmark_runtime as bench
else:
    import httpx

    from . import benchmark_runtime as bench

DEFAULT_TUNING_CASES: tuple[str, ...] = (
    "speak_wav_pitch0",
    "stream_ndjson_wav_pitch0",
    "stream_ndjson_opus_pitch0",
)
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"


@dataclass(frozen=True)
class TuningParameter:
    name: str
    env_var: str
    description: str

    def default_value(self, provider: str) -> int:
        if self.name == "synth-workers":
            return 2 if provider == "cpu" else 1
        raise RuntimeError(f"Unsupported tuning parameter: {self.name}")

    def value_from_run(self, run: bench.ProviderRun) -> int:
        if self.name == "synth-workers":
            return run.synthesis_workers
        raise RuntimeError(f"Unsupported tuning parameter: {self.name}")


@dataclass(frozen=True)
class TuneArgs:
    provider: str
    parameter: str
    step: int
    iterations: int
    warmup: int
    concurrency: int
    cuda_lib_dir: str | None
    cases: list[str] | None
    full_suite: bool
    sweep_cases: bool
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_config_file: str | None


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class StepResult:
    requested_label: str
    requested_value: int | None
    effective_value: int
    run: bench.ProviderRun


RecommendationVerdict = Literal["use-candidate", "keep-previous", "continue-testing"]


@dataclass(frozen=True)
class BuiltinRecommendation:
    verdict: RecommendationVerdict
    weighted_score: float
    summary: str
    bullets: list[str]


PARAMETERS: dict[str, TuningParameter] = {
    "synth-workers": TuningParameter(
        name="synth-workers",
        env_var="KOKORO_SYNTH_WORKERS",
        description="Tune concurrent synthesis workers against the current runtime mode.",
    )
}


def colorize(text: str, *styles: str) -> str:
    if not USE_COLOR or not styles:
        return text
    return "".join(styles) + text + RESET


def status_color(level: RecommendationVerdict) -> str:
    if level == "use-candidate":
        return GREEN
    if level == "keep-previous":
        return RED
    return YELLOW


def queue_wait_color(wait_ms: float) -> str:
    if wait_ms >= 1500:
        return RED
    if wait_ms >= 300:
        return YELLOW
    return GREEN


def format_delta(delta: float, *, prefer_lower: bool) -> str:
    good = delta < 0 if prefer_lower else delta > 0
    color = GREEN if good else RED if delta != 0 else GRAY
    return colorize(f"{delta:+.2f}%".rjust(8), color)


def parse_args() -> TuneArgs:
    parser = argparse.ArgumentParser(
        description="Interactive tuner for Kokoro WebUI performance parameters."
    )
    _ = parser.add_argument(
        "--provider", default="auto", choices=["cpu", "auto", "cuda"]
    )
    _ = parser.add_argument(
        "--parameter",
        default="synth-workers",
        choices=sorted(PARAMETERS),
    )
    _ = parser.add_argument("--step", type=int, default=1)
    _ = parser.add_argument("--iterations", type=int, default=2)
    _ = parser.add_argument("--warmup", type=int, default=1)
    _ = parser.add_argument("--concurrency", type=int, default=4)
    _ = parser.add_argument("--cuda-lib-dir", default=None)
    _ = parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of benchmark case names.",
    )
    _ = parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Benchmark all benchmark_runtime.py cases instead of the focused tuning suite.",
    )
    _ = parser.add_argument(
        "--sweep-cases",
        action="store_true",
        help="Run the tuning loop separately for each selected case.",
    )
    _ = parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for recommendation analysis.",
    )
    _ = parser.add_argument(
        "--llm-api-key",
        default=None,
        help="Optional API key for the recommendation model.",
    )
    _ = parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional model name for the recommendation analysis.",
    )
    _ = parser.add_argument(
        "--llm-config-file",
        default=None,
        help="Optional JSON file containing llm_base_url, llm_api_key, and llm_model.",
    )
    namespace = parser.parse_args()
    provider_obj = cast(object, namespace.provider)
    parameter_obj = cast(object, namespace.parameter)
    step_obj = cast(object, namespace.step)
    iterations_obj = cast(object, namespace.iterations)
    warmup_obj = cast(object, namespace.warmup)
    concurrency_obj = cast(object, namespace.concurrency)
    cuda_lib_dir_obj = cast(object, namespace.cuda_lib_dir)
    full_suite_obj = cast(object, namespace.full_suite)
    sweep_cases_obj = cast(object, namespace.sweep_cases)
    cases_obj = cast(object, namespace.cases)
    llm_base_url_obj = cast(object, namespace.llm_base_url)
    llm_api_key_obj = cast(object, namespace.llm_api_key)
    llm_model_obj = cast(object, namespace.llm_model)
    llm_config_file_obj = cast(object, namespace.llm_config_file)
    if not isinstance(provider_obj, str):
        raise SystemExit("--provider must be a string")
    if not isinstance(parameter_obj, str):
        raise SystemExit("--parameter must be a string")
    if not isinstance(step_obj, int) or step_obj < 1:
        raise SystemExit("--step must be an integer greater than 0")
    if not isinstance(iterations_obj, int) or iterations_obj < 1:
        raise SystemExit("--iterations must be an integer greater than 0")
    if not isinstance(warmup_obj, int) or warmup_obj < 0:
        raise SystemExit("--warmup must be an integer 0 or greater")
    if not isinstance(concurrency_obj, int) or concurrency_obj < 1:
        raise SystemExit("--concurrency must be an integer greater than 0")
    if cuda_lib_dir_obj is not None and not isinstance(cuda_lib_dir_obj, str):
        raise SystemExit("--cuda-lib-dir must be a string when provided")
    if not isinstance(full_suite_obj, bool):
        raise SystemExit("--full-suite must resolve to a boolean")
    if not isinstance(sweep_cases_obj, bool):
        raise SystemExit("--sweep-cases must resolve to a boolean")
    if llm_base_url_obj is not None and not isinstance(llm_base_url_obj, str):
        raise SystemExit("--llm-base-url must be a string when provided")
    if llm_api_key_obj is not None and not isinstance(llm_api_key_obj, str):
        raise SystemExit("--llm-api-key must be a string when provided")
    if llm_model_obj is not None and not isinstance(llm_model_obj, str):
        raise SystemExit("--llm-model must be a string when provided")
    if llm_config_file_obj is not None and not isinstance(llm_config_file_obj, str):
        raise SystemExit("--llm-config-file must be a string when provided")
    if cases_obj is None:
        cases: list[str] | None = None
    else:
        if not isinstance(cases_obj, list):
            raise SystemExit("--cases must be a list of strings")
        cases = []
        for case_obj in cast(list[object], cases_obj):
            if not isinstance(case_obj, str):
                raise SystemExit("--cases must be a list of strings")
            cases.append(case_obj)
    return TuneArgs(
        provider=provider_obj,
        parameter=parameter_obj,
        step=step_obj,
        iterations=iterations_obj,
        warmup=warmup_obj,
        concurrency=concurrency_obj,
        cuda_lib_dir=cuda_lib_dir_obj,
        cases=cases,
        full_suite=full_suite_obj,
        sweep_cases=sweep_cases_obj,
        llm_base_url=llm_base_url_obj,
        llm_api_key=llm_api_key_obj,
        llm_model=llm_model_obj,
        llm_config_file=llm_config_file_obj,
    )


def resolve_llm_config(args: TuneArgs) -> LlmConfig | None:
    file_values: dict[str, str] = {}
    if args.llm_config_file is not None:
        config_path = Path(args.llm_config_file)
        raw_obj = cast(object, json.loads(config_path.read_text(encoding="utf-8")))
        config_mapping = bench.as_object_mapping(raw_obj)
        for key in ("llm_base_url", "llm_api_key", "llm_model"):
            value = config_mapping.get(key)
            if isinstance(value, str) and value.strip():
                file_values[key] = value.strip()

    base_url = (args.llm_base_url or file_values.get("llm_base_url") or "").strip()
    api_key = (args.llm_api_key or file_values.get("llm_api_key") or "").strip()
    model = (args.llm_model or file_values.get("llm_model") or "").strip()
    if not any((base_url, api_key, model)):
        return None
    missing: list[str] = []
    if not base_url:
        missing.append("llm_base_url")
    if not api_key:
        missing.append("llm_api_key")
    if not model:
        missing.append("llm_model")
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(
            f"LLM recommendation config is incomplete. Missing: {missing_str}"
        )
    return LlmConfig(base_url=base_url, api_key=api_key, model=model)


def resolve_cases(args: TuneArgs) -> list[str] | None:
    if args.full_suite:
        return None
    if args.cases is not None:
        return args.cases
    return list(DEFAULT_TUNING_CASES)


def validate_tuning_cases(
    cases: list[str] | None, concurrency: int
) -> list[str] | None:
    selected = bench.validate_case_selection(cases, concurrency)
    return [case.name for case in selected]


def case_weight(case_name: str) -> float:
    return 2.5 if case_name.startswith("stream_") else 1.0


def percent_delta(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def queue_delta_summary(
    previous: StepResult, current: StepResult
) -> tuple[float, float]:
    return (
        percent_delta(previous.run.queue_wait_avg_ms, current.run.queue_wait_avg_ms),
        percent_delta(previous.run.queue_wait_max_ms, current.run.queue_wait_max_ms),
    )


def run_benchmark_step(
    *,
    provider: str,
    parameter: TuningParameter,
    requested_value: int | None,
    iterations: int,
    warmup: int,
    concurrency: int,
    cuda_lib_dir: str | None,
    cases: list[str] | None,
) -> StepResult:
    env = bench.build_child_env(provider, cuda_lib_dir, os.environ)
    if requested_value is None:
        _ = env.pop(parameter.env_var, None)
    else:
        env[parameter.env_var] = str(requested_value)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_runtime.py"),
        "--child",
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--concurrency",
        str(concurrency),
    ]
    if cases is not None:
        command.extend(["--cases", *cases])
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
            f"Tuning benchmark failed for provider={provider} value={requested_value}: {completed.returncode}"
        )
    payload = completed.stdout.strip().splitlines()[-1]
    parsed = bench.as_object_mapping(cast(object, json.loads(payload)))
    run = bench.provider_run_from_json(parsed)
    effective_value = parameter.value_from_run(run)
    requested_label = "default" if requested_value is None else str(requested_value)
    return StepResult(
        requested_label=requested_label,
        requested_value=requested_value,
        effective_value=effective_value,
        run=run,
    )


def print_step_header(title: str) -> None:
    styled_title = colorize(title, BOLD, CYAN)
    underline = colorize("-" * len(title), DIM, CYAN)
    print(f"\n{styled_title}")
    print(underline)


def print_run_overview(label: str, step: StepResult) -> None:
    provider_label = step.run.active_provider or "unavailable"
    label_text = colorize(label, BOLD, BLUE if label == "baseline" else MAGENTA)
    provider_text = colorize(provider_label, CYAN if "CPU" in provider_label else GREEN)
    overview = (
        f"{label_text}: requested={colorize(step.requested_label, BOLD)} "
        f"effective={colorize(str(step.effective_value), BOLD)} "
        f"provider={provider_text} "
        f"execution={colorize(step.run.scheduler_execution_model, YELLOW)} "
        f"queue={colorize(str(step.run.synthesis_queue_limit), BOLD)} "
        f"concurrency={colorize(str(step.run.concurrency), BOLD)}"
    )
    print(overview)
    queue_summary = (
        f"  queue wait avg={colorize(f'{step.run.queue_wait_avg_ms:.2f}ms', queue_wait_color(step.run.queue_wait_avg_ms))} "
        f"max={colorize(f'{step.run.queue_wait_max_ms:.2f}ms', queue_wait_color(step.run.queue_wait_max_ms))} "
        f"samples={colorize(str(step.run.queue_wait_samples), DIM)} "
        f"rejected={colorize(str(step.run.queue_rejected_jobs_total), RED if step.run.queue_rejected_jobs_total else GREEN)}"
    )
    print(queue_summary)
    if step.run.provider_fallback:
        print(
            colorize(
                f"  fallback active: requested {step.run.requested_provider}",
                YELLOW,
            )
        )
    if step.run.provider_error:
        print(colorize(f"  provider error: {step.run.provider_error}", RED))
    if step.run.runtime_error:
        print(colorize(f"  runtime error: {step.run.runtime_error}", RED))


def compare_steps(previous: StepResult, current: StepResult) -> None:
    previous_cases = {case.name: case for case in previous.run.cases}
    current_cases = {case.name: case for case in current.run.cases}
    case_names = [
        case.name for case in current.run.cases if case.name in previous_cases
    ]
    print_step_header("Comparison")
    print(
        colorize("case".ljust(28), DIM),
        colorize("prev_ms".rjust(9), DIM),
        colorize("curr_ms".rjust(9), DIM),
        colorize("prev_p95".rjust(9), DIM),
        colorize("curr_p95".rjust(9), DIM),
        colorize("prev_rps".rjust(9), DIM),
        colorize("curr_rps".rjust(9), DIM),
        colorize("delta%".rjust(8), DIM),
    )
    latency_deltas: list[float] = []
    improved_cases = 0
    for case_name in case_names:
        prev_case = previous_cases[case_name]
        curr_case = current_cases[case_name]
        delta_pct = ((curr_case.mean_ms - prev_case.mean_ms) / prev_case.mean_ms) * 100
        latency_deltas.append(delta_pct)
        if curr_case.mean_ms < prev_case.mean_ms:
            improved_cases += 1
        print(
            case_name.ljust(28),
            f"{prev_case.mean_ms:9.2f}",
            f"{curr_case.mean_ms:9.2f}",
            f"{prev_case.p95_ms:9.2f}",
            f"{curr_case.p95_ms:9.2f}",
            f"{prev_case.throughput_rps:9.2f}",
            f"{curr_case.throughput_rps:9.2f}",
            format_delta(delta_pct, prefer_lower=True),
        )
    avg_delta = statistics.fmean(latency_deltas) if latency_deltas else 0.0
    print()
    summary = (
        f"Latency improved in {improved_cases}/{len(case_names)} cases. "
        f"Average mean_ms delta: {avg_delta:+.2f}%"
    )
    summary_color = GREEN if avg_delta < 0 else RED if avg_delta > 0 else GRAY
    print(colorize(summary, BOLD, summary_color))


def build_builtin_recommendation(
    previous: StepResult, current: StepResult
) -> BuiltinRecommendation:
    previous_cases = {case.name: case for case in previous.run.cases}
    current_cases = {case.name: case for case in current.run.cases}
    weighted_components: list[float] = []
    stream_regressions = 0
    stream_improvements = 0
    per_case_notes: list[str] = []

    for case_name, current_case in current_cases.items():
        previous_case = previous_cases.get(case_name)
        if previous_case is None:
            continue
        weight = case_weight(case_name)
        mean_delta = percent_delta(previous_case.mean_ms, current_case.mean_ms)
        p95_delta = percent_delta(previous_case.p95_ms, current_case.p95_ms)
        rps_delta = percent_delta(
            previous_case.throughput_rps, current_case.throughput_rps
        )
        component = (0.5 * mean_delta) + (0.35 * p95_delta) - (0.15 * rps_delta)
        weighted_components.append(component * weight)
        if case_name.startswith("stream_"):
            if mean_delta > 10 or p95_delta > 15 or rps_delta < -8:
                stream_regressions += 1
            if mean_delta < -8 and p95_delta < -8 and rps_delta > 5:
                stream_improvements += 1
        per_case_notes.append(
            f"{case_name}: mean {mean_delta:+.2f}%, p95 {p95_delta:+.2f}%, rps {rps_delta:+.2f}%"
        )

    queue_avg_delta, queue_max_delta = queue_delta_summary(previous, current)
    rejected_delta = (
        current.run.queue_rejected_jobs_total - previous.run.queue_rejected_jobs_total
    )
    weighted_score = (
        statistics.fmean(weighted_components) if weighted_components else 0.0
    )
    weighted_score += (-0.1 * queue_avg_delta) + (-0.05 * queue_max_delta)
    weighted_score += 25.0 * rejected_delta

    bullets = [
        f"Weighted score {weighted_score:+.2f}. Queue wait avg {queue_avg_delta:+.2f}% and max {queue_max_delta:+.2f}%.",
    ]
    if per_case_notes:
        bullets.append(
            per_case_notes[0]
            if len(per_case_notes) == 1
            else "; ".join(per_case_notes[:2])
        )

    if rejected_delta > 0:
        return BuiltinRecommendation(
            verdict="keep-previous",
            weighted_score=round(weighted_score, 2),
            summary="Keep previous value because the candidate increased overload rejections.",
            bullets=bullets,
        )
    if stream_regressions > 0 and stream_improvements == 0:
        return BuiltinRecommendation(
            verdict="keep-previous",
            weighted_score=round(weighted_score, 2),
            summary="Keep previous value because streaming performance regressed under load.",
            bullets=bullets,
        )
    if weighted_score <= -8 and stream_regressions == 0:
        return BuiltinRecommendation(
            verdict="use-candidate",
            weighted_score=round(weighted_score, 2),
            summary="Use the candidate value because weighted latency and throughput improved without new queue pressure.",
            bullets=bullets,
        )
    if weighted_score >= 8:
        return BuiltinRecommendation(
            verdict="keep-previous",
            weighted_score=round(weighted_score, 2),
            summary="Keep previous value because the candidate is worse after weighting latency, p95, throughput, and queue pressure.",
            bullets=bullets,
        )
    return BuiltinRecommendation(
        verdict="continue-testing",
        weighted_score=round(weighted_score, 2),
        summary="Result is mixed. Keep testing higher values only if you are targeting a different workload mix.",
        bullets=bullets,
    )


def print_builtin_recommendation(recommendation: BuiltinRecommendation) -> None:
    print_step_header("Built-in Recommendation")
    verdict_label = {
        "use-candidate": "USE CANDIDATE",
        "keep-previous": "KEEP PREVIOUS",
        "continue-testing": "CONTINUE TESTING",
    }[recommendation.verdict]
    print(
        colorize(
            f"{verdict_label}: {recommendation.summary}",
            BOLD,
            status_color(recommendation.verdict),
        )
    )
    for bullet in recommendation.bullets:
        print(f"{colorize('-', status_color(recommendation.verdict))} {bullet}")


def build_llm_recommendation_payload(
    previous: StepResult, current: StepResult, parameter: TuningParameter
) -> dict[str, object]:
    previous_cases = {case.name: case for case in previous.run.cases}
    current_cases = {case.name: case for case in current.run.cases}
    case_payload: list[dict[str, object]] = []
    for case_name, current_case in current_cases.items():
        previous_case = previous_cases.get(case_name)
        if previous_case is None:
            continue
        delta_mean_ms = (
            ((current_case.mean_ms - previous_case.mean_ms) / previous_case.mean_ms)
            * 100
            if previous_case.mean_ms > 0
            else 0.0
        )
        delta_throughput = current_case.throughput_rps - previous_case.throughput_rps
        case_payload.append(
            {
                "name": case_name,
                "previous_mean_ms": previous_case.mean_ms,
                "current_mean_ms": current_case.mean_ms,
                "previous_p95_ms": previous_case.p95_ms,
                "current_p95_ms": current_case.p95_ms,
                "previous_rps": previous_case.throughput_rps,
                "current_rps": current_case.throughput_rps,
                "delta_mean_ms_pct": round(delta_mean_ms, 2),
                "delta_rps": round(delta_throughput, 2),
            }
        )
    return {
        "parameter": parameter.name,
        "previous_value": previous.effective_value,
        "current_value": current.effective_value,
        "provider": current.run.active_provider,
        "execution_model": current.run.scheduler_execution_model,
        "concurrency": current.run.concurrency,
        "previous_queue_wait_avg_ms": previous.run.queue_wait_avg_ms,
        "current_queue_wait_avg_ms": current.run.queue_wait_avg_ms,
        "previous_queue_wait_max_ms": previous.run.queue_wait_max_ms,
        "current_queue_wait_max_ms": current.run.queue_wait_max_ms,
        "previous_rejected_jobs": previous.run.queue_rejected_jobs_total,
        "current_rejected_jobs": current.run.queue_rejected_jobs_total,
        "cases": case_payload,
    }


def request_llm_recommendation(
    llm_config: LlmConfig,
    previous: StepResult,
    current: StepResult,
    parameter: TuningParameter,
) -> str:
    prompt_payload = build_llm_recommendation_payload(previous, current, parameter)
    system_prompt = (
        "You are evaluating benchmark results for a TTS server tuning tool. "
        "Be concise and practical. State whether the new candidate should replace the previous value. "
        "Base the recommendation on latency, p95, throughput, and queue pressure. "
        "Give 2 short bullets max and a final recommendation sentence."
    )
    user_prompt = (
        "Analyze this tuning round and recommend whether to keep the previous value or use the candidate.\n\n"
        + json.dumps(prompt_payload, indent=2)
    )
    endpoint = llm_config.base_url.rstrip("/") + "/chat/completions"
    response = httpx.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": llm_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.6,
            "max_tokens": 4096,
        },
        timeout=60.0,
    )
    _ = response.raise_for_status()
    payload = bench.as_object_mapping(cast(object, response.json()))
    choices = bench.as_object_list(payload.get("choices"))
    if not choices:
        raise RuntimeError("LLM response did not include choices.")
    first_choice = bench.as_object_mapping(choices[0])
    message = bench.as_object_mapping(first_choice.get("message") or {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response did not include message content.")
    return content.strip()


def print_llm_status(message: str) -> None:
    print(colorize(message, BOLD, BLUE), flush=True)


def run_tuning_session(
    *,
    args: TuneArgs,
    parameter: TuningParameter,
    llm_config: LlmConfig | None,
    cases: list[str] | None,
    session_label: str | None = None,
) -> int:
    baseline_value = parameter.default_value(args.provider)

    if session_label is not None:
        print_step_header(session_label)
    print_step_header("Kokoro WebUI Tuning")
    print(f"Provider: {args.provider}")
    print(f"Parameter: {parameter.name} ({parameter.description})")
    run_config = (
        f"Iterations: {args.iterations}  Warmup: {args.warmup}  "
        f"Concurrency: {args.concurrency}  Step: {args.step}"
    )
    print(run_config)
    if cases is None:
        print("Cases: full benchmark suite")
    else:
        print(f"Cases: {', '.join(cases)}")
    if os.getenv(parameter.env_var):
        print(
            f"Note: {parameter.env_var} is set in the current shell or .env, but the baseline run will force the built-in default first."
        )
    if llm_config is not None:
        print(
            f"LLM recommendation: enabled via model {llm_config.model} at {llm_config.base_url}"
        )

    baseline = run_benchmark_step(
        provider=args.provider,
        parameter=parameter,
        requested_value=baseline_value,
        iterations=args.iterations,
        warmup=args.warmup,
        concurrency=args.concurrency,
        cuda_lib_dir=args.cuda_lib_dir,
        cases=cases,
    )
    baseline = StepResult(
        requested_label="default",
        requested_value=baseline.requested_value,
        effective_value=baseline.effective_value,
        run=baseline.run,
    )
    history: list[StepResult] = [baseline]
    print_step_header("Baseline")
    print_run_overview("baseline", baseline)

    current = baseline
    next_value = current.effective_value + args.step
    while True:
        candidate = run_benchmark_step(
            provider=args.provider,
            parameter=parameter,
            requested_value=next_value,
            iterations=args.iterations,
            warmup=args.warmup,
            concurrency=args.concurrency,
            cuda_lib_dir=args.cuda_lib_dir,
            cases=cases,
        )
        history.append(candidate)
        print_step_header("Candidate")
        print_run_overview("candidate", candidate)
        compare_steps(current, candidate)
        builtin_recommendation = build_builtin_recommendation(current, candidate)
        print_builtin_recommendation(builtin_recommendation)
        if llm_config is not None:
            print_llm_status(
                f"\nLLM analysis: waiting for {llm_config.model} at {llm_config.base_url} ..."
            )
            try:
                recommendation = request_llm_recommendation(
                    llm_config, current, candidate, parameter
                )
            except Exception as exc:
                print(f"\nLLM recommendation unavailable: {exc}")
            else:
                print_step_header("LLM Recommendation")
                print(recommendation)

        choice = prompt_choice()
        if choice == "c":
            current = candidate
            next_value = candidate.effective_value + args.step
            continue
        if choice == "u":
            print_history(history)
            print(f"\nSelected {parameter.env_var}={candidate.effective_value}")
            return 0
        if choice == "p":
            print_history(history)
            print(f"\nKept {parameter.env_var}={current.effective_value}")
            return 0
        print_history(history)
        print("\nExited without selecting a new value.")
        return 0


def print_history(history: list[StepResult]) -> None:
    print_step_header("Run History")
    print(
        colorize("step".ljust(6), DIM),
        colorize("requested".rjust(10), DIM),
        colorize("effective".rjust(10), DIM),
        colorize("provider".rjust(24), DIM),
        colorize("execution".rjust(16), DIM),
        colorize("conc".rjust(6), DIM),
    )
    for index, step in enumerate(history, start=1):
        provider_label = step.run.active_provider or "unavailable"
        print(
            str(index).ljust(6),
            f"{step.requested_label:>10}",
            f"{step.effective_value:>10}",
            f"{provider_label:>24}",
            f"{step.run.scheduler_execution_model:>16}",
            f"{step.run.concurrency:>6}",
        )


def prompt_choice() -> str:
    while True:
        choice = (
            input(
                "\n[c] continue with a higher value, [u] use current candidate, [p] keep previous value, [x] exit: "
            )
            .strip()
            .lower()
        )
        if choice in {"c", "u", "p", "x"}:
            return choice
        print("Enter c, u, p, or x.")


def main() -> int:
    args = parse_args()
    parameter = PARAMETERS[args.parameter]
    cases = validate_tuning_cases(resolve_cases(args), args.concurrency)
    llm_config = resolve_llm_config(args)
    if args.sweep_cases:
        if cases is None:
            cases = list(DEFAULT_TUNING_CASES)
        for case_name in cases:
            session_label = f"Case Sweep: {case_name}"
            _ = run_tuning_session(
                args=args,
                parameter=parameter,
                llm_config=llm_config,
                cases=[case_name],
                session_label=session_label,
            )
        return 0
    return run_tuning_session(
        args=args,
        parameter=parameter,
        llm_config=llm_config,
        cases=cases,
    )


if __name__ == "__main__":
    raise SystemExit(main())
