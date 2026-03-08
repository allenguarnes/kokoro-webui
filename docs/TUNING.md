# Tuning

This project includes two internal tools for performance tuning:

- `scripts/tune_runtime.py`: interactive tuning loop
- `scripts/benchmark_runtime.py`: raw benchmark harness

Use the interactive tuner first. Use the raw benchmark script when you want one-shot summaries or machine-readable JSON.

## Goals

The tuning flow is designed to answer questions like:

- Does `KOKORO_SYNTH_WORKERS` help on this machine?
- Does the answer change between CPU and CUDA?
- Does a setting help one-shot synthesis but hurt streaming?
- Are we reducing queue wait at the cost of worse end-user latency?

The tool is intentionally concurrency-aware. Worker-pool changes only matter under contention, so the tuner defaults to concurrent HTTP benchmark batches instead of serial single-request runs.

## What To Tune

Current supported parameter:

- `KOKORO_SYNTH_WORKERS`

Related runtime settings that affect interpretation:

- `KOKORO_PROVIDER`
- `KOKORO_SYNTH_QUEUE`
- `KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY`
- `KOKORO_CUDA_LIB_DIR`

## Quick Start

CPU:

```bash
uv run python scripts/tune_runtime.py --provider cpu
```

CUDA / auto:

```bash
uv run python scripts/tune_runtime.py --provider auto --cuda-lib-dir /opt/cuda-12.9/lib64
```

Higher concurrent load:

```bash
uv run python scripts/tune_runtime.py --provider auto --concurrency 8 --cuda-lib-dir /opt/cuda-12.9/lib64
```

Case-by-case sweep:

```bash
uv run python scripts/tune_runtime.py --provider cpu --sweep-cases
```

Raw benchmark output:

```bash
uv run python scripts/benchmark_runtime.py --providers cpu auto --concurrency 4 --cuda-lib-dir /opt/cuda-12.9/lib64
```

## Interactive Tuner

`scripts/tune_runtime.py` runs:

1. a baseline using the built-in default for the selected provider
2. a candidate run with the parameter increased by `--step`
3. a comparison table
4. a built-in recommendation
5. an optional LLM recommendation
6. a prompt asking whether to continue, keep, accept, or exit

Prompt choices:

- `c`: continue with a higher candidate value
- `u`: accept the current candidate
- `p`: keep the previous value
- `x`: exit without selecting a new value

## Important Flags

### `--provider`

Choose runtime mode:

- `cpu`
- `auto`
- `cuda`

### `--concurrency`

Number of simultaneous requests per batch.

This is one of the most important knobs in the tuning process because it determines whether the benchmark is actually stressing the queue and worker pool.

General guidance:

- `2`: light contention
- `4`: reasonable default for local tuning
- `8+`: stronger hosted-service style contention

### `--cases`

Restrict the benchmark to specific named cases.

Examples:

- `speak_wav_pitch0`
- `stream_ndjson_wav_pitch0`
- `stream_ndjson_opus_pitch0`
- `stream_ws_wav_pitch0`

### `--sweep-cases`

Run the same tuning loop once per selected case instead of mixing workloads together.

Use this when you need to answer:

- does `speak` want a different setting than streaming?
- does WAV behave differently than Opus?
- is one case masking another in the mixed workload result?

### `--iterations` and `--warmup`

- `--warmup`: ignored runs before measurement
- `--iterations`: measured rounds

Use higher values when you want less noisy comparisons.

Reasonable starting point:

```bash
--warmup 1 --iterations 4
```

## Reading The Results

Each round reports:

- `mean_ms`: average request latency
- `p95_ms`: tail latency
- `throughput_rps`: completed requests per second
- queue wait average / max
- rejected jobs

Those numbers matter more than “did queue wait go down”.

### Practical interpretation

Good candidate:

- equal or lower `mean_ms`
- equal or lower `p95_ms`
- equal or higher `throughput_rps`
- no new rejections

Bad candidate:

- queue wait improves
- but `mean_ms` and `p95_ms` get worse
- or streaming throughput drops

That usually means the system is admitting more work sooner, but the concurrent jobs are interfering with each other enough that real user-facing performance is worse.

## Built-in Recommendation

The tuner includes a deterministic recommendation after each round.

It weighs:

- latency
- p95
- throughput
- queue pressure

It intentionally weights streaming regressions more heavily than one-shot synthesis improvements because streaming is the more fragile hosted-service path.

Possible outcomes:

- `USE CANDIDATE`
- `KEEP PREVIOUS`
- `CONTINUE TESTING`

Treat this as an informed heuristic, not an oracle.

## Optional LLM Recommendation

You can enable an additional recommendation layer via any OpenAI-compatible chat-completions endpoint.

CLI flags:

```bash
uv run python scripts/tune_runtime.py \
  --provider auto \
  --cuda-lib-dir /opt/cuda-12.9/lib64 \
  --llm-base-url https://your-openai-compatible-server/v1 \
  --llm-api-key YOUR_KEY \
  --llm-model YOUR_MODEL
```

Or via JSON config file:

```json
{
  "llm_base_url": "https://your-openai-compatible-server/v1",
  "llm_api_key": "YOUR_KEY",
  "llm_model": "YOUR_MODEL"
}
```

Example:

```bash
uv run python scripts/tune_runtime.py \
  --provider auto \
  --concurrency 8 \
  --cuda-lib-dir /opt/cuda-12.9/lib64 \
  --llm-config-file /path/to/llm-config.json
```

If no LLM settings are provided, the tuner behaves normally with no LLM calls.

If the LLM request fails, the benchmark result still completes and the script prints a warning instead of aborting.

## Current Limitation

When `--concurrency > 1`, concurrent benchmark mode currently supports HTTP endpoints only:

- `/api/speak`
- `/api/speak-stream`

WebSocket benchmark cases such as `stream_ws_wav_pitch0` are currently limited to `--concurrency 1`.

The tuner now validates this up front and stops immediately with a clear message instead of failing deep in the benchmark run.

## Suggested Workflow

1. Start with the mixed default tuning set.
2. Increase concurrency until you see real queue pressure.
3. Use the built-in recommendation as the default decision signal.
4. If the result is mixed, rerun with `--sweep-cases`.
5. Only enable `KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY=1` when intentionally testing shared-session CUDA concurrency.
6. Prefer throughput + p95 + queue behavior over average latency alone.

## Notes On CUDA

For shared-runtime CUDA mode:

- `KOKORO_SYNTH_WORKERS=1` is still the safe default
- `workers > 1` is experimental
- more workers can reduce queue wait while still making real request latency worse

That is exactly why the tuning tools exist: to measure the tradeoff instead of guessing from core count or GPU presence.
