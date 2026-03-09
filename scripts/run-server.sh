#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHOW_CHECKS=1
CHECK_ONLY=0
APP_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --quiet-checks)
      SHOW_CHECKS=0
      ;;
    --check-only)
      CHECK_ONLY=1
      ;;
    --help)
      cat <<'USAGE'
Usage: ./scripts/run-server.sh [--quiet-checks] [--check-only]

Options:
  --quiet-checks  Suppress startup prerequisite output
  --check-only    Print prerequisite status and exit without starting the server
  --help          Show this help message
USAGE
      exit 0
      ;;
    *)
      APP_ARGS+=("$arg")
      ;;
  esac
done

load_env_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line="$raw_line"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    if [[ "$line" == export\ * ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" == *=* ]] || continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    [[ -n "$key" ]] || continue
    [[ -v "$key" ]] && continue

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]] || [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    else
      value="${value%% \#*}"
      value="${value%"${value##*[![:space:]]}"}"
    fi

    export "$key=$value"
  done < "$env_file"
}

load_env_file .env

if [[ -n "${KOKORO_CUDA_LIB_DIR:-}" ]]; then
  if [[ -d "$KOKORO_CUDA_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$KOKORO_CUDA_LIB_DIR:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  else
    printf 'warning: KOKORO_CUDA_LIB_DIR does not exist: %s\n' "$KOKORO_CUDA_LIB_DIR" >&2
  fi
fi

FAILURES=0
WARNINGS=0
COLOR_RESET=$'\033[0m'
COLOR_OK=$'\033[32m'
COLOR_WARN=$'\033[33m'
COLOR_ERROR=$'\033[31m'

print_line() {
  local level="$1"
  local label="$2"
  local message="$3"
  local colored_level="$level"
  if [[ "$SHOW_CHECKS" -eq 1 ]]; then
    case "$level" in
      ok)
        colored_level="${COLOR_OK}${level}${COLOR_RESET}"
        ;;
      warn)
        colored_level="${COLOR_WARN}${level}${COLOR_RESET}"
        ;;
      error)
        colored_level="${COLOR_ERROR}${level}${COLOR_RESET}"
        ;;
    esac
    printf '[%b] %-18s %s\n' "$colored_level" "$label" "$message"
  fi
}

ok() {
  print_line "ok" "$1" "$2"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  print_line "warn" "$1" "$2"
}

error() {
  FAILURES=$((FAILURES + 1))
  print_line "error" "$1" "$2"
}

section() {
  if [[ "$SHOW_CHECKS" -eq 1 ]]; then
    printf '\n%s\n' "$1"
  fi
}

check_command() {
  local label="$1"
  local command_name="$2"
  local required="$3"
  if command -v "$command_name" >/dev/null 2>&1; then
    local resolved
    resolved="$(command -v "$command_name")"
    ok "$label" "$resolved"
    return 0
  fi
  if [[ "$required" == "required" ]]; then
    error "$label" "command not found"
  else
    warn "$label" "command not found"
  fi
  return 1
}

check_file() {
  local label="$1"
  local file_path="$2"
  if [[ -f "$file_path" ]]; then
    ok "$label" "$file_path"
  else
    error "$label" "missing: $file_path"
  fi
}

check_python_version() {
  local label="python3"
  local version_output

  if command -v uv >/dev/null 2>&1; then
    label="uv-python"
    if ! version_output="$(uv run python - <<'PY'
import sys
major = sys.version_info.major
minor = sys.version_info.minor
print(f"{major}.{minor}")
print("supported" if (major, minor) in {(3, 12), (3, 13)} else "unsupported")
PY
)"; then
      error "$label" "failed to inspect interpreter version"
      return
    fi
  elif command -v python3 >/dev/null 2>&1; then
    if ! version_output="$(python3 - <<'PY'
import sys
major = sys.version_info.major
minor = sys.version_info.minor
print(f"{major}.{minor}")
print("supported" if (major, minor) in {(3, 12), (3, 13)} else "unsupported")
PY
)"; then
      error "$label" "failed to inspect interpreter version"
      return
    fi
  else
    error "python" "uv and python3 are both missing"
    return
  fi

  local version
  local status
  version="$(printf '%s\n' "$version_output" | sed -n '1p')"
  status="$(printf '%s\n' "$version_output" | sed -n '2p')"
  if [[ "$status" == "supported" ]]; then
    ok "$label" "version $version"
  else
    error "$label" "unsupported version $version (expected 3.12 or 3.13)"
  fi
}

check_cuda12_runtime_libs() {
  local label="$1"
  local base_dir="$2"
  local cublas_lt="$base_dir/libcublasLt.so.12"
  local cudart="$base_dir/libcudart.so.12"
  if [[ -e "$cublas_lt" && -e "$cudart" ]]; then
    ok "$label" "$base_dir"
  else
    warn "$label" "CUDA 12 runtime libs not found in $base_dir"
  fi
}

check_ffmpeg_rubberband() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg" "missing; opus output and pitch shifting will not work"
    return
  fi

  ok "ffmpeg" "$(command -v ffmpeg)"
  local filters_output
  filters_output="$(ffmpeg -hide_banner -filters 2>/dev/null)"
  if grep -Eq '(^|[[:space:]])rubberband([[:space:]]|$)' <<<"$filters_output"; then
    ok "rubberband" "ffmpeg filter available"
  else
    warn "rubberband" "ffmpeg found but rubberband filter is unavailable; non-zero pitch shifting will fail"
  fi
}

check_python_runtime_capabilities() {
  if ! command -v uv >/dev/null 2>&1; then
    return
  fi

  local probe_output
  if ! probe_output="$(uv run python - <<'PY'
from __future__ import annotations

import importlib.metadata as md
import os
import platform
from pathlib import Path

from app.config import (
    get_allowed_origins,
    get_api_key,
    get_allow_experimental_cuda_concurrency,
    get_require_auth,
    get_runtime_provider_mode,
    get_synthesis_queue_limit,
    get_synthesis_workers,
    get_web_ui_enabled,
)
from app.scheduler import build_scheduler_policy
from app.runtime import get_runtime_status

items: list[tuple[str, str, str]] = []
root = Path.cwd()
venv_python = root / '.venv' / 'bin' / 'python'
items.append(('venv', 'ok' if venv_python.exists() else 'warn', str(venv_python) if venv_python.exists() else '.venv not created yet'))

for package_name in ('kokoro-onnx', 'onnxruntime', 'onnxruntime-gpu'):
    try:
        version = md.version(package_name)
        items.append((package_name, 'ok', version))
    except md.PackageNotFoundError:
        level = 'warn' if package_name == 'onnxruntime-gpu' else 'error'
        items.append((package_name, level, 'not installed'))

if platform.system() == 'Linux':
    linux_stack: list[str] = []
    missing_linux_stack: list[str] = []
    for package_name in ('uvloop', 'httptools'):
        try:
            linux_stack.append(f'{package_name} {md.version(package_name)}')
        except md.PackageNotFoundError:
            missing_linux_stack.append(package_name)
    if linux_stack:
        items.append(('server-stack', 'ok', ', '.join(linux_stack)))
    if missing_linux_stack:
        items.append((
            'server-stack',
            'warn',
            f"missing {', '.join(missing_linux_stack)}; install 'uv sync --extra server' for faster Linux event loop/parser",
        ))

model_path = Path(os.getenv('KOKORO_MODEL_PATH', root / 'models' / 'kokoro-v1.0.onnx'))
voices_path = Path(os.getenv('KOKORO_VOICES_PATH', root / 'models' / 'voices-v1.0.bin'))
items.append(('model', 'ok' if model_path.exists() else 'error', str(model_path)))
items.append(('voices', 'ok' if voices_path.exists() else 'error', str(voices_path)))

web_ui_enabled = get_web_ui_enabled()
items.append(('web-ui', 'ok', 'enabled' if web_ui_enabled else 'disabled (headless/api mode)'))

require_auth = get_require_auth()
items.append(('auth', 'ok', 'required' if require_auth else 'disabled'))

allowed_origins = get_allowed_origins()
if allowed_origins:
    items.append(('cors', 'ok', ', '.join(allowed_origins)))
elif require_auth:
    items.append(('cors', 'warn', 'no cross-origin browser access configured; set KOKORO_ALLOWED_ORIGINS for a browser client on another origin'))
else:
    items.append(('cors', 'ok', 'same-origin only'))

try:
    configured_api_key = get_api_key()
except Exception as exc:
    configured_api_key = None
    items.append(('api-key', 'error', str(exc)))
else:
    if require_auth:
        items.append(('api-key', 'ok', 'configured'))
    elif configured_api_key:
        items.append(('api-key', 'ok', 'configured (inactive until auth is enabled)'))
    else:
        items.append(('api-key', 'ok', 'not configured'))

requested_provider = 'auto'
try:
    requested_provider = get_runtime_provider_mode()
except Exception as exc:
    items.append(('provider', 'error', str(exc)))
else:
    items.append(('provider', 'ok', requested_provider))

default_synthesis_workers = 2 if requested_provider == 'cpu' else 1
try:
    synthesis_workers = get_synthesis_workers(default=default_synthesis_workers)
except Exception as exc:
    items.append(('synth-workers', 'error', str(exc)))
else:
    is_explicit = 'KOKORO_SYNTH_WORKERS' in os.environ
    detail = str(synthesis_workers)
    if not is_explicit:
        detail = f'{detail} (default)'
    items.append(('synth-workers', 'ok', detail))

default_synthesis_queue = synthesis_workers * 4
try:
    synthesis_queue = get_synthesis_queue_limit(default=default_synthesis_queue)
except Exception as exc:
    items.append(('synth-queue', 'error', str(exc)))
else:
    is_explicit = 'KOKORO_SYNTH_QUEUE' in os.environ
    detail = str(synthesis_queue)
    if not is_explicit:
        detail = f'{detail} (default)'
    items.append(('synth-queue', 'ok', detail))

allow_experimental_gpu_concurrency = get_allow_experimental_cuda_concurrency()
items.append((
    'gpu-experimental',
    'ok' if allow_experimental_gpu_concurrency else 'ok',
    'enabled' if allow_experimental_gpu_concurrency else 'disabled',
))

try:
    status = get_runtime_status()
except Exception as exc:
    items.append(('runtime', 'error', str(exc)))
else:
    available = status.available_providers
    items.append(('providers', 'ok', ', '.join(available) if available else '(none)'))
    if status.attempted_providers:
        items.append(('attempted', 'ok', ', '.join(status.attempted_providers)))

    if status.active_providers:
        items.append(('active', 'ok', ', '.join(status.active_providers)))
    elif status.runtime_error:
        items.append(('runtime', 'error', status.runtime_error))
    else:
        items.append(('runtime', 'warn', 'no active providers'))

    if status.provider_fallback:
        items.append(('fallback', 'warn', f"requested {status.requested_provider}, active {', '.join(status.active_providers) or 'none'}"))
    if status.provider_error:
        level = 'error' if os.getenv('KOKORO_STRICT_PROVIDER', '').strip() in {'1', 'true', 'yes', 'on'} else 'warn'
        items.append(('provider-error', level, status.provider_error))
    if status.requested_provider != 'cpu':
        if 'CUDAExecutionProvider' in status.active_providers:
            items.append(('cuda', 'ok', 'CUDAExecutionProvider active'))
        elif 'CUDAExecutionProvider' in available:
            items.append(('cuda', 'warn', 'CUDAExecutionProvider detected but unusable; runtime is using CPU'))
        else:
            items.append(('cuda', 'warn', 'CUDAExecutionProvider unavailable; runtime will use CPU'))
    if status.active_providers == ['CPUExecutionProvider'] and 'KOKORO_SYNTH_WORKERS' not in os.environ:
        items.append(('cpu-tuning', 'ok', 'default worker cap is conservative; try KOKORO_SYNTH_WORKERS=3 on higher-core CPUs if serving concurrent requests'))

    active_provider = status.active_providers[0] if status.active_providers else None
    try:
        scheduler_policy = build_scheduler_policy(
            requested_provider=requested_provider,
            active_provider=active_provider,
            worker_limit=synthesis_workers,
            queue_limit=synthesis_queue,
            allow_experimental_gpu_concurrency=allow_experimental_gpu_concurrency,
        )
    except Exception as exc:
        items.append(('scheduler', 'error', str(exc)))
    else:
        items.append(('scheduler', 'ok', scheduler_policy.execution_model))
        if scheduler_policy.warning:
            items.append(('scheduler-warn', 'warn', scheduler_policy.warning))

for label, level, detail in items:
    print(f'{label}\t{level}\t{detail}')
PY
)"; then
    error "runtime" "uv runtime probe failed"
    return
  fi

  while IFS=$'\t' read -r label level detail; do
    [[ -z "$label" ]] && continue
    case "$level" in
      ok)
        ok "$label" "$detail"
        ;;
      warn)
        warn "$label" "$detail"
        ;;
      error)
        error "$label" "$detail"
        ;;
    esac
  done <<< "$probe_output"
}

check_nvidia_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi" "not found; NVIDIA GPU runtime not detected"
    return
  fi

  local nvidia_output
  if ! nvidia_output="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"; then
    warn "nvidia-smi" "present but failed to query GPU"
    return
  fi
  ok "nvidia-smi" "$nvidia_output"
}

check_cuda_runtime_paths() {
  if [[ -n "${KOKORO_CUDA_LIB_DIR:-}" ]]; then
    if [[ -d "$KOKORO_CUDA_LIB_DIR" ]]; then
      check_cuda12_runtime_libs "cuda-lib-dir" "$KOKORO_CUDA_LIB_DIR"
    else
      warn "cuda-lib-dir" "configured path is missing: $KOKORO_CUDA_LIB_DIR"
    fi
    return
  fi

  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcublasLt.so.12'; then
    ok "cuda-lib-path" "CUDA 12 runtime libs found via ldconfig"
  else
    warn "cuda-lib-path" "CUDA 12 runtime libs not found on standard linker paths"
  fi
}

section "Kokoro WebUI preflight"
check_command "uv" "uv" required
check_python_version
check_file "script" "$ROOT_DIR/scripts/run-server.sh"
check_ffmpeg_rubberband
check_nvidia_runtime
check_cuda_runtime_paths
check_python_runtime_capabilities

if [[ "$SHOW_CHECKS" -eq 1 ]]; then
  printf '\nSummary: %s error(s), %s warning(s)\n' "$FAILURES" "$WARNINGS"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  exit "$FAILURES"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  printf 'Startup blocked because required prerequisites are missing or invalid.\n' >&2
  exit 1
fi

exec uv run python -m app.main "${APP_ARGS[@]}"
