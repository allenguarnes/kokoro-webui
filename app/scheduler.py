from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, ParamSpec, Protocol, TypeVar, final

P = ParamSpec("P")
T = TypeVar("T")
SchedulerRuntimeKind = Literal["cpu", "gpu"]
SchedulerExecutionModel = Literal["shared-runtime", "session-pool"]


class SynthesisOverloadedError(RuntimeError):
    pass


@dataclass(frozen=True)
class SchedulerPolicy:
    requested_provider: str
    runtime_kind: SchedulerRuntimeKind
    execution_model: SchedulerExecutionModel
    worker_limit: int
    queue_limit: int
    prefers_serial_workers: bool
    experimental_gpu_concurrency: bool
    concurrency_note: str
    warning: str | None


@dataclass(frozen=True)
class SchedulerMetrics:
    worker_limit: int
    queue_limit: int
    capacity_limit: int
    reserved_jobs: int
    active_jobs: int
    queued_jobs: int
    available_slots: int
    admitted_jobs_total: int
    completed_jobs_total: int
    rejected_jobs_total: int
    queue_wait_last_ms: float
    queue_wait_avg_ms: float
    queue_wait_max_ms: float
    queue_wait_samples: int


class ExecutionBackend(Protocol):
    async def acquire_slot(self) -> None: ...

    def release_slot(self) -> None: ...

    async def execute(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T: ...

    def shutdown(self) -> None: ...


@final
class SharedRuntimeExecutionBackend:
    def __init__(self, *, worker_limit: int) -> None:
        self.worker_limit: int = worker_limit
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.worker_limit,
                thread_name_prefix="kokoro-synth",
            )
        )
        self._worker_slots: asyncio.Semaphore = asyncio.Semaphore(self.worker_limit)

    async def acquire_slot(self) -> None:
        _ = await self._worker_slots.acquire()

    def release_slot(self) -> None:
        self._worker_slots.release()

    async def execute(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        loop = asyncio.get_running_loop()
        task = functools.partial(function, *args, **kwargs)
        return await loop.run_in_executor(self._executor, task)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def build_execution_backend(policy: SchedulerPolicy) -> ExecutionBackend:
    # Session-pool execution is not implemented yet, but this factory gives it
    # a dedicated seam when the runtime layer is ready to lease isolated sessions.
    return SharedRuntimeExecutionBackend(worker_limit=policy.worker_limit)


def build_scheduler_policy(
    *,
    requested_provider: str,
    worker_limit: int,
    queue_limit: int,
    allow_experimental_gpu_concurrency: bool = False,
) -> SchedulerPolicy:
    if requested_provider == "cpu":
        return SchedulerPolicy(
            requested_provider=requested_provider,
            runtime_kind="cpu",
            execution_model="shared-runtime",
            worker_limit=worker_limit,
            queue_limit=queue_limit,
            prefers_serial_workers=False,
            experimental_gpu_concurrency=False,
            concurrency_note=(
                "CPU mode can use a small worker pool when serving concurrent requests."
            ),
            warning=None,
        )

    if (
        requested_provider == "cuda"
        and worker_limit > 1
        and not allow_experimental_gpu_concurrency
    ):
        raise RuntimeError(
            "KOKORO_SYNTH_WORKERS > 1 is blocked when KOKORO_PROVIDER=cuda because the current GPU path shares one runtime session. "
            + "Set KOKORO_ALLOW_EXPERIMENTAL_CUDA_CONCURRENCY=1 only if you are intentionally benchmarking shared-session GPU concurrency."
        )

    warning: str | None = None
    experimental_gpu_concurrency = worker_limit > 1
    if experimental_gpu_concurrency:
        if allow_experimental_gpu_concurrency:
            warning = "Experimental shared-session GPU concurrency is enabled. Benchmark latency, throughput, and VRAM usage before treating this as production-safe."
        else:
            warning = "GPU-preferred mode currently shares one runtime session. If this resolves to CUDA, KOKORO_SYNTH_WORKERS > 1 is an experimental tuning path."

    return SchedulerPolicy(
        requested_provider=requested_provider,
        runtime_kind="gpu",
        execution_model="shared-runtime",
        worker_limit=worker_limit,
        queue_limit=queue_limit,
        prefers_serial_workers=True,
        experimental_gpu_concurrency=experimental_gpu_concurrency,
        concurrency_note=(
            "GPU-preferred mode currently shares one runtime session; keep workers at 1 unless benchmarked."
        ),
        warning=warning,
    )


@final
class SynthesisScheduler:
    def __init__(self, *, policy: SchedulerPolicy) -> None:
        self.policy: SchedulerPolicy = policy
        self.worker_limit: int = policy.worker_limit
        self.queue_limit: int = policy.queue_limit
        self.capacity_limit: int = self.worker_limit + self.queue_limit
        self._backend: ExecutionBackend = build_execution_backend(policy)
        self._admission_tokens: asyncio.Queue[None] = asyncio.Queue(
            maxsize=self.capacity_limit
        )
        for _ in range(self.capacity_limit):
            self._admission_tokens.put_nowait(None)
        self._reserved_jobs: int = 0
        self._active_jobs: int = 0
        self._admitted_jobs_total: int = 0
        self._completed_jobs_total: int = 0
        self._rejected_jobs_total: int = 0
        self._queue_wait_last_ms: float = 0.0
        self._queue_wait_total_ms: float = 0.0
        self._queue_wait_max_ms: float = 0.0
        self._queue_wait_samples: int = 0

    def snapshot(self) -> SchedulerMetrics:
        queued_jobs = max(0, self._reserved_jobs - self._active_jobs)
        available_slots = max(0, self.capacity_limit - self._reserved_jobs)
        queue_wait_avg_ms = (
            self._queue_wait_total_ms / self._queue_wait_samples
            if self._queue_wait_samples
            else 0.0
        )
        return SchedulerMetrics(
            worker_limit=self.worker_limit,
            queue_limit=self.queue_limit,
            capacity_limit=self.capacity_limit,
            reserved_jobs=self._reserved_jobs,
            active_jobs=self._active_jobs,
            queued_jobs=queued_jobs,
            available_slots=available_slots,
            admitted_jobs_total=self._admitted_jobs_total,
            completed_jobs_total=self._completed_jobs_total,
            rejected_jobs_total=self._rejected_jobs_total,
            queue_wait_last_ms=round(self._queue_wait_last_ms, 2),
            queue_wait_avg_ms=round(queue_wait_avg_ms, 2),
            queue_wait_max_ms=round(self._queue_wait_max_ms, 2),
            queue_wait_samples=self._queue_wait_samples,
        )

    async def run(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        try:
            token = self._admission_tokens.get_nowait()
        except asyncio.QueueEmpty as exc:
            self._rejected_jobs_total += 1
            raise SynthesisOverloadedError(
                "Synthesis queue is full. Try again shortly."
            ) from exc

        self._reserved_jobs += 1
        self._admitted_jobs_total += 1
        admitted_at = time.perf_counter()
        try:
            await self._backend.acquire_slot()
            queue_wait_ms = (time.perf_counter() - admitted_at) * 1000
            self._queue_wait_last_ms = queue_wait_ms
            self._queue_wait_total_ms += queue_wait_ms
            self._queue_wait_max_ms = max(self._queue_wait_max_ms, queue_wait_ms)
            self._queue_wait_samples += 1
            self._active_jobs += 1
            try:
                return await self._backend.execute(function, *args, **kwargs)
            finally:
                self._active_jobs -= 1
                self._backend.release_slot()
        finally:
            self._reserved_jobs -= 1
            self._completed_jobs_total += 1
            self._admission_tokens.put_nowait(token)

    def shutdown(self) -> None:
        self._backend.shutdown()
