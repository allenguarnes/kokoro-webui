from __future__ import annotations

import asyncio
import threading
import unittest

from app.scheduler import (
    SynthesisOverloadedError,
    SynthesisScheduler,
    build_scheduler_policy,
)


class SchedulerPolicyTests(unittest.TestCase):
    def test_cpu_policy_allows_parallel_workers(self) -> None:
        policy = build_scheduler_policy(
            requested_provider="cpu",
            active_provider="CPUExecutionProvider",
            worker_limit=3,
            queue_limit=12,
        )

        self.assertEqual(policy.runtime_kind, "cpu")
        self.assertEqual(policy.execution_model, "shared-runtime")
        self.assertFalse(policy.prefers_serial_workers)
        self.assertFalse(policy.experimental_gpu_concurrency)
        self.assertIn("worker pool", policy.concurrency_note)
        self.assertIsNone(policy.warning)

    def test_gpu_policy_blocks_when_auto_resolves_to_gpu_without_opt_in(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            _ = build_scheduler_policy(
                requested_provider="auto",
                active_provider="CUDAExecutionProvider",
                worker_limit=2,
                queue_limit=8,
            )

        self.assertIn("active runtime provider is GPU-backed", str(context.exception))

    def test_gpu_policy_warns_when_auto_workers_exceed_one_without_active_gpu(
        self,
    ) -> None:
        policy = build_scheduler_policy(
            requested_provider="auto",
            active_provider=None,
            worker_limit=2,
            queue_limit=8,
        )

        self.assertEqual(policy.runtime_kind, "gpu")
        self.assertEqual(policy.execution_model, "shared-runtime")
        self.assertTrue(policy.prefers_serial_workers)
        self.assertTrue(policy.experimental_gpu_concurrency)
        self.assertIn("keep workers at 1", policy.concurrency_note)
        self.assertIn("experimental tuning path", policy.warning or "")

    def test_cuda_policy_blocks_multiple_workers_without_opt_in(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            _ = build_scheduler_policy(
                requested_provider="cuda",
                active_provider="CUDAExecutionProvider",
                worker_limit=2,
                queue_limit=8,
            )

        self.assertIn("KOKORO_SYNTH_WORKERS > 1 is blocked", str(context.exception))

    def test_cuda_policy_allows_multiple_workers_with_opt_in(self) -> None:
        policy = build_scheduler_policy(
            requested_provider="cuda",
            active_provider="CUDAExecutionProvider",
            worker_limit=2,
            queue_limit=8,
            allow_experimental_gpu_concurrency=True,
        )

        self.assertEqual(policy.runtime_kind, "gpu")
        self.assertTrue(policy.experimental_gpu_concurrency)
        self.assertIn(
            "Experimental shared-session GPU concurrency", policy.warning or ""
        )


class SchedulerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_snapshot_tracks_active_and_queued_jobs(self) -> None:
        policy = build_scheduler_policy(
            requested_provider="cpu",
            active_provider="CPUExecutionProvider",
            worker_limit=1,
            queue_limit=1,
        )
        scheduler = SynthesisScheduler(policy=policy)
        started = threading.Event()
        release = threading.Event()

        def blocking_job(label: str) -> str:
            _ = label
            started.set()
            _ = release.wait(timeout=2.0)
            return "done"

        try:
            first_task = asyncio.create_task(
                scheduler.run_interactive(blocking_job, "first")
            )
            while not started.is_set():
                await asyncio.sleep(0.01)

            second_task = asyncio.create_task(
                scheduler.run_interactive(blocking_job, "second")
            )
            await asyncio.sleep(0.05)

            metrics = scheduler.snapshot()
            self.assertEqual(metrics.worker_limit, 1)
            self.assertEqual(metrics.queue_limit, 1)
            self.assertEqual(metrics.reserved_jobs, 2)
            self.assertEqual(metrics.active_jobs, 1)
            self.assertEqual(metrics.queued_jobs, 1)
            self.assertEqual(metrics.available_slots, 0)
            self.assertEqual(metrics.queue_wait_samples, 1)

            with self.assertRaises(SynthesisOverloadedError):
                _ = await scheduler.run_interactive(blocking_job, "third")

            release.set()
            self.assertEqual(await first_task, "done")
            self.assertEqual(await second_task, "done")

            final_metrics = scheduler.snapshot()
            self.assertEqual(final_metrics.reserved_jobs, 0)
            self.assertEqual(final_metrics.active_jobs, 0)
            self.assertEqual(final_metrics.queued_jobs, 0)
            self.assertEqual(final_metrics.admitted_jobs_total, 2)
            self.assertEqual(final_metrics.completed_jobs_total, 2)
            self.assertEqual(final_metrics.rejected_jobs_total, 1)
            self.assertEqual(final_metrics.queue_wait_samples, 2)
            self.assertGreater(final_metrics.queue_wait_max_ms, 0.0)
            self.assertGreater(final_metrics.queue_wait_avg_ms, 0.0)
        finally:
            scheduler.shutdown()

    async def test_stream_jobs_leave_reserved_capacity_for_interactive_jobs(
        self,
    ) -> None:
        policy = build_scheduler_policy(
            requested_provider="cpu",
            active_provider="CPUExecutionProvider",
            worker_limit=1,
            queue_limit=1,
        )
        scheduler = SynthesisScheduler(policy=policy)
        first_started = threading.Event()
        release = threading.Event()

        def blocking_job(label: str) -> str:
            if label == "stream-1":
                first_started.set()
            _ = release.wait(timeout=2.0)
            return label

        try:
            first_stream = asyncio.create_task(
                scheduler.run_stream(blocking_job, "stream-1")
            )
            while not first_started.is_set():
                await asyncio.sleep(0.01)

            with self.assertRaises(SynthesisOverloadedError):
                _ = await scheduler.run_stream(blocking_job, "stream-2")

            interactive_job = asyncio.create_task(
                scheduler.run_interactive(blocking_job, "interactive-1")
            )
            await asyncio.sleep(0.05)

            metrics = scheduler.snapshot()
            self.assertEqual(metrics.interactive_reserve_slots, 1)
            self.assertEqual(metrics.stream_capacity_limit, 1)
            self.assertEqual(metrics.reserved_jobs, 2)
            self.assertEqual(metrics.queued_jobs, 1)

            release.set()
            self.assertEqual(await first_stream, "stream-1")
            self.assertEqual(await interactive_job, "interactive-1")
        finally:
            scheduler.shutdown()
