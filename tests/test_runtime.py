from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

import app.runtime as runtime
from app.runtime import KokoroEngineWithSession


class FakeSession:
    _providers: list[str]

    def __init__(self, providers: list[str]) -> None:
        self._providers = list(providers)

    def get_providers(self) -> list[str]:
        return list(self._providers)


class FakeSessionOptions:
    log_severity_level: int

    def __init__(self) -> None:
        self.log_severity_level = 0


class FakeEngine:
    sess: FakeSession

    def __init__(self, session: FakeSession) -> None:
        self.sess = session

    def create(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        lang: str,
    ) -> tuple[list[float], int]:
        _ = (text, voice, speed, lang)
        return [0.0], 24000


class FakeKokoroFactory:
    def __call__(self, model_path: str, voices_path: str) -> FakeEngine:
        _ = (model_path, voices_path)
        return FakeEngine(FakeSession(["CPUExecutionProvider"]))

    @classmethod
    def from_session(cls, session: FakeSession, voices_path: str) -> FakeEngine:
        _ = voices_path
        return FakeEngine(session)


class FakeOrt:
    available_providers: list[str]
    failures: dict[tuple[str, ...], RuntimeError]
    requests: list[list[str]]
    session_options: list[FakeSessionOptions]
    logger_severity: list[int]

    def __init__(
        self,
        available_providers: list[str],
        failures: dict[tuple[str, ...], RuntimeError] | None = None,
    ) -> None:
        self.available_providers = list(available_providers)
        self.failures = failures or {}
        self.requests = []
        self.session_options = []
        self.logger_severity = []

    def get_available_providers(self) -> list[str]:
        return list(self.available_providers)

    def set_default_logger_severity(self, severity: int) -> None:
        self.logger_severity.append(severity)

    def SessionOptions(self) -> FakeSessionOptions:
        options = FakeSessionOptions()
        self.session_options.append(options)
        return options

    def InferenceSession(
        self,
        model_path: str,
        sess_options: FakeSessionOptions | None = None,
        providers: list[str] | None = None,
    ) -> FakeSession:
        _ = (model_path, sess_options)
        resolved = list(providers or [])
        self.requests.append(resolved)
        failure = self.failures.get(tuple(resolved))
        if failure is not None:
            raise failure
        return FakeSession(resolved)


class RuntimeSelectionTests(unittest.TestCase):
    existing_path: Path = Path()

    @override
    def setUp(self) -> None:
        self.existing_path = Path(__file__)
        runtime.clear_runtime_caches()

    @override
    def tearDown(self) -> None:
        runtime.clear_runtime_caches()

    def test_auto_prefers_cpu_when_cuda_provider_is_unavailable(self) -> None:
        fake_ort = FakeOrt(["CPUExecutionProvider"])
        with (
            patch.object(runtime, "KokoroRuntime", FakeKokoroFactory()),
            patch.object(runtime, "ort", fake_ort),
            patch.object(runtime, "MODEL_PATH", self.existing_path),
            patch.object(runtime, "VOICES_PATH", self.existing_path),
            patch.object(runtime, "get_runtime_provider_mode", return_value="auto"),
            patch.object(runtime, "get_runtime_provider_strict", return_value=False),
        ):
            status = runtime.get_runtime_status()

        self.assertEqual(fake_ort.requests, [["CPUExecutionProvider"]])
        self.assertEqual(fake_ort.logger_severity, [4])
        self.assertEqual(fake_ort.session_options[0].log_severity_level, 4)
        self.assertEqual(status.requested_provider, "auto")
        self.assertEqual(status.active_providers, ["CPUExecutionProvider"])
        self.assertFalse(status.provider_fallback)
        self.assertIsNone(status.provider_error)

    def test_auto_falls_back_to_cpu_when_cuda_session_fails(self) -> None:
        fake_ort = FakeOrt(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            failures={
                ("CUDAExecutionProvider", "CPUExecutionProvider"): RuntimeError(
                    "CUDA init failed"
                )
            },
        )
        with (
            patch.object(runtime, "KokoroRuntime", FakeKokoroFactory()),
            patch.object(runtime, "ort", fake_ort),
            patch.object(runtime, "MODEL_PATH", self.existing_path),
            patch.object(runtime, "VOICES_PATH", self.existing_path),
            patch.object(runtime, "get_runtime_provider_mode", return_value="auto"),
            patch.object(runtime, "get_runtime_provider_strict", return_value=False),
        ):
            status = runtime.get_runtime_status()
            tts = cast(KokoroEngineWithSession, runtime.get_tts())

        self.assertEqual(
            fake_ort.requests,
            [
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ["CPUExecutionProvider"],
            ],
        )
        self.assertEqual(status.active_providers, ["CPUExecutionProvider"])
        self.assertTrue(status.provider_fallback)
        self.assertEqual(status.provider_error, "CUDA init failed")
        self.assertEqual(tts.sess.get_providers(), ["CPUExecutionProvider"])

    def test_cuda_strict_surfaces_runtime_error(self) -> None:
        fake_ort = FakeOrt(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            failures={
                ("CUDAExecutionProvider", "CPUExecutionProvider"): RuntimeError(
                    "Failed to load libcublasLt.so.12"
                )
            },
        )
        with (
            patch.object(runtime, "KokoroRuntime", FakeKokoroFactory()),
            patch.object(runtime, "ort", fake_ort),
            patch.object(runtime, "MODEL_PATH", self.existing_path),
            patch.object(runtime, "VOICES_PATH", self.existing_path),
            patch.object(runtime, "get_runtime_provider_mode", return_value="cuda"),
            patch.object(runtime, "get_runtime_provider_strict", return_value=True),
        ):
            status = runtime.get_runtime_status()

        self.assertEqual(status.requested_provider, "cuda")
        self.assertEqual(status.active_providers, [])
        self.assertFalse(status.provider_fallback)
        self.assertIn("libcublasLt", status.runtime_error or "")


if __name__ == "__main__":
    _ = unittest.main()
