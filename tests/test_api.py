from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

import app.api as api
import app.audio as audio
import app.config as config
import app.runtime as runtime
from app.runtime import RuntimeStatus
from app.schemas import RenderedChunk, SynthesisRequest


def make_rendered_chunk(payload: SynthesisRequest, text: str) -> RenderedChunk:
    audio_bytes = f"{payload.voice}|{payload.pitch}|{payload.format}|{text}".encode(
        "utf-8"
    )
    return {
        "audio_bytes": audio_bytes,
        "media_type": "audio/ogg" if payload.format == "opus" else "audio/wav",
        "filename": "kokoro-output.ogg"
        if payload.format == "opus"
        else "kokoro-output.wav",
        "sample_rate": 24000,
        "duration_sec": 0.5,
    }


class ApiIntegrationTests(unittest.TestCase):
    client: TestClient | None = None
    existing_path: Path = Path()
    patch_stack: ExitStack | None = None

    def get_client(self) -> TestClient:
        client = self.client
        assert client is not None
        return client

    @override
    def setUp(self) -> None:
        self.existing_path = Path(__file__)
        self.patch_stack = ExitStack()
        _ = self.patch_stack.enter_context(
            patch.object(config, "get_runtime_provider_mode", return_value="cpu")
        )
        _ = self.patch_stack.enter_context(
            patch.object(config, "get_synthesis_workers", return_value=2)
        )
        _ = self.patch_stack.enter_context(
            patch.object(config, "get_synthesis_queue_limit", return_value=8)
        )
        self.client = TestClient(api.create_app())
        audio.ffmpeg_supports_rubberband.cache_clear()
        runtime.clear_runtime_caches()

    @override
    def tearDown(self) -> None:
        client = self.client
        if client is not None:
            client.close()
        patch_stack = self.patch_stack
        if patch_stack is not None:
            patch_stack.close()
        audio.ffmpeg_supports_rubberband.cache_clear()
        runtime.clear_runtime_caches()

    def test_health_reports_runtime_capabilities(self) -> None:
        fake_status = RuntimeStatus(
            requested_provider="auto",
            attempted_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
            provider_fallback=True,
            provider_error="Failed to load CUDA runtime.",
            runtime_error=None,
        )
        with (
            patch.object(config, "MODEL_PATH", self.existing_path),
            patch.object(config, "VOICES_PATH", self.existing_path),
            patch.object(
                runtime, "load_voice_names", return_value=["af_heart", "bf_alice"]
            ),
            patch.object(runtime, "get_runtime_status", return_value=fake_status),
            patch.object(audio, "ffmpeg_supports_rubberband", return_value=True),
            patch.object(runtime, "websocket_runtime_available", return_value=True),
        ):
            response = self.get_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, object], response.json())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["active_provider"], "CPUExecutionProvider")
        self.assertEqual(payload["active_providers"], ["CPUExecutionProvider"])
        self.assertTrue(cast(bool, payload["provider_fallback"]))
        self.assertEqual(payload["provider_error"], "Failed to load CUDA runtime.")
        self.assertIsNone(payload["runtime_error"])
        queue = cast(dict[str, object], payload["queue"])
        worker_limit = cast(int, queue["worker_limit"])
        queue_limit = cast(int, queue["queue_limit"])
        capacity_limit = cast(int, queue["capacity_limit"])
        self.assertGreaterEqual(worker_limit, 1)
        self.assertGreaterEqual(queue_limit, 0)
        self.assertEqual(capacity_limit, worker_limit + queue_limit)
        self.assertIn("queue_wait_last_ms", queue)
        self.assertIn("queue_wait_avg_ms", queue)
        self.assertIn("queue_wait_max_ms", queue)
        self.assertIn("queue_wait_samples", queue)

    def test_capabilities_reports_runtime_and_formats(self) -> None:
        fake_status = RuntimeStatus(
            requested_provider="auto",
            attempted_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
            provider_fallback=True,
            provider_error="Failed to load CUDA runtime.",
            runtime_error=None,
        )
        with (
            patch.object(
                runtime, "load_voice_names", return_value=["af_heart", "bf_alice"]
            ),
            patch.object(runtime, "get_runtime_status", return_value=fake_status),
            patch.object(audio, "ffmpeg_supports_rubberband", return_value=True),
            patch.object(runtime, "websocket_runtime_available", return_value=True),
        ):
            response = self.get_client().get("/api/capabilities")

        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, object], response.json())
        self.assertEqual(payload["voices"], ["af_heart", "bf_alice"])
        self.assertEqual(payload["requested_provider"], "auto")
        self.assertEqual(
            payload["attempted_providers"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(
            payload["available_providers"],
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
        )
        self.assertTrue(payload["pitch_shifting"])
        self.assertTrue(payload["websocket_streaming"])
        self.assertGreaterEqual(cast(int, payload["synthesis_workers"]), 1)
        self.assertGreaterEqual(cast(int, payload["synthesis_queue_limit"]), 0)
        scheduler = cast(dict[str, object], payload["scheduler"])
        self.assertEqual(scheduler["execution_model"], "shared-runtime")
        self.assertIn(scheduler["runtime_kind"], {"cpu", "gpu"})
        self.assertIsInstance(scheduler["concurrency_note"], str)
        self.assertEqual(
            scheduler["supported_execution_models"],
            ["shared-runtime", "session-pool"],
        )

    def test_health_reports_runtime_error_as_not_ready(self) -> None:
        fake_status = RuntimeStatus(
            requested_provider="cuda",
            attempted_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=[],
            provider_fallback=False,
            provider_error=None,
            runtime_error="Failed to allocate CUDA memory.",
        )
        with (
            patch.object(config, "MODEL_PATH", self.existing_path),
            patch.object(config, "VOICES_PATH", self.existing_path),
            patch.object(runtime, "load_voice_names", return_value=["af_heart"]),
            patch.object(runtime, "get_runtime_status", return_value=fake_status),
            patch.object(audio, "ffmpeg_supports_rubberband", return_value=True),
            patch.object(runtime, "websocket_runtime_available", return_value=True),
        ):
            response = self.get_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, object], response.json())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["runtime_error"], "Failed to allocate CUDA memory.")

    def test_queue_overload_returns_503(self) -> None:
        request_payload = {
            "text": "Hello from test.",
            "voice": "af_heart",
            "speed": 1.0,
            "pitch": 0.0,
            "lang": "en-us",
            "format": "wav",
        }
        started = threading.Event()
        release = threading.Event()

        def blocking_render(payload: SynthesisRequest, text: str) -> RenderedChunk:
            started.set()
            _ = release.wait(timeout=2.0)
            return make_rendered_chunk(payload, text)

        async def run_test() -> None:
            with (
                patch.object(config, "get_synthesis_workers", return_value=1),
                patch.object(config, "get_synthesis_queue_limit", return_value=0),
                patch.object(audio, "synthesize_chunk", side_effect=blocking_render),
            ):
                test_app = api.create_app()
                transport = httpx.ASGITransport(app=test_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    first_request = asyncio.create_task(
                        client.post("/api/speak", json=request_payload)
                    )
                    while not started.is_set():
                        await asyncio.sleep(0.01)
                    overload_response = await client.post(
                        "/api/speak", json=request_payload
                    )
                    self.assertEqual(overload_response.status_code, 503)
                    release.set()
                    first_response = await first_request
                    self.assertEqual(first_response.status_code, 200)

        asyncio.run(run_test())

    def test_native_speak_returns_audio_headers(self) -> None:
        request_payload = {
            "text": "Hello from test.",
            "voice": "af_heart",
            "speed": 1.0,
            "pitch": 0.0,
            "lang": "en-us",
            "format": "wav",
        }
        with patch.object(audio, "synthesize_chunk", side_effect=make_rendered_chunk):
            response = self.get_client().post("/api/speak", json=request_payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-audio-format"], "wav")
        self.assertEqual(response.headers["x-sample-rate"], "24000")
        self.assertEqual(response.content, b"af_heart|0.0|wav|Hello from test.")

    def test_native_stream_returns_meta_chunk_and_done_events(self) -> None:
        text = f"{'A' * 70}. {'B' * 70}."
        request_payload = {
            "text": text,
            "voice": "af_heart",
            "speed": 1.0,
            "pitch": 0.0,
            "lang": "en-us",
            "format": "wav",
            "target_chunk_chars": 80,
        }
        with patch.object(audio, "synthesize_chunk", side_effect=make_rendered_chunk):
            response = self.get_client().post("/api/speak-stream", json=request_payload)

        self.assertEqual(response.status_code, 200)
        events = cast(
            list[dict[str, object]],
            [json.loads(line) for line in response.text.splitlines() if line.strip()],
        )
        self.assertEqual(events[0]["type"], "meta")
        self.assertEqual(events[1]["type"], "chunk")
        self.assertEqual(events[2]["type"], "chunk")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[0]["total_chunks"], 2)

    def test_websocket_stream_returns_chunk_flow(self) -> None:
        request_payload = {
            "text": "Hello over websocket.",
            "voice": "af_heart",
            "speed": 1.0,
            "pitch": 0.0,
            "lang": "en-us",
            "format": "opus",
            "target_chunk_chars": 120,
        }
        with patch.object(audio, "synthesize_chunk", side_effect=make_rendered_chunk):
            with self.get_client().websocket_connect("/ws/speak-stream") as websocket:
                websocket.send_text(json.dumps(request_payload))
                meta = cast(dict[str, object], json.loads(websocket.receive_text()))
                chunk = cast(dict[str, object], json.loads(websocket.receive_text()))
                audio_bytes = websocket.receive_bytes()
                done = cast(dict[str, object], json.loads(websocket.receive_text()))

        self.assertEqual(meta["type"], "meta")
        self.assertEqual(chunk["type"], "chunk")
        self.assertEqual(chunk["format"], "opus")
        self.assertEqual(audio_bytes, b"af_heart|0.0|opus|Hello over websocket.")
        self.assertEqual(done["type"], "done")

    def test_openai_speech_accepts_voice_pitch_suffix(self) -> None:
        captured_requests: list[SynthesisRequest] = []

        def capture_request(payload: SynthesisRequest, text: str) -> RenderedChunk:
            captured_requests.append(payload)
            return make_rendered_chunk(payload, text)

        request_payload = {
            "model": "kokoro",
            "input": "OpenAI compatible request.",
            "voice": "af_heart+2.0",
            "response_format": "wav",
            "speed": 1.0,
        }
        with patch.object(audio, "synthesize_chunk", side_effect=capture_request):
            response = self.get_client().post("/v1/audio/speech", json=request_payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-audio-format"], "wav")
        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0].voice, "af_heart")
        self.assertEqual(captured_requests[0].pitch, 2.0)

    def test_openai_speech_rejects_out_of_range_voice_pitch_suffix(self) -> None:
        request_payload = {
            "model": "kokoro",
            "input": "Out of range pitch.",
            "voice": "af_heart+7.0",
            "response_format": "wav",
            "speed": 1.0,
        }

        response = self.get_client().post("/v1/audio/speech", json=request_payload)

        self.assertEqual(response.status_code, 400)
        payload = cast(dict[str, object], response.json())
        error = cast(dict[str, object], payload["error"])
        message = cast(str, error["message"])
        self.assertIn("voice pitch suffix must be between", message)

    def test_encode_opus_timeout_maps_to_runtime_error(self) -> None:
        with (
            patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(
                subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 0.01),
            ),
        ):
            with self.assertRaises(RuntimeError) as context:
                _ = audio.encode_opus(np.zeros(4, dtype=np.float32), 24000, "32k")

        self.assertIn("timed out", str(context.exception))


if __name__ == "__main__":
    _ = unittest.main()
