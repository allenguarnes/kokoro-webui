from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app.audio as audio
import app.config as config
import app.main as main
import app.runtime as runtime
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

    def get_client(self) -> TestClient:
        client = self.client
        assert client is not None
        return client

    @override
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.existing_path = Path(__file__)
        audio.ffmpeg_supports_rubberband.cache_clear()
        runtime.get_tts.cache_clear()

    @override
    def tearDown(self) -> None:
        client = self.client
        if client is not None:
            client.close()
        audio.ffmpeg_supports_rubberband.cache_clear()
        runtime.get_tts.cache_clear()

    def test_health_reports_runtime_capabilities(self) -> None:
        with (
            patch.object(config, "MODEL_PATH", self.existing_path),
            patch.object(config, "VOICES_PATH", self.existing_path),
            patch.object(
                runtime, "load_voice_names", return_value=["af_heart", "bf_alice"]
            ),
            patch.object(audio, "ffmpeg_supports_rubberband", return_value=True),
            patch.object(runtime, "websocket_runtime_available", return_value=True),
        ):
            response = self.get_client().get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, object], response.json())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["voices"], ["af_heart", "bf_alice"])
        self.assertTrue(payload["pitch_shifting"])
        self.assertTrue(payload["websocket_streaming"])

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
                done = cast(dict[str, object], json.loads(websocket.receive_text()))

        self.assertEqual(meta["type"], "meta")
        self.assertEqual(chunk["type"], "chunk")
        self.assertEqual(chunk["format"], "opus")
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
