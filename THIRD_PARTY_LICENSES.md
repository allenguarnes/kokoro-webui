# Third-Party Licenses

This project is licensed under Apache-2.0. It depends on third-party software and model assets with their own licenses.

## Python Dependencies

- `fastapi` - MIT
- `kokoro-onnx` - MIT
- `uvicorn` - BSD-3-Clause
- `websockets` - BSD-3-Clause
- `soundfile` - BSD-3-Clause
- `onnxruntime` (transitive runtime) - MIT

See each package repository or distributed metadata for authoritative license texts.

## Model Assets

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

Source: `https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0`

Upstream Kokoro model family license: Apache-2.0 (`https://huggingface.co/hexgrad/Kokoro-82M`)

## System Runtime Tools

This project invokes external system binaries for audio processing when available:

- `ffmpeg`
- `rubberband` filter (when present in ffmpeg build)

If you distribute binaries, containers, or packaged applications that include these tools, you are responsible for complying with their license terms for the specific build you distribute.

Reference:

- FFmpeg legal: `https://ffmpeg.org/legal.html`
- Rubber Band licensing: `https://breakfastquay.com/rubberband/license.html`
