# API Examples

Practical examples for using the Kokoro WebUI API.

## Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

## Single Audio Render

### WAV (default)

```bash
curl -X POST http://127.0.0.1:8000/api/speak \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello world",
    "voice": "af_heart",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output output.wav
```

### With pitch adjustment

```bash
curl -X POST http://127.0.0.1:8000/api/speak \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello world",
    "voice": "af_heart",
    "speed": 1.0,
    "pitch": -1.0,
    "format": "wav"
  }' \
  --output output.wav
```

### Compressed Opus

```bash
curl -X POST http://127.0.0.1:8000/api/speak \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello world",
    "voice": "af_heart",
    "format": "opus",
    "opus_bitrate": "32k"
  }' \
  --output output.opus
```

## Streaming Audio

### NDJSON stream (server sends chunks)

```bash
curl -N -X POST http://127.0.0.1:8000/api/speak-stream \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "First sentence. Second sentence. Third sentence.",
    "voice": "af_heart",
    "speed": 1.0,
    "pitch": 1.5,
    "format": "opus",
    "opus_bitrate": "32k",
    "target_chunk_chars": 120
  }'
```

### WebSocket stream (real-time)

Requires [`websocat`](https://github.com/vi/websocat):

```bash
printf '%s\n' '{
  "text": "First sentence. Second sentence. Third sentence.",
  "voice": "af_heart",
  "speed": 1.0,
  "pitch": 0.0,
  "format": "wav",
  "target_chunk_chars": 120
}' | websocat ws://127.0.0.1:8000/ws/speak-stream
```

## OpenAI-Compatible API

### Basic request

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Hello from the OpenAI-compatible endpoint",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  --output output.wav
```

### With pitch (using voice suffix)

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Higher pitch voice",
    "voice": "af_heart+2.0",
    "response_format": "wav"
  }' \
  --output output.wav
```

### Stream to player (WAV)

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Streaming audio directly to playback",
    "voice": "af_heart",
    "response_format": "wav"
  }' | ffplay -i -
```

### Stream raw PCM

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Raw PCM streaming",
    "voice": "af_heart",
    "response_format": "pcm"
  }' | ffplay -f s16le -ar 24000 -ch_layout mono -i -
```

## Python Examples

### Simple synthesis

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/speak",
    json={
        "text": "Hello world",
        "voice": "af_heart",
        "format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Streaming with playback

```python
import requests
import json
import base64
import soundfile as sf
import io

response = requests.post(
    "http://127.0.0.1:8000/api/speak-stream",
    json={
        "text": "Streaming example with multiple sentences.",
        "voice": "af_heart",
        "format": "wav"
    },
    stream=True
)

audio_chunks = []
for line in response.iter_lines():
    if line:
        msg = json.loads(line)
        if msg["type"] == "chunk":
            audio_data = base64.b64decode(msg["audio_base64"])
            audio_chunks.append(audio_data)
        elif msg["type"] == "done":
            break

# Combine and play
full_audio = b"".join(audio_chunks)
data, samplerate = sf.read(io.BytesIO(full_audio))
# ... play audio
```

### WebSocket client

```python
import asyncio
import json
import base64
import websockets

async def stream_audio():
    uri = "ws://127.0.0.1:8000/ws/speak-stream"
    
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "text": "WebSocket streaming example",
            "voice": "af_heart",
            "format": "wav"
        }))
        
        audio_chunks = []
        async for message in ws:
            msg = json.loads(message)
            
            if msg["type"] == "chunk":
                audio_data = base64.b64decode(msg["audio_base64"])
                audio_chunks.append(audio_data)
            elif msg["type"] == "done":
                break
        
        # Process audio_chunks...
        print(f"Received {len(audio_chunks)} chunks")

asyncio.run(stream_audio())
```

## JavaScript Examples

### Fetch API (simple)

```javascript
const response = await fetch('http://127.0.0.1:8000/api/speak', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Hello world',
    voice: 'af_heart',
    format: 'wav'
  })
});

const blob = await response.blob();
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

### WebSocket streaming

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/ws/speak-stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    text: 'WebSocket streaming in JavaScript',
    voice: 'af_heart',
    format: 'wav'
  }));
};

const audioChunks = [];

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  switch (msg.type) {
    case 'meta':
      console.log(`Expecting ${msg.total_chunks} chunks`);
      break;
    case 'chunk':
      audioChunks.push(msg.audio_base64);
      break;
    case 'done':
      console.log('Streaming complete');
      // Play or save audio...
      ws.close();
      break;
    case 'error':
      console.error('Error:', msg.detail);
      ws.close();
      break;
  }
};
```

## OpenAI Client Examples

### Using the official OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # Or your KOKORO_API_KEY if auth is enabled
)

response = client.audio.speech.create(
    model="kokoro",
    input="Using the official OpenAI client",
    voice="af_heart",
    response_format="wav"
)

response.stream_to_file("output.wav")
```

### With LangChain

```python
from langchain_community.tools import OpenAITTS

# Configure for local Kokoro
tts = OpenAITTS(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

audio = tts.invoke({
    "input": "LangChain integration",
    "voice": "af_heart"
})
```
