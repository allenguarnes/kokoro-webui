import { apiExampleButtons, apiExampleCode, apiExampleTitle } from "./dom.js";

const apiExamples = {
  "native-health": {
    title: "GET /api/health",
    request: "curl http://127.0.0.1:8000/api/health",
  },
  "native-speak": {
    title: "POST /api/speak",
    request:
      "curl -X POST http://127.0.0.1:8000/api/speak \\\n" +
      '  -H "Content-Type: application/json" \\\n' +
      "  -d '{\n" +
      '    "text": "Hello from Kokoro WebUI",\n' +
      '    "voice": "af_heart",\n' +
      '    "speed": 1.0,\n' +
      '    "pitch": 0.0,\n' +
      '    "lang": "en-us",\n' +
      '    "format": "wav"\n' +
      "  }' --output output.wav",
  },
  "native-chunk-plan": {
    title: "POST /api/chunk-plan",
    request:
      "curl -X POST http://127.0.0.1:8000/api/chunk-plan \\\n" +
      '  -H "Content-Type: application/json" \\\n' +
      "  -d '{\n" +
      '    "text": "Split this text into sentence-safe chunks.",\n' +
      '    "target_chunk_chars": 360\n' +
      "  }'",
  },
  "native-speak-stream": {
    title: "POST /api/speak-stream",
    request:
      "curl -N -X POST http://127.0.0.1:8000/api/speak-stream \\\n" +
      '  -H "Content-Type: application/json" \\\n' +
      "  -d '{\n" +
      '    "text": "Stream this as NDJSON chunks.",\n' +
      '    "voice": "af_heart",\n' +
      '    "speed": 1.0,\n' +
      '    "pitch": 0.0,\n' +
      '    "lang": "en-us",\n' +
      '    "format": "opus"\n' +
      "  }'",
  },
  "native-ws-speak-stream": {
    title: "WS /ws/speak-stream",
    request:
      "wscat -c ws://127.0.0.1:8000/ws/speak-stream\n" +
      '{"text":"Stream this over WebSocket.","voice":"af_heart","speed":1.0,"pitch":0.0,"lang":"en-us","format":"opus"}',
  },
  "openai-models": {
    title: "GET /v1/models",
    request:
      "curl http://127.0.0.1:8000/v1/models \\\n" +
      '  -H "Authorization: Bearer local-dev"',
  },
  "openai-model-kokoro": {
    title: "GET /v1/models/kokoro",
    request:
      "curl http://127.0.0.1:8000/v1/models/kokoro \\\n" +
      '  -H "Authorization: Bearer local-dev"',
  },
  "openai-audio-speech": {
    title: "POST /v1/audio/speech",
    request:
      "curl -X POST http://127.0.0.1:8000/v1/audio/speech \\\n" +
      '  -H "Authorization: Bearer local-dev" \\\n' +
      '  -H "Content-Type: application/json" \\\n' +
      "  -d '{\n" +
      '    "model": "kokoro",\n' +
      '    "input": "Hello from the OpenAI-compatible endpoint.",\n' +
      '    "voice": "af_heart+2.0",\n' +
      '    "response_format": "wav"\n' +
      "  }' --output speech.wav",
  },
};

function setApiExampleTitle(title) {
  if (!apiExampleTitle) {
    return;
  }
  const parts = String(title).trim().split(/\s+/);
  const method = parts.shift() || "";
  const route = parts.join(" ");
  apiExampleTitle.innerHTML = `<span class="api-method">${method}</span><span class="api-route">${route}</span>`;
}

export function setApiExample(exampleKey) {
  if (!apiExampleTitle || !apiExampleCode) {
    return;
  }
  const example = apiExamples[exampleKey];
  if (!example) {
    return;
  }
  setApiExampleTitle(example.title);
  apiExampleCode.textContent = example.request;
  apiExampleButtons.forEach((button) => {
    const isActive = button.dataset.apiExample === exampleKey;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

export function initializeApiExamples() {
  if (!apiExampleButtons.length) {
    return;
  }
  apiExampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setApiExample(button.dataset.apiExample);
    });
  });
  const defaultButton =
    apiExampleButtons.find((button) => button.classList.contains("is-active")) ||
    apiExampleButtons[0];
  if (defaultButton) {
    setApiExample(defaultButton.dataset.apiExample);
  }
}
