import {
  audioDuration,
  chunkBar,
  chunkProgress,
  fileSize,
  genTime,
  pauseMsInput,
  player,
  speedInput,
} from "./dom.js";
import { appState } from "./state.js";
import {
  buildSynthesisPayload,
  formatBytes,
  formatDuration,
  formatSignedSemitones,
  formatWavRateLabel,
  getPitchSemitones,
  setStatus,
} from "./ui.js";

function decodeBase64ToBytes(base64Text) {
  const binary = atob(base64Text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function createWavHeader(sampleRate, pcmByteLength) {
  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);
  const writeAscii = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + pcmByteLength, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, pcmByteLength, true);
  return buffer;
}

function createPlaybackBlob(chunkMeta, bytes) {
  if (chunkMeta.format === "pcm") {
    const sampleRate = Number(chunkMeta.sample_rate) || 24000;
    return new Blob([createWavHeader(sampleRate, bytes.byteLength), bytes], {
      type: "audio/wav",
    });
  }
  return new Blob([bytes], { type: chunkMeta.mime_type });
}

function createQueueEntry(meta, blob) {
  return {
    index: meta.chunk_index,
    totalChunks: meta.total_chunks,
    blob,
    bytes: meta.bytes,
    durationSec: meta.duration_sec,
    synthMs: meta.synth_ms,
    format: meta.format,
    opus_bitrate: meta.opus_bitrate,
    sample_rate: meta.sample_rate,
  };
}

function getInterChunkPauseMs(speed, pauseValue) {
  if (pauseValue > 0) {
    return pauseValue;
  }
  const pause = Math.round(260 / speed);
  return Math.max(140, Math.min(420, pause));
}

export function resetPlaybackState() {
  if (appState.chunkState?.queueWaiter) {
    appState.chunkState.streamDone = true;
    appState.chunkState.queueWaiter();
    appState.chunkState.queueWaiter = null;
  }
  appState.playbackToken += 1;
  appState.chunkState = null;
  if (appState.streamAbortController) {
    appState.streamAbortController.abort();
    appState.streamAbortController = null;
  }
  if (appState.activeSocket) {
    appState.activeSocket.close(1000, "reset");
    appState.activeSocket = null;
  }
  if (appState.currentObjectUrl) {
    URL.revokeObjectURL(appState.currentObjectUrl);
    appState.currentObjectUrl = null;
  }
  player.pause();
  player.currentTime = 0;
  player.removeAttribute("src");
  player.load();
}

export function handlePageExit() {
  resetPlaybackState();
}

function waitForPlaybackEndOrAbort(token) {
  return new Promise((resolve) => {
    let settled = false;
    let frameId = null;

    const onEnded = () => finish();

    function finish() {
      if (settled) {
        return;
      }
      settled = true;
      player.removeEventListener("ended", onEnded);
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
      resolve();
    }

    function watchToken() {
      if (token !== appState.playbackToken) {
        finish();
        return;
      }
      frameId = window.requestAnimationFrame(watchToken);
    }

    player.addEventListener("ended", onEnded, { once: true });
    watchToken();
  });
}

function notifyQueue(state) {
  if (state.queueWaiter) {
    state.queueWaiter();
    state.queueWaiter = null;
  }
}

function buildAuthHeaders() {
  if (!appState.apiKey) {
    return {};
  }
  return {
    Authorization: `Bearer ${appState.apiKey}`,
  };
}

async function waitForQueue(state, token) {
  if (
    token !== appState.playbackToken ||
    state.queue.length ||
    state.streamDone ||
    state.streamError
  ) {
    return;
  }
  await new Promise((resolve) => {
    state.queueWaiter = resolve;
  });
}

export async function readStreamIntoQueue(state, token) {
  appState.streamAbortController = new AbortController();
  const response = await fetch("/api/speak-stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...buildAuthHeaders(),
    },
    body: JSON.stringify(buildSynthesisPayload()),
    signal: appState.streamAbortController.signal,
  });

  if (response.status === 401) {
    const error = new Error("Authentication failed.");
    error.name = "AuthRequiredError";
    throw error;
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Streaming synthesis failed.");
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming reader is unavailable in this browser.");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (token === appState.playbackToken) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const message = JSON.parse(line);
      if (message.type === "meta") {
        state.totalChunks = message.total_chunks;
        chunkProgress.textContent = `0 / ${state.totalChunks}`;
        setStatus(
          `Chunk plan ready with ${state.totalChunks} sentence-safe chunks${
            message.pitch !== 0
              ? ` at ${formatSignedSemitones(Number(message.pitch))}`
              : ""
          }.`,
        );
      } else if (message.type === "chunk") {
        state.queue.push(
          createQueueEntry(
            message,
            createPlaybackBlob(
              message,
              decodeBase64ToBytes(message.audio_base64),
            ),
          ),
        );
        notifyQueue(state);
      } else if (message.type === "error") {
        state.streamError = message.detail || "Streaming synthesis failed.";
        notifyQueue(state);
        return;
      } else if (message.type === "done") {
        state.streamDone = true;
        notifyQueue(state);
        return;
      }
    }
  }

  state.streamDone = true;
  notifyQueue(state);
}

export function readWebSocketIntoQueue(state, token) {
  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/speak-stream`;
    const socket = new WebSocket(wsUrl);
    socket.binaryType = "arraybuffer";
    appState.activeSocket = socket;

    socket.onopen = () => {
      const payload = buildSynthesisPayload();
      if (appState.apiKey) {
        payload.api_key = appState.apiKey;
      }
      socket.send(JSON.stringify(payload));
    };

    socket.onmessage = (event) => {
      if (token !== appState.playbackToken || !appState.chunkState) {
        return;
      }
      if (typeof event.data === "string") {
        const message = JSON.parse(event.data);
        if (message.type === "meta") {
          state.totalChunks = message.total_chunks;
          chunkProgress.textContent = `0 / ${state.totalChunks}`;
          setStatus(
            `Chunk plan ready with ${state.totalChunks} sentence-safe chunks${
              message.pitch !== 0
                ? ` at ${formatSignedSemitones(Number(message.pitch))}`
                : ""
            }.`,
          );
        } else if (message.type === "chunk") {
          state.pendingChunkMeta = message;
        } else if (message.type === "error") {
          state.streamError = message.detail || "WebSocket streaming failed.";
          notifyQueue(state);
        } else if (message.type === "done") {
          state.streamDone = true;
          notifyQueue(state);
        }
        return;
      }

      if (!(event.data instanceof ArrayBuffer) || !state.pendingChunkMeta) {
        state.streamError =
          "WebSocket stream lost chunk metadata before audio bytes arrived.";
        notifyQueue(state);
        return;
      }

      const chunkMeta = state.pendingChunkMeta;
      state.pendingChunkMeta = null;
      state.queue.push(
        createQueueEntry(chunkMeta, createPlaybackBlob(chunkMeta, event.data)),
      );
      notifyQueue(state);
    };

    socket.onerror = () => {
      reject(new Error("WebSocket transport failed."));
    };

    socket.onclose = () => {
      if (
        token === appState.playbackToken &&
        appState.chunkState &&
        !state.streamDone
      ) {
        state.streamDone = true;
        notifyQueue(state);
      }
      if (appState.activeSocket === socket) {
        appState.activeSocket = null;
      }
      resolve();
    };
  });
}

export async function playQueueFromStream(state, token) {
  while (token === appState.playbackToken) {
    if (!state.queue.length) {
      await waitForQueue(state, token);
      if (token !== appState.playbackToken) {
        return;
      }
    }

    if (state.streamError) {
      throw new Error(state.streamError);
    }

    if (!state.queue.length) {
      if (state.streamDone) {
        return;
      }
      continue;
    }

    const chunk = state.queue.shift();
    state.playedChunks += 1;
    state.totalElapsedMs += chunk.synthMs;
    state.totalBytes += chunk.bytes;
    state.totalDurationSec += chunk.durationSec;

    chunkProgress.textContent = `${state.playedChunks} / ${chunk.totalChunks}`;
    chunkBar.max = state.totalChunks || 1;
    chunkBar.value = state.playedChunks;
    genTime.textContent = `${state.totalElapsedMs.toFixed(0)} ms`;
    fileSize.textContent = formatBytes(state.totalBytes);
    audioDuration.textContent = formatDuration(state.totalDurationSec);

    if (appState.currentObjectUrl) {
      URL.revokeObjectURL(appState.currentObjectUrl);
    }
    appState.currentObjectUrl = URL.createObjectURL(chunk.blob);
    player.src = appState.currentObjectUrl;

    setStatus(
      `Playing chunk ${chunk.index + 1} of ${chunk.totalChunks} as ${chunk.format.toUpperCase()}${
        chunk.opus_bitrate
          ? ` @ ${chunk.opus_bitrate}`
          : formatWavRateLabel(chunk.sample_rate)
      }${
        getPitchSemitones() !== 0
          ? ` at ${formatSignedSemitones(getPitchSemitones())}`
          : ""
      }`,
    );

    const endedPromise = waitForPlaybackEndOrAbort(token);
    try {
      await player.play();
    } catch (error) {
      if (token !== appState.playbackToken) {
        return;
      }
      throw error;
    }
    await endedPromise;

    if (state.playedChunks < chunk.totalChunks) {
      const pauseMs = getInterChunkPauseMs(
        Number(speedInput.value),
        Number(pauseMsInput.value),
      );
      setStatus(`Queue pause ${pauseMs} ms before next chunk.`);
      await new Promise((resolve) => window.setTimeout(resolve, pauseMs));
    }
  }
}
