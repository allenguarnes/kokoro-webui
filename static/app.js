const form = document.getElementById("ttsForm");
const textInput = document.getElementById("text");
const voiceInput = document.getElementById("voice");
const langInput = document.getElementById("lang");
const transportInput = document.getElementById("transport");
const formatInput = document.getElementById("format");
const formatQualityInput = document.getElementById("formatQuality");
const formatQualityLabel = document.getElementById("formatQualityLabel");
const speedInput = document.getElementById("speed");
const speedValue = document.getElementById("speedValue");
const pitchInput = document.getElementById("pitch");
const pitchValue = document.getElementById("pitchValue");
const chunkTargetInput = document.getElementById("chunkTarget");
const pauseMsInput = document.getElementById("pauseMs");
const pauseMode = document.getElementById("pauseMode");
const statusText = document.getElementById("statusText");
const errorText = document.getElementById("errorText");
const submitButton = document.getElementById("submitButton");
const submitButtonLabel = submitButton.querySelector(".button-label");
const stopButton = document.getElementById("stopButton");
const exportButton = document.getElementById("exportButton");
const exportButtonLabel = exportButton.querySelector(".button-label");
const player = document.getElementById("player");
const systemStatus = document.getElementById("systemStatus");
const themeToggle = document.getElementById("themeToggle");
const genTime = document.getElementById("genTime");
const charCount = document.getElementById("charCount");
const wordCount = document.getElementById("wordCount");
const audioDuration = document.getElementById("audioDuration");
const fileSize = document.getElementById("fileSize");
const chunkProgress = document.getElementById("chunkProgress");
const chunkBar = document.getElementById("chunkBar");
const streamMode = document.getElementById("streamMode");

let currentObjectUrl = null;
let playbackToken = 0;
let chunkState = null;
let streamAbortController = null;
let activeSocket = null;
let currentTheme = "dark";
const customSelectRegistry = new Map();
let lastExportRequest = null;
let availableOpusBitrates = ["16k", "24k", "32k", "48k"];
let availableWavSampleRates = [
  "native",
  "16000",
  "22050",
  "24000",
  "44100",
  "48000",
];
let selectedOpusBitrate = "32k";
let selectedWavSampleRate = "native";
let pitchShiftingAvailable = true;
let maxPitchSemitones = 6;

function formatVoiceLabel(voice) {
  const [, rawName = voice] = String(voice).split("_", 2);
  return rawName
    .split(/[-\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function setSystemStatus(runtimeReady, websocketReady) {
  const title = `Runtime: ${runtimeReady ? "ready" : "unavailable"}. WebSocket: ${
    websocketReady ? "ready" : "unavailable"
  }.`;
  systemStatus.title = title;
  systemStatus.setAttribute("aria-label", title);
  systemStatus.className =
    runtimeReady && websocketReady
      ? "status-widget status-widget-ok"
      : "status-widget status-widget-warn";
}

function getSelectLabel(selectEl) {
  const selected = selectEl.options[selectEl.selectedIndex];
  return selected ? selected.textContent : "";
}

function buildCustomSelectMenu(selectEl, menuEl) {
  menuEl.innerHTML = "";
  Array.from(selectEl.children).forEach((node) => {
    if (node.tagName === "OPTGROUP") {
      const group = document.createElement("div");
      group.className = "custom-select-group";

      const groupLabel = document.createElement("div");
      groupLabel.className = "custom-select-group-label";
      groupLabel.textContent = node.label;
      group.appendChild(groupLabel);

      Array.from(node.children).forEach((optionEl) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "custom-select-option";
        item.textContent = optionEl.textContent;
        item.dataset.value = optionEl.value;
        item.disabled = optionEl.disabled;
        if (optionEl.selected) {
          item.classList.add("is-selected");
        }
        group.appendChild(item);
      });
      menuEl.appendChild(group);
      return;
    }

    if (node.tagName !== "OPTION") {
      return;
    }
    const item = document.createElement("button");
    item.type = "button";
    item.className = "custom-select-option";
    item.textContent = node.textContent;
    item.dataset.value = node.value;
    item.disabled = node.disabled;
    if (node.selected) {
      item.classList.add("is-selected");
    }
    menuEl.appendChild(item);
  });
}

function closeCustomSelect(exceptSelect = null) {
  customSelectRegistry.forEach((instance, selectEl) => {
    if (exceptSelect && selectEl === exceptSelect) {
      return;
    }
    instance.root.classList.remove("is-open");
    instance.trigger.setAttribute("aria-expanded", "false");
  });
}

function refreshCustomSelect(selectEl) {
  const instance = customSelectRegistry.get(selectEl);
  if (!instance) {
    return;
  }
  instance.trigger.textContent = getSelectLabel(selectEl);
  buildCustomSelectMenu(selectEl, instance.menu);
  const isDisabled = selectEl.disabled;
  instance.trigger.disabled = isDisabled;
  instance.root.classList.toggle("is-disabled", isDisabled);
}

function initializeCustomSelect(selectEl) {
  selectEl.classList.add("native-select");

  const root = document.createElement("div");
  root.className = "custom-select";
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "custom-select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");

  const menu = document.createElement("div");
  menu.className = "custom-select-menu";
  menu.setAttribute("role", "listbox");

  root.appendChild(trigger);
  root.appendChild(menu);
  selectEl.insertAdjacentElement("afterend", root);

  customSelectRegistry.set(selectEl, { root, trigger, menu });
  refreshCustomSelect(selectEl);

  trigger.addEventListener("click", () => {
    if (selectEl.disabled) {
      return;
    }
    const opening = !root.classList.contains("is-open");
    closeCustomSelect(selectEl);
    root.classList.toggle("is-open", opening);
    trigger.setAttribute("aria-expanded", opening ? "true" : "false");
  });

  menu.addEventListener("click", (event) => {
    const target = event.target.closest(".custom-select-option");
    if (!target || target.disabled) {
      return;
    }
    if (selectEl.value !== target.dataset.value) {
      selectEl.value = target.dataset.value;
      selectEl.dispatchEvent(new Event("change", { bubbles: true }));
    }
    refreshCustomSelect(selectEl);
    closeCustomSelect();
  });
}

function initializeCustomSelects() {
  [
    voiceInput,
    langInput,
    transportInput,
    formatInput,
    formatQualityInput,
  ].forEach((selectEl) => initializeCustomSelect(selectEl));
}

function applyTheme(theme) {
  currentTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = currentTheme;
  const themeIcon = themeToggle.querySelector(".icon-theme");
  if (themeIcon) {
    themeIcon.className =
      currentTheme === "dark"
        ? "icon icon-theme icon-theme-dark"
        : "icon icon-theme icon-theme-light";
  }
  const label =
    currentTheme === "light" ? "Switch to dark mode" : "Switch to light mode";
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
}

function toggleTheme() {
  const nextTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  window.localStorage.setItem("kokoro_theme", nextTheme);
}

function initDisplayMode() {
  const savedTheme = window.localStorage.getItem("kokoro_theme");
  if (savedTheme) {
    applyTheme(savedTheme);
  } else {
    const prefersLight = window.matchMedia(
      "(prefers-color-scheme: light)",
    ).matches;
    applyTheme(prefersLight ? "light" : "dark");
  }
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  errorText.hidden = !isError;
  errorText.textContent = isError ? message : "";
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  exportButton.disabled = isBusy || !lastExportRequest;
  if (submitButtonLabel) {
    submitButtonLabel.textContent = isBusy ? "Generating..." : "Play";
  }
}

function buildSynthesisPayload() {
  const isOpus = formatInput.value === "opus";
  const qualityValue = formatQualityInput.value;
  return {
    text: textInput.value,
    voice: voiceInput.value,
    speed: Number(speedInput.value),
    pitch: pitchShiftingAvailable ? Number(pitchInput.value) : 0,
    lang: langInput.value,
    format: formatInput.value,
    opus_bitrate: isOpus ? qualityValue : selectedOpusBitrate,
    wav_sample_rate: isOpus ? selectedWavSampleRate : qualityValue,
    target_chunk_chars: Number(chunkTargetInput.value),
  };
}

function getPitchSemitones() {
  return Number(pitchInput.value);
}

function formatSignedSemitones(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} st`;
}

function formatPitchValue() {
  if (!pitchShiftingAvailable) {
    return "Unavailable";
  }
  return formatSignedSemitones(getPitchSemitones());
}

function updatePitchControlAvailability() {
  pitchInput.disabled = !pitchShiftingAvailable;
  pitchInput.min = String(-maxPitchSemitones);
  pitchInput.max = String(maxPitchSemitones);
  if (!pitchShiftingAvailable) {
    pitchInput.value = "0";
  } else {
    const clampedValue = Math.max(
      -maxPitchSemitones,
      Math.min(maxPitchSemitones, Number(pitchInput.value)),
    );
    pitchInput.value = String(clampedValue);
  }
  pitchValue.textContent = formatPitchValue();
}

function buildExportFilename(format) {
  return `kokoro-output.${format === "opus" ? "opus" : "wav"}`;
}

function populateVoices(voices) {
  const previousValue = voiceInput.value;
  voiceInput.innerHTML = "";

  const groupLabels = {
    af: "American Female",
    am: "American Male",
    bf: "British Female",
    bm: "British Male",
    jf: "Japanese Female",
    jm: "Japanese Male",
    zf: "Mandarin Female",
    zm: "Mandarin Male",
    other: "Other",
  };

  const grouped = new Map();
  voices.forEach((voice) => {
    const prefix = String(voice).split("_", 1)[0] || "other";
    const groupKey = Object.hasOwn(groupLabels, prefix) ? prefix : "other";
    if (!grouped.has(groupKey)) {
      grouped.set(groupKey, []);
    }
    grouped.get(groupKey).push(voice);
  });

  const orderedGroups = [
    "af",
    "am",
    "bf",
    "bm",
    "jf",
    "jm",
    "zf",
    "zm",
    "other",
  ];
  orderedGroups.forEach((groupKey) => {
    const groupVoices = grouped.get(groupKey);
    if (!groupVoices?.length) {
      return;
    }

    const optgroup = document.createElement("optgroup");
    optgroup.label = groupLabels[groupKey];
    groupVoices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice;
      option.textContent = formatVoiceLabel(voice);
      optgroup.appendChild(option);
    });
    voiceInput.appendChild(optgroup);
  });

  if (!voiceInput.children.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No voices found";
    option.disabled = true;
    option.selected = true;
    voiceInput.appendChild(option);
  }

  voiceInput.value = voices.includes(previousValue)
    ? previousValue
    : (voices[0] ?? "");
  refreshCustomSelect(voiceInput);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "--";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "--";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatWavRateLabel(rate) {
  if (!Number.isFinite(rate) || rate <= 0) {
    return "";
  }
  return ` @ ${Math.round(rate).toLocaleString()} Hz`;
}

function qualityOptionText(value, isOpus) {
  if (isOpus) {
    return value;
  }
  if (value === "native") {
    return "Native";
  }
  return `${Number(value).toLocaleString()} Hz`;
}

function rebuildFormatQualityOptions() {
  const isOpus = formatInput.value === "opus";
  const options = isOpus ? availableOpusBitrates : availableWavSampleRates;
  const preferredValue = isOpus ? selectedOpusBitrate : selectedWavSampleRate;

  formatQualityLabel.textContent = isOpus ? "Opus Bitrate" : "WAV Sample Rate";
  formatQualityInput.innerHTML = "";
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = qualityOptionText(value, isOpus);
    formatQualityInput.appendChild(option);
  });

  formatQualityInput.value = options.includes(preferredValue)
    ? preferredValue
    : options[0];
  if (isOpus) {
    selectedOpusBitrate = formatQualityInput.value;
  } else {
    selectedWavSampleRate = formatQualityInput.value;
  }
  refreshCustomSelect(formatQualityInput);
}

function decodeBase64ToBlob(base64Text, mimeType) {
  const binary = atob(base64Text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mimeType });
}

function getInterChunkPauseMs(speed, pauseValue) {
  if (pauseValue > 0) {
    return pauseValue;
  }
  const pause = Math.round(260 / speed);
  return Math.max(140, Math.min(420, pause));
}

function resetPlaybackState() {
  if (chunkState?.queueWaiter) {
    chunkState.streamDone = true;
    chunkState.queueWaiter();
    chunkState.queueWaiter = null;
  }
  playbackToken += 1;
  chunkState = null;
  if (streamAbortController) {
    streamAbortController.abort();
    streamAbortController = null;
  }
  if (activeSocket) {
    activeSocket.close(1000, "reset");
    activeSocket = null;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  player.pause();
  player.currentTime = 0;
  player.removeAttribute("src");
  player.load();
}

function handlePageExit() {
  resetPlaybackState();
}

function updateTextStats() {
  const text = textInput.value.trim();
  charCount.textContent = String(text.length);
  wordCount.textContent = text ? String(text.split(/\s+/).length) : "0";
}

function updatePauseLabel() {
  const pauseValue = Number(pauseMsInput.value);
  pauseMode.textContent =
    pauseValue > 0 ? `${pauseValue} ms fixed` : "0 = auto";
}

function updateFormatControlState() {
  rebuildFormatQualityOptions();
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
      if (token !== playbackToken) {
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

async function waitForQueue(state, token) {
  if (
    token !== playbackToken ||
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

async function readStreamIntoQueue(state, token) {
  streamAbortController = new AbortController();
  const response = await fetch("/api/speak-stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildSynthesisPayload()),
    signal: streamAbortController.signal,
  });

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

  while (token === playbackToken) {
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
        state.queue.push({
          index: message.chunk_index,
          totalChunks: message.total_chunks,
          blob: decodeBase64ToBlob(message.audio_base64, message.mime_type),
          bytes: message.bytes,
          durationSec: message.duration_sec,
          synthMs: message.synth_ms,
          format: message.format,
          opus_bitrate: message.opus_bitrate,
          sample_rate: message.sample_rate,
        });
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

function readWebSocketIntoQueue(state, token) {
  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/speak-stream`;
    const socket = new WebSocket(wsUrl);
    activeSocket = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify(buildSynthesisPayload()));
    };

    socket.onmessage = (event) => {
      if (token !== playbackToken || !chunkState) {
        return;
      }
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
        state.queue.push({
          index: message.chunk_index,
          totalChunks: message.total_chunks,
          blob: decodeBase64ToBlob(message.audio_base64, message.mime_type),
          bytes: message.bytes,
          durationSec: message.duration_sec,
          synthMs: message.synth_ms,
          format: message.format,
          opus_bitrate: message.opus_bitrate,
          sample_rate: message.sample_rate,
        });
        notifyQueue(state);
      } else if (message.type === "error") {
        state.streamError = message.detail || "WebSocket streaming failed.";
        notifyQueue(state);
      } else if (message.type === "done") {
        state.streamDone = true;
        notifyQueue(state);
      }
    };

    socket.onerror = () => {
      reject(new Error("WebSocket transport failed."));
    };

    socket.onclose = () => {
      if (token === playbackToken && chunkState && !state.streamDone) {
        state.streamDone = true;
        notifyQueue(state);
      }
      if (activeSocket === socket) {
        activeSocket = null;
      }
      resolve();
    };
  });
}

async function playQueueFromStream(state, token) {
  while (token === playbackToken) {
    if (!state.queue.length) {
      await waitForQueue(state, token);
      if (token !== playbackToken) {
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

    if (currentObjectUrl) {
      URL.revokeObjectURL(currentObjectUrl);
    }
    currentObjectUrl = URL.createObjectURL(chunk.blob);
    player.src = currentObjectUrl;

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
      if (token !== playbackToken) {
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

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    populateVoices(data.voices || []);
    if (Array.isArray(data.opus_bitrates) && data.opus_bitrates.length > 0) {
      availableOpusBitrates = data.opus_bitrates;
      if (!availableOpusBitrates.includes(selectedOpusBitrate)) {
        selectedOpusBitrate = availableOpusBitrates[0];
      }
    }
    if (
      Array.isArray(data.wav_sample_rates) &&
      data.wav_sample_rates.length > 0
    ) {
      availableWavSampleRates = data.wav_sample_rates;
      if (!availableWavSampleRates.includes(selectedWavSampleRate)) {
        selectedWavSampleRate = availableWavSampleRates[0];
      }
    }
    const websocketEnabled = Boolean(data.websocket_streaming);
    const wsOption = transportInput.querySelector('option[value="ws"]');
    if (wsOption) {
      wsOption.disabled = !websocketEnabled;
    }
    if (transportInput.value === "ws" && !websocketEnabled) {
      transportInput.value = "ndjson";
    }
    pitchShiftingAvailable = data.pitch_shifting !== false;
    if (
      Number.isFinite(data.max_pitch_semitones) &&
      data.max_pitch_semitones > 0
    ) {
      maxPitchSemitones = data.max_pitch_semitones;
    }
    updatePitchControlAvailability();
    refreshCustomSelect(transportInput);
    streamMode.textContent =
      transportInput.value === "ws" ? "WebSocket" : "NDJSON";
    updateFormatControlState();
    if (data.ok) {
      setSystemStatus(true, websocketEnabled);
      setStatus("Ready for synthesis");
      return;
    }

    setSystemStatus(false, websocketEnabled);
    setStatus(`Missing runtime files: ${data.missing.join(", ")}`, true);
  } catch (_error) {
    setSystemStatus(false, false);
    setStatus("Unable to reach the backend.", true);
  }
}

async function synthesize(event) {
  event.preventDefault();
  resetPlaybackState();
  lastExportRequest = null;
  exportButton.disabled = true;
  setBusy(true);
  setStatus("Preparing stream...");
  genTime.textContent = "--";
  fileSize.textContent = "--";
  audioDuration.textContent = "--";
  chunkProgress.textContent = "0 / 0";
  chunkBar.max = 1;
  chunkBar.value = 0;
  streamMode.textContent =
    transportInput.value === "ws" ? "WebSocket" : "NDJSON";

  if (!textInput.value.trim()) {
    setStatus("Enter text before generating.", true);
    lastExportRequest = null;
    exportButton.disabled = true;
    setBusy(false);
    return;
  }

  try {
    const token = playbackToken;
    lastExportRequest = buildSynthesisPayload();
    const state = {
      queue: [],
      queueWaiter: null,
      totalChunks: 0,
      playedChunks: 0,
      totalElapsedMs: 0,
      totalBytes: 0,
      totalDurationSec: 0,
      streamDone: false,
      streamError: null,
    };
    chunkState = state;

    const readPromise =
      transportInput.value === "ws"
        ? readWebSocketIntoQueue(state, token)
        : readStreamIntoQueue(state, token);
    await playQueueFromStream(state, token);
    await readPromise;

    if (state.streamError) {
      throw new Error(state.streamError);
    }
    if (token === playbackToken) {
      setStatus(`Playback finished across ${state.totalChunks} chunks.`);
    }
  } catch (error) {
    lastExportRequest = null;
    if (error.name !== "AbortError") {
      setStatus(error.message || "Synthesis failed.", true);
    }
    genTime.textContent = "--";
    fileSize.textContent = "--";
  } finally {
    streamAbortController = null;
    setBusy(false);
  }
}

async function exportAudio() {
  if (!lastExportRequest) {
    return;
  }

  exportButton.disabled = true;
  exportButtonLabel.textContent = "Exporting...";

  try {
    const response = await fetch("/api/speak", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(lastExportRequest),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Export failed.");
    }

    const blob = await response.blob();
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = buildExportFilename(lastExportRequest.format);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(downloadUrl);
    setStatus(`Exported ${lastExportRequest.format.toUpperCase()} download.`);
  } catch (error) {
    setStatus(error.message || "Export failed.", true);
  } finally {
    exportButtonLabel.textContent = "Export Audio";
    exportButton.disabled = !lastExportRequest;
  }
}

speedInput.addEventListener("input", () => {
  speedValue.textContent = `${Number(speedInput.value).toFixed(2)}x`;
});

pitchInput.addEventListener("input", () => {
  pitchValue.textContent = formatPitchValue();
});

pauseMsInput.addEventListener("input", updatePauseLabel);
formatInput.addEventListener("change", updateFormatControlState);
formatQualityInput.addEventListener("change", () => {
  if (formatInput.value === "opus") {
    selectedOpusBitrate = formatQualityInput.value;
  } else {
    selectedWavSampleRate = formatQualityInput.value;
  }
});
transportInput.addEventListener("change", () => {
  streamMode.textContent =
    transportInput.value === "ws" ? "WebSocket" : "NDJSON";
});
textInput.addEventListener("input", updateTextStats);
themeToggle.addEventListener("click", toggleTheme);
exportButton.addEventListener("click", exportAudio);

stopButton.addEventListener("click", () => {
  resetPlaybackState();
  chunkProgress.textContent = "0 / 0";
  chunkBar.max = 1;
  chunkBar.value = 0;
  setStatus("Playback stopped");
});

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", () => {
    textInput.value = button.dataset.text || "";
    updateTextStats();
    textInput.focus();
  });
});

form.addEventListener("submit", synthesize);
window.addEventListener("DOMContentLoaded", () => {
  initializeCustomSelects();
  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    if (!event.target.closest(".custom-select")) {
      closeCustomSelect();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeCustomSelect();
    }
  });
  initDisplayMode();
  updateTextStats();
  updatePauseLabel();
  updatePitchControlAvailability();
  updateFormatControlState();
  streamMode.textContent =
    transportInput.value === "ws" ? "WebSocket" : "NDJSON";
  loadHealth();
});
window.addEventListener("pagehide", handlePageExit);
window.addEventListener("beforeunload", handlePageExit);
