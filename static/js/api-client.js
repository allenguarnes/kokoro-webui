import {
  authApiKeyInput,
  audioDuration,
  chunkBar,
  chunkProgress,
  exportButton,
  exportButtonLabel,
  fileSize,
  genTime,
  gpuVram,
  transportInput,
} from "./dom.js";
import { appState, createChunkState } from "./state.js";
import {
  buildSynthesisPayload,
  buildExportFilename,
  populateVoices,
  refreshCustomSelect,
  setBusy,
  setAuthPanelState,
  setStatus,
  setSystemStatus,
  setUiLocked,
  syncTransportModeText,
  updateFormatControlState,
  updateQueueMonitorLayout,
  updatePitchControlAvailability,
} from "./ui.js";
import {
  playQueueFromStream,
  readStreamIntoQueue,
  readWebSocketIntoQueue,
  resetPlaybackState,
} from "./playback.js";

class AuthRequiredError extends Error {
  constructor(message = "Authentication failed.") {
    super(message);
    this.name = "AuthRequiredError";
  }
}

class AuthRateLimitError extends Error {
  constructor(message = "Too many authentication failures. Try again later.") {
    super(message);
    this.name = "AuthRateLimitError";
  }
}

function isAuthError(error) {
  return (
    error instanceof AuthRequiredError ||
    error instanceof AuthRateLimitError ||
    error?.name === "AuthRequiredError" ||
    error?.name === "AuthRateLimitError"
  );
}

const HEALTH_POLL_INTERVAL_MS = 4000;
const HIDDEN_HEALTH_POLL_INTERVAL_MS = 20000;
const HEALTH_POLL_MAX_INTERVAL_MS = 60000;
const HEALTH_REQUEST_TIMEOUT_MS = 8000;
let healthPollTimerId = null;
let healthPollInFlight = false;
let healthPollFailureCount = 0;
let healthPollAbortController = null;
let healthPollCycle = 0;

function getHealthPollIntervalMs() {
  const baseInterval = document.hidden
    ? HIDDEN_HEALTH_POLL_INTERVAL_MS
    : HEALTH_POLL_INTERVAL_MS;
  const backoffFactor = 2 ** Math.min(healthPollFailureCount, 4);
  const backoffInterval = Math.min(
    baseInterval * backoffFactor,
    HEALTH_POLL_MAX_INTERVAL_MS,
  );
  const jitterWindow = Math.max(250, Math.round(backoffInterval * 0.1));
  const jitterOffset = Math.round(Math.random() * jitterWindow);
  return backoffInterval + jitterOffset;
}

function authHeaders() {
  if (!appState.apiKey) {
    return {};
  }
  return {
    Authorization: `Bearer ${appState.apiKey}`,
  };
}

function apiFetch(input, init = {}) {
  const { timeoutMs = 0, signal, ...restInit } = init;
  const headers = new Headers(init.headers || {});
  Object.entries(authHeaders()).forEach(([key, value]) => {
    headers.set(key, value);
  });

  if (!timeoutMs) {
    return fetch(input, { ...restInit, signal, headers });
  }

  const controller = new AbortController();
  let timeoutId = null;
  const onAbort = () => {
    controller.abort(signal?.reason);
  };
  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason);
    } else {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  }
  timeoutId = window.setTimeout(() => {
    controller.abort(new Error("Health request timed out."));
  }, timeoutMs);

  return fetch(input, { ...restInit, signal: controller.signal, headers }).finally(() => {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener?.("abort", onAbort);
  });
}

function sanitizeAuthMessage(message) {
  if (typeof message !== "string") {
    return "Authentication failed.";
  }
  const normalized = message.trim().toLowerCase();
  if (normalized.includes("too many authentication failures")) {
    return "Too many authentication failures. Try again later.";
  }
  return "Authentication failed.";
}

function sanitizeOperationalMessage(error, fallback) {
  const message = typeof error?.message === "string" ? error.message.trim() : "";
  if (!message) {
    return fallback;
  }
  if (
    /network|fetch|timeout|timed out|load failed|backend|connection|abort/i.test(
      message,
    )
  ) {
    return "Unable to reach the backend.";
  }
  return fallback;
}

function handleAuthFailure(message = "Authentication failed.") {
  appState.apiKey = null;
  stopHealthPolling();
  resetPlaybackState();
  appState.lastExportRequest = null;
  setUiLocked(true);
  setAuthPanelState(true, message, true);
  setStatus(message, true);
  if (authApiKeyInput) {
    authApiKeyInput.value = "";
  }
}

function summarizeDiagnostic(detail, fallback) {
  if (typeof detail !== "string" || !detail.trim()) {
    return null;
  }
  return fallback;
}

async function buildAuthError(response) {
  const payload = await response.json().catch(() => ({}));
  const message = sanitizeAuthMessage(
    payload.detail ||
      payload?.error?.message ||
      (response.status === 429
        ? "Too many authentication failures. Try again later."
        : "Authentication failed."),
  );
  if (response.status === 429) {
    return new AuthRateLimitError(message);
  }
  return new AuthRequiredError(message);
}

function failQueuedPlayback(state, message) {
  state.streamError = message;
  if (state.queueWaiter) {
    state.queueWaiter();
    state.queueWaiter = null;
  }
}

async function loadPublicConfig() {
  const response = await fetch("/api/public-config");
  if (!response.ok) {
    throw new Error("Unable to load public configuration.");
  }
  return response.json();
}

export async function initializeAccess() {
  const publicConfig = await loadPublicConfig();
  appState.authRequired = publicConfig.auth_required === true;
  if (!appState.authRequired) {
    setAuthPanelState(false);
    setUiLocked(false);
    await loadHealth();
    startHealthPolling();
    return;
  }

  setUiLocked(true);
  setAuthPanelState(true, "Enter the configured API key to use this server.");
  setStatus("API key required.", true);
}

export async function submitApiKey(apiKey) {
  const trimmedKey = String(apiKey || "").trim();
  if (!trimmedKey) {
    setAuthPanelState(true, "Enter an API key.", true);
    return false;
  }

  appState.apiKey = trimmedKey;
  setAuthPanelState(true, "Validating API key...", false, false);
  try {
    await loadHealth({ strictErrors: true });
    setUiLocked(false);
    setAuthPanelState(false);
    setStatus("Authenticated. Ready for synthesis");
    startHealthPolling();
    return true;
  } catch (error) {
    if (isAuthError(error)) {
      handleAuthFailure(error.message || "Authentication failed.");
      return false;
    }
    appState.apiKey = null;
    const userMessage = sanitizeOperationalMessage(
      error,
      "Unable to reach the backend.",
    );
    setStatus(userMessage, true);
    setAuthPanelState(
      true,
      userMessage,
      true,
    );
    return false;
  }
}

export function clearApiKey() {
  appState.apiKey = null;
  stopHealthPolling();
  resetPlaybackState();
  appState.lastExportRequest = null;
  setUiLocked(true);
  setAuthPanelState(true, "Enter the configured API key to use this server.");
  setStatus("API key cleared.");
  if (authApiKeyInput) {
    authApiKeyInput.value = "";
  }
}

async function pollHealth() {
  if (healthPollInFlight) {
    return;
  }
  if (appState.authRequired && !appState.apiKey) {
    return;
  }
  const pollCycle = ++healthPollCycle;
  healthPollInFlight = true;
  const abortController = new AbortController();
  healthPollAbortController = abortController;
  try {
    await loadHealth({
      quiet: true,
      includeCapabilities: false,
      signal: abortController.signal,
      timeoutMs: HEALTH_REQUEST_TIMEOUT_MS,
    });
    if (pollCycle === healthPollCycle) {
      healthPollFailureCount = 0;
    }
  } catch (error) {
    if (isAuthError(error)) {
      handleAuthFailure(error.message || "Authentication failed.");
      return;
    }
    if (pollCycle === healthPollCycle) {
      healthPollFailureCount += 1;
    }
  } finally {
    if (healthPollAbortController === abortController) {
      healthPollAbortController = null;
    }
    if (pollCycle !== healthPollCycle) {
      return;
    }
    healthPollInFlight = false;
    if ((!appState.authRequired || appState.apiKey) && healthPollTimerId === null) {
      scheduleHealthPoll();
    }
  }
}

function startHealthPolling() {
  stopHealthPolling();
  healthPollCycle += 1;
  scheduleHealthPoll();
}

function scheduleHealthPoll(delayMs = getHealthPollIntervalMs()) {
  if (healthPollTimerId !== null) {
    window.clearTimeout(healthPollTimerId);
  }
  healthPollTimerId = window.setTimeout(() => {
    healthPollTimerId = null;
    void pollHealth();
  }, delayMs);
}

function stopHealthPolling() {
  if (healthPollTimerId === null) {
    healthPollCycle += 1;
    healthPollInFlight = false;
    healthPollFailureCount = 0;
    healthPollAbortController?.abort();
    healthPollAbortController = null;
    return;
  }
  window.clearTimeout(healthPollTimerId);
  healthPollTimerId = null;
  healthPollCycle += 1;
  healthPollInFlight = false;
  healthPollFailureCount = 0;
  healthPollAbortController?.abort();
  healthPollAbortController = null;
}

export async function loadHealth(options = {}) {
  const quiet = options.quiet === true;
  const includeCapabilities = options.includeCapabilities !== false;
  const strictErrors = options.strictErrors === true;
  const healthResponse = await apiFetch("/api/health", {
    signal: options.signal,
    timeoutMs: options.timeoutMs,
  });
  if (healthResponse.status === 401 || healthResponse.status === 429) {
    throw await buildAuthError(healthResponse);
  }
  try {
    const health = await healthResponse.json();
    let capabilities = null;
    if (includeCapabilities) {
      const capabilitiesResponse = await apiFetch("/api/capabilities");
      if (capabilitiesResponse.status === 401 || capabilitiesResponse.status === 429) {
        throw await buildAuthError(capabilitiesResponse);
      }
      capabilities = await capabilitiesResponse.json();
    }
    if (capabilities) {
      populateVoices(capabilities.voices || []);
      if (
        Array.isArray(capabilities.formats) &&
        capabilities.formats.length > 0
      ) {
        appState.availableFormats = capabilities.formats;
        if (!appState.availableFormats.includes(appState.selectedFormat)) {
          appState.selectedFormat = appState.availableFormats[0];
        }
      }
      if (
        Array.isArray(capabilities.opus_bitrates) &&
        capabilities.opus_bitrates.length > 0
      ) {
        appState.availableOpusBitrates = capabilities.opus_bitrates;
        if (
          !appState.availableOpusBitrates.includes(appState.selectedOpusBitrate)
        ) {
          appState.selectedOpusBitrate = appState.availableOpusBitrates[0];
        }
      }
      if (
        Array.isArray(capabilities.wav_sample_rates) &&
        capabilities.wav_sample_rates.length > 0
      ) {
        appState.availableWavSampleRates = capabilities.wav_sample_rates;
        if (
          !appState.availableWavSampleRates.includes(
            appState.selectedWavSampleRate,
          )
        ) {
          appState.selectedWavSampleRate = appState.availableWavSampleRates[0];
        }
      }
    }
    const websocketEnabled = capabilities
      ? Boolean(capabilities.websocket_streaming)
      : transportInput.querySelector('option[value="ws"]')?.disabled !== true;
    const wsOption = transportInput.querySelector('option[value="ws"]');
    if (wsOption && capabilities) {
      wsOption.disabled = !websocketEnabled;
    }
    if (transportInput.value === "ws" && !websocketEnabled) {
      transportInput.value = "ndjson";
    }
    if (capabilities) {
      appState.pitchShiftingAvailable = capabilities.pitch_shifting !== false;
      if (
        Number.isFinite(capabilities.max_pitch_semitones) &&
        capabilities.max_pitch_semitones > 0
      ) {
        appState.maxPitchSemitones = capabilities.max_pitch_semitones;
      }
    }
    updatePitchControlAvailability();
    refreshCustomSelect(transportInput);
    syncTransportModeText();
    updateFormatControlState();
    const activeProvider =
      typeof health.active_provider === "string"
        ? health.active_provider
        : null;
    const providerFallback = health.provider_fallback === true;
    const providerError = summarizeDiagnostic(
      health.provider_error,
      "Provider reported a startup issue. Check server logs.",
    );
    const runtimeError = summarizeDiagnostic(
      health.runtime_error,
      "Runtime unavailable. Check server logs.",
    );
    const runtimeActivityState =
      typeof health?.runtime_activity?.state === "string"
        ? health.runtime_activity.state
        : null;
    updateQueueMonitorLayout(activeProvider);
    const processGroupVramMb = Number(health?.gpu?.process_group_vram_used_mb);
    const processVramMb = Number(health?.gpu?.process_vram_used_mb);
    const displayedVramMb =
      Number.isFinite(processGroupVramMb) && processGroupVramMb >= 0
        ? processGroupVramMb
        : processVramMb;
    gpuVram.textContent =
      Number.isFinite(displayedVramMb) && displayedVramMb >= 0
        ? `${displayedVramMb.toFixed(displayedVramMb >= 1024 ? 0 : 1)} MiB`
        : "--";
    if (health.ok) {
      setSystemStatus(
        true,
        websocketEnabled,
        activeProvider,
        providerFallback,
        providerError,
        runtimeError,
        runtimeActivityState,
      );
      if (!quiet && providerFallback && activeProvider) {
        setStatus("Ready for synthesis with CPU fallback.", true);
        return;
      }
      if (!quiet) {
        setStatus("Ready for synthesis");
      }
      return;
    }

    setSystemStatus(
      false,
      websocketEnabled,
      activeProvider,
      providerFallback,
      providerError,
      runtimeError,
      runtimeActivityState,
    );
    if (runtimeError && !quiet) {
      setStatus(runtimeError, true);
      return;
    }
    if (!quiet) {
      const missingFiles = Array.isArray(health.missing) ? health.missing : [];
      setStatus(
        missingFiles.length
          ? "Runtime files are missing. Check server setup."
          : "Runtime unavailable. Check server setup.",
        true,
      );
    }
  } catch (error) {
    if (isAuthError(error)) {
      throw error;
    }
    updateQueueMonitorLayout(null);
    gpuVram.textContent = "--";
    setSystemStatus(false, false, null, false, null, null, null);
    if (!quiet) {
      setStatus("Unable to reach the backend.", true);
    }
    if (strictErrors) {
      throw error;
    }
  }
}

function handleVisibilityChange() {
  if (appState.authRequired && !appState.apiKey) {
    stopHealthPolling();
    return;
  }
  healthPollFailureCount = 0;
  startHealthPolling();
  if (!document.hidden) {
    stopHealthPolling();
    void pollHealth();
  }
}

document.addEventListener("visibilitychange", handleVisibilityChange);

export async function synthesize(event) {
  event.preventDefault();
  resetPlaybackState();
  appState.lastExportRequest = null;
  exportButton.disabled = true;
  setBusy(true);
  setStatus("Preparing stream...");
  genTime.textContent = "--";
  fileSize.textContent = "--";
  audioDuration.textContent = "--";
  gpuVram.textContent = gpuVram.textContent || "--";
  chunkProgress.textContent = "0 / 0";
  chunkBar.max = 1;
  chunkBar.value = 0;
  syncTransportModeText();

  if (!document.getElementById("text").value.trim()) {
    setStatus("Enter text before generating.", true);
    appState.lastExportRequest = null;
    exportButton.disabled = true;
    setBusy(false);
    return;
  }

  try {
    const token = appState.playbackToken;
    const state = createChunkState();
    appState.lastExportRequest = buildSynthesisPayload();
    appState.chunkState = state;

    const readPromise =
      transportInput.value === "ws"
        ? readWebSocketIntoQueue(state, token)
        : readStreamIntoQueue(state, token);
    const guardedReadPromise = readPromise.catch((error) => {
      failQueuedPlayback(state, error.message || "Synthesis failed.");
      throw error;
    });
    await Promise.all([playQueueFromStream(state, token), guardedReadPromise]);

    if (state.streamError) {
      throw new Error(state.streamError);
    }
    if (token === appState.playbackToken) {
      setStatus(`Playback finished across ${state.totalChunks} chunks.`);
    }
  } catch (error) {
    appState.lastExportRequest = null;
    if (isAuthError(error)) {
      handleAuthFailure(error.message || "Authentication failed.");
    } else if (error.name !== "AbortError") {
      setStatus(sanitizeOperationalMessage(error, "Synthesis failed."), true);
    }
    genTime.textContent = "--";
    fileSize.textContent = "--";
  } finally {
    appState.streamAbortController = null;
    setBusy(false);
  }
}

export async function exportAudio() {
  if (!appState.lastExportRequest) {
    return;
  }

  exportButton.disabled = true;
  exportButtonLabel.textContent = "Exporting...";

  try {
    const response = await apiFetch("/api/speak", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(appState.lastExportRequest),
    });

    if (response.status === 401 || response.status === 429) {
      throw await buildAuthError(response);
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Export failed.");
    }

    const blob = await response.blob();
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = buildExportFilename(appState.lastExportRequest.format);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(downloadUrl);
    setStatus(
      `Exported ${appState.lastExportRequest.format.toUpperCase()} download.`,
    );
  } catch (error) {
    if (isAuthError(error)) {
      handleAuthFailure(error.message || "Authentication failed.");
    } else {
      setStatus(sanitizeOperationalMessage(error, "Export failed."), true);
    }
  } finally {
    exportButtonLabel.textContent = "Export Audio";
    exportButton.disabled = !appState.lastExportRequest;
  }
}
