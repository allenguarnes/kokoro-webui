import {
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

const HEALTH_POLL_INTERVAL_MS = 2000;
let healthPollTimerId = null;
let healthPollInFlight = false;

function authHeaders() {
  if (!appState.apiKey) {
    return {};
  }
  return {
    Authorization: `Bearer ${appState.apiKey}`,
  };
}

function apiFetch(input, init = {}) {
  const headers = new Headers(init.headers || {});
  Object.entries(authHeaders()).forEach(([key, value]) => {
    headers.set(key, value);
  });
  return fetch(input, { ...init, headers });
}

function handleAuthFailure(message = "Authentication failed.") {
  stopHealthPolling();
  appState.apiKey = null;
  setUiLocked(true);
  setAuthPanelState(true, message, true);
  setStatus(message, true);
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
    await loadHealth();
    setUiLocked(false);
    setAuthPanelState(false);
    setStatus("Authenticated. Ready for synthesis");
    startHealthPolling();
    return true;
  } catch (error) {
    if (error instanceof AuthRequiredError) {
      handleAuthFailure("Authentication failed.");
      return false;
    }
    appState.apiKey = null;
    setAuthPanelState(
      true,
      error.message || "Unable to reach the backend.",
      true,
    );
    return false;
  }
}

export function clearApiKey() {
  stopHealthPolling();
  appState.apiKey = null;
  setUiLocked(true);
  setAuthPanelState(true, "Enter the configured API key to use this server.");
  setStatus("API key cleared.");
}

async function pollHealth() {
  if (healthPollInFlight) {
    return;
  }
  if (appState.authRequired && !appState.apiKey) {
    return;
  }
  healthPollInFlight = true;
  try {
    await loadHealth({ quiet: true, includeCapabilities: false });
  } catch (error) {
    if (error instanceof AuthRequiredError) {
      handleAuthFailure("Authentication failed.");
    }
  } finally {
    healthPollInFlight = false;
  }
}

function startHealthPolling() {
  stopHealthPolling();
  healthPollTimerId = window.setInterval(() => {
    void pollHealth();
  }, HEALTH_POLL_INTERVAL_MS);
}

function stopHealthPolling() {
  if (healthPollTimerId === null) {
    return;
  }
  window.clearInterval(healthPollTimerId);
  healthPollTimerId = null;
  healthPollInFlight = false;
}

export async function loadHealth(options = {}) {
  const quiet = options.quiet === true;
  const includeCapabilities = options.includeCapabilities !== false;
  const [healthResponse, capabilitiesResponse] = await Promise.all([
    apiFetch("/api/health"),
    includeCapabilities ? apiFetch("/api/capabilities") : Promise.resolve(null),
  ]);
  if (healthResponse.status === 401 || capabilitiesResponse?.status === 401) {
    throw new AuthRequiredError();
  }
  try {
    const health = await healthResponse.json();
    const capabilities = capabilitiesResponse
      ? await capabilitiesResponse.json()
      : null;
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
    const providerError =
      typeof health.provider_error === "string" ? health.provider_error : null;
    const runtimeError =
      typeof health.runtime_error === "string" ? health.runtime_error : null;
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
      setStatus(`Runtime unavailable: ${runtimeError}`, true);
      return;
    }
    if (!quiet) {
      setStatus(`Missing runtime files: ${health.missing.join(", ")}`, true);
    }
  } catch (error) {
    if (error instanceof AuthRequiredError) {
      throw error;
    }
    updateQueueMonitorLayout(null);
    gpuVram.textContent = "--";
    setSystemStatus(false, false, null, false, null, null, null);
    if (!quiet) {
      setStatus("Unable to reach the backend.", true);
    }
  }
}

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
    await playQueueFromStream(state, token);
    await readPromise;

    if (state.streamError) {
      throw new Error(state.streamError);
    }
    if (token === appState.playbackToken) {
      setStatus(`Playback finished across ${state.totalChunks} chunks.`);
    }
  } catch (error) {
    appState.lastExportRequest = null;
    if (error.name === "AuthRequiredError") {
      handleAuthFailure(error.message || "Authentication failed.");
    } else if (error.name !== "AbortError") {
      setStatus(error.message || "Synthesis failed.", true);
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

    if (response.status === 401) {
      throw new AuthRequiredError();
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
    if (error instanceof AuthRequiredError) {
      handleAuthFailure(error.message || "Authentication failed.");
    } else {
      setStatus(error.message || "Export failed.", true);
    }
  } finally {
    exportButtonLabel.textContent = "Export Audio";
    exportButton.disabled = !appState.lastExportRequest;
  }
}
