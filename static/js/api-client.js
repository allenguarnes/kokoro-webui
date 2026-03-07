import {
  audioDuration,
  chunkBar,
  chunkProgress,
  exportButton,
  exportButtonLabel,
  fileSize,
  genTime,
  transportInput,
} from "./dom.js";
import { appState, createChunkState } from "./state.js";
import {
  buildSynthesisPayload,
  buildExportFilename,
  populateVoices,
  refreshCustomSelect,
  setBusy,
  setStatus,
  setSystemStatus,
  syncTransportModeText,
  updateFormatControlState,
  updatePitchControlAvailability,
} from "./ui.js";
import {
  playQueueFromStream,
  readStreamIntoQueue,
  readWebSocketIntoQueue,
  resetPlaybackState,
} from "./playback.js";

export async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    populateVoices(data.voices || []);
    if (Array.isArray(data.opus_bitrates) && data.opus_bitrates.length > 0) {
      appState.availableOpusBitrates = data.opus_bitrates;
      if (
        !appState.availableOpusBitrates.includes(appState.selectedOpusBitrate)
      ) {
        appState.selectedOpusBitrate = appState.availableOpusBitrates[0];
      }
    }
    if (
      Array.isArray(data.wav_sample_rates) &&
      data.wav_sample_rates.length > 0
    ) {
      appState.availableWavSampleRates = data.wav_sample_rates;
      if (
        !appState.availableWavSampleRates.includes(
          appState.selectedWavSampleRate,
        )
      ) {
        appState.selectedWavSampleRate = appState.availableWavSampleRates[0];
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
    appState.pitchShiftingAvailable = data.pitch_shifting !== false;
    if (
      Number.isFinite(data.max_pitch_semitones) &&
      data.max_pitch_semitones > 0
    ) {
      appState.maxPitchSemitones = data.max_pitch_semitones;
    }
    updatePitchControlAvailability();
    refreshCustomSelect(transportInput);
    syncTransportModeText();
    updateFormatControlState();
    const activeProvider =
      typeof data.active_provider === "string" ? data.active_provider : null;
    const providerFallback = data.provider_fallback === true;
    const providerError =
      typeof data.provider_error === "string" ? data.provider_error : null;
    const runtimeError =
      typeof data.runtime_error === "string" ? data.runtime_error : null;
    if (data.ok) {
      setSystemStatus(
        true,
        websocketEnabled,
        activeProvider,
        providerFallback,
        providerError,
        runtimeError,
      );
      if (providerFallback && activeProvider) {
        setStatus("Ready for synthesis with CPU fallback.", true);
        return;
      }
      setStatus("Ready for synthesis");
      return;
    }

    setSystemStatus(
      false,
      websocketEnabled,
      activeProvider,
      providerFallback,
      providerError,
      runtimeError,
    );
    if (runtimeError) {
      setStatus(`Runtime unavailable: ${runtimeError}`, true);
      return;
    }
    setStatus(`Missing runtime files: ${data.missing.join(", ")}`, true);
  } catch (_error) {
    setSystemStatus(false, false);
    setStatus("Unable to reach the backend.", true);
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
    if (error.name !== "AbortError") {
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
    const response = await fetch("/api/speak", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(appState.lastExportRequest),
    });

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
    setStatus(error.message || "Export failed.", true);
  } finally {
    exportButtonLabel.textContent = "Export Audio";
    exportButton.disabled = !appState.lastExportRequest;
  }
}
