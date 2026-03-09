import { initializeApiExamples } from "./js/api-docs.js";
import {
  clearApiKey,
  exportAudio,
  initializeAccess,
  submitApiKey,
  synthesize,
} from "./js/api-client.js";
import {
  authApiKeyInput,
  authClearButton,
  authForm,
  chips,
  exportButton,
  formatInput,
  formatQualityInput,
  form,
  pauseMsInput,
  pitchInput,
  speedInput,
  stopButton,
  textInput,
  themeToggle,
  transportInput,
} from "./js/dom.js";
import { appState } from "./js/state.js";
import { handlePageExit, resetPlaybackState } from "./js/playback.js";
import {
  closeCustomSelect,
  initDisplayMode,
  initializeCustomSelects,
  refreshCustomSelect,
  setStatus,
  syncTransportModeText,
  toggleTheme,
  updateFormatControlState,
  updatePauseLabel,
  updatePitchControlAvailability,
  updateTextStats,
} from "./js/ui.js";

speedInput.addEventListener("input", () => {
  document.getElementById("speedValue").textContent =
    `${Number(speedInput.value).toFixed(2)}x`;
});

pitchInput.addEventListener("input", () => {
  document.getElementById("pitchValue").textContent = `${
    Number(pitchInput.value) >= 0 ? "+" : ""
  }${Number(pitchInput.value).toFixed(1)} st`;
});

pauseMsInput.addEventListener("input", updatePauseLabel);
formatInput.addEventListener("change", updateFormatControlState);
formatQualityInput.addEventListener("change", () => {
  if (formatInput.value === "opus") {
    appState.selectedOpusBitrate = formatQualityInput.value;
  } else {
    appState.selectedWavSampleRate = formatQualityInput.value;
  }
});
transportInput.addEventListener("change", syncTransportModeText);
textInput.addEventListener("input", updateTextStats);
themeToggle.addEventListener("click", toggleTheme);
exportButton.addEventListener("click", exportAudio);
authForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitApiKey(authApiKeyInput.value);
});
authClearButton?.addEventListener("click", () => {
  if (authApiKeyInput) {
    authApiKeyInput.value = "";
  }
  clearApiKey();
});

stopButton.addEventListener("click", () => {
  resetPlaybackState();
  document.getElementById("chunkProgress").textContent = "0 / 0";
  document.getElementById("chunkBar").max = 1;
  document.getElementById("chunkBar").value = 0;
  setStatus("Playback stopped");
});

chips.forEach((button) => {
  button.addEventListener("click", () => {
    textInput.value = button.dataset.text || "";
    updateTextStats();
    textInput.focus();
  });
});

form.addEventListener("submit", synthesize);
window.addEventListener("DOMContentLoaded", () => {
  initializeCustomSelects();
  initializeApiExamples();
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
  syncTransportModeText();
  refreshCustomSelect(transportInput);
  initializeAccess();
});

window.addEventListener("pagehide", handlePageExit);
window.addEventListener("beforeunload", handlePageExit);
