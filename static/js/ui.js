import {
  appShell,
  appWorkspace,
  audioDuration,
  authApiKeyInput,
  authClearButton,
  authMessage,
  authPanel,
  authUnlockButton,
  charStatCard,
  charCount,
  chunkTargetInput,
  errorText,
  exportButton,
  formatInput,
  formatQualityInput,
  formatQualityLabel,
  langInput,
  pauseMode,
  pauseMsInput,
  pitchInput,
  pitchValue,
  speedInput,
  statusText,
  streamMode,
  submitButton,
  submitButtonLabel,
  systemStatus,
  providerBadge,
  textInput,
  textStatCard,
  textStats,
  themeToggle,
  transportInput,
  voiceInput,
  vramStatCard,
  wordStatCard,
  wordCount,
} from "./dom.js";
import { appState } from "./state.js";

export function formatVoiceLabel(voice) {
  const [prefix = "", rawName = voice] = String(voice).split("_", 2);
  const baseLabel = rawName
    .split(/[-\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
  const canonicalPrefixes = new Set([
    "af",
    "am",
    "bf",
    "bm",
    "jf",
    "jm",
    "zf",
    "zm",
  ]);
  if (canonicalPrefixes.has(prefix)) {
    return baseLabel;
  }
  return `${baseLabel} (${prefix || "other"})`;
}

function formatProviderLabel(activeProvider) {
  if (!activeProvider) {
    return "Runtime";
  }

  const labels = {
    CPUExecutionProvider: "CPU",
    CUDAExecutionProvider: "CUDA",
    TensorrtExecutionProvider: "TensorRT",
    DmlExecutionProvider: "DirectML",
    CoreMLExecutionProvider: "CoreML",
  };
  return (
    labels[activeProvider] || activeProvider.replace(/ExecutionProvider$/, "")
  );
}

function isGpuProvider(activeProvider) {
  return (
    activeProvider === "CUDAExecutionProvider" ||
    activeProvider === "TensorrtExecutionProvider"
  );
}

function formatRuntimeActivityLabel(runtimeActivityState) {
  const labels = {
    active: "Active",
    idling: "Idle",
    unloaded: "Unloaded",
  };
  return labels[runtimeActivityState] || null;
}

export function setSystemStatus(
  runtimeReady,
  websocketReady,
  activeProvider = null,
  providerFallback = false,
  providerError = null,
  runtimeError = null,
  runtimeActivityState = null,
) {
  const providerLabel = formatProviderLabel(activeProvider);
  const activityLabel = formatRuntimeActivityLabel(runtimeActivityState);
  const details = [
    `Runtime: ${runtimeReady ? "Ready" : "Unavailable"},`,
    `WebSocket: ${websocketReady ? "Ready" : "Unavailable"},`,
  ];
  if (activeProvider) {
    details.push(`Provider: ${activeProvider}`);
  }
  if (activityLabel) {
    details.push(`Activity: ${activityLabel}.`);
  }
  if (providerFallback) {
    details.push("GPU fallback: active.");
  }
  if (providerError) {
    details.push(`Provider error: ${providerError}.`);
  }
  if (runtimeError) {
    details.push(`Runtime error: ${runtimeError}.`);
  }

  const title = details.join(" ");
  systemStatus.title = title;
  systemStatus.setAttribute("aria-label", title);
  systemStatus.className =
    runtimeReady && websocketReady && !providerFallback
      ? "status-widget status-widget-ok"
      : "status-widget status-widget-warn";

  providerBadge.textContent = activityLabel
    ? `${providerLabel} · ${activityLabel}`
    : providerLabel;
  providerBadge.title = activityLabel
    ? `Active runtime provider: ${providerLabel}. Runtime activity: ${activityLabel}.`
    : `Active runtime provider: ${providerLabel}`;
  providerBadge.setAttribute("aria-label", providerBadge.title);
  providerBadge.className = "runtime-badge";
  if (!activeProvider) {
    providerBadge.classList.add("runtime-badge-idle");
  } else if (providerLabel === "CUDA" || providerLabel === "TensorRT") {
    providerBadge.classList.add("runtime-badge-accel");
  } else {
    providerBadge.classList.add("runtime-badge-default");
  }
  if (providerFallback || runtimeError) {
    providerBadge.classList.add("runtime-badge-warn");
  }
}

export function updateQueueMonitorLayout(activeProvider = null) {
  const gpuMode = isGpuProvider(activeProvider);
  if (charStatCard) {
    charStatCard.hidden = gpuMode;
  }
  if (wordStatCard) {
    wordStatCard.hidden = gpuMode;
  }
  if (textStatCard) {
    textStatCard.hidden = !gpuMode;
  }
  if (vramStatCard) {
    vramStatCard.hidden = !gpuMode;
  }
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

export function closeCustomSelect(exceptSelect = null) {
  appState.customSelectRegistry.forEach((instance, selectEl) => {
    if (exceptSelect && selectEl === exceptSelect) {
      return;
    }
    instance.root.classList.remove("is-open");
    instance.trigger.setAttribute("aria-expanded", "false");
  });
}

export function refreshCustomSelect(selectEl) {
  const instance = appState.customSelectRegistry.get(selectEl);
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

  appState.customSelectRegistry.set(selectEl, { root, trigger, menu });
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

export function initializeCustomSelects() {
  [
    voiceInput,
    langInput,
    transportInput,
    formatInput,
    formatQualityInput,
  ].forEach((selectEl) => initializeCustomSelect(selectEl));
}

export function applyTheme(theme) {
  appState.currentTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = appState.currentTheme;
  const themeIcon = themeToggle.querySelector(".icon-theme");
  if (themeIcon) {
    themeIcon.className =
      appState.currentTheme === "dark"
        ? "icon icon-theme icon-theme-dark"
        : "icon icon-theme icon-theme-light";
  }
  const label =
    appState.currentTheme === "light"
      ? "Switch to dark mode"
      : "Switch to light mode";
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
}

export function toggleTheme() {
  const nextTheme = appState.currentTheme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  window.localStorage.setItem("kokoro_theme", nextTheme);
}

export function initDisplayMode() {
  const savedTheme = window.localStorage.getItem("kokoro_theme");
  if (savedTheme) {
    applyTheme(savedTheme);
    return;
  }
  const prefersLight = window.matchMedia(
    "(prefers-color-scheme: light)",
  ).matches;
  applyTheme(prefersLight ? "light" : "dark");
}

export function setStatus(message, isError = false) {
  statusText.textContent = message;
  errorText.hidden = !isError;
  errorText.textContent = isError ? message : "";
}

export function setAuthPanelState(
  visible,
  message = "",
  isError = false,
  clearVisible = true,
) {
  if (authPanel) {
    authPanel.hidden = !visible;
  }
  appShell?.classList.toggle("shell-auth-locked", visible);
  appWorkspace?.classList.toggle("workspace-locked", visible);
  if (authMessage) {
    authMessage.textContent = message;
    authMessage.classList.toggle("auth-message-error", isError);
  }
  if (authClearButton) {
    authClearButton.hidden = !visible || !clearVisible;
  }
  if (authApiKeyInput) {
    authApiKeyInput.disabled = !visible;
    if (visible) {
      window.requestAnimationFrame(() => {
        authApiKeyInput.focus();
        authApiKeyInput.select();
      });
    }
  }
  if (authUnlockButton) {
    authUnlockButton.disabled = !visible;
  }
}

export function setUiLocked(locked) {
  appShell?.classList.toggle("shell-ui-locked", locked);
  appWorkspace?.classList.toggle("workspace-ui-locked", locked);
  [
    textInput,
    voiceInput,
    langInput,
    transportInput,
    formatInput,
    formatQualityInput,
    speedInput,
    pitchInput,
    chunkTargetInput,
    pauseMsInput,
    submitButton,
    exportButton,
  ].forEach((control) => {
    control.disabled = locked;
  });
  if (!locked) {
    refreshCustomSelect(voiceInput);
    refreshCustomSelect(langInput);
    refreshCustomSelect(transportInput);
    refreshCustomSelect(formatInput);
    refreshCustomSelect(formatQualityInput);
  }
}

export function setBusy(isBusy) {
  const uiLocked = textInput?.disabled === true;
  submitButton.disabled = isBusy || uiLocked;
  exportButton.disabled = isBusy || uiLocked || !appState.lastExportRequest;
  if (submitButtonLabel) {
    submitButtonLabel.textContent = isBusy ? "Generating..." : "Play";
  }
}

export function buildSynthesisPayload() {
  const isOpus = formatInput.value === "opus";
  const qualityValue = formatQualityInput.value;
  return {
    text: textInput.value,
    voice: voiceInput.value,
    speed: Number(speedInput.value),
    pitch: appState.pitchShiftingAvailable ? Number(pitchInput.value) : 0,
    lang: langInput.value,
    format: formatInput.value,
    opus_bitrate: isOpus ? qualityValue : appState.selectedOpusBitrate,
    wav_sample_rate: isOpus ? appState.selectedWavSampleRate : qualityValue,
    target_chunk_chars: Number(chunkTargetInput.value),
  };
}

export function getPitchSemitones() {
  return Number(pitchInput.value);
}

export function formatSignedSemitones(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} st`;
}

function formatPitchValue() {
  if (!appState.pitchShiftingAvailable) {
    return "Unavailable";
  }
  return formatSignedSemitones(getPitchSemitones());
}

export function updatePitchControlAvailability() {
  pitchInput.disabled = !appState.pitchShiftingAvailable;
  pitchInput.min = String(-appState.maxPitchSemitones);
  pitchInput.max = String(appState.maxPitchSemitones);
  if (!appState.pitchShiftingAvailable) {
    pitchInput.value = "0";
  } else {
    const clampedValue = Math.max(
      -appState.maxPitchSemitones,
      Math.min(appState.maxPitchSemitones, Number(pitchInput.value)),
    );
    pitchInput.value = String(clampedValue);
  }
  pitchValue.textContent = formatPitchValue();
}

export function buildExportFilename(format) {
  if (format === "opus") {
    return "kokoro-output.opus";
  }
  if (format === "pcm") {
    return "kokoro-output.pcm";
  }
  return "kokoro-output.wav";
}

export function populateVoices(voices) {
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

  ["af", "am", "bf", "bm", "jf", "jm", "zf", "zm", "other"].forEach(
    (groupKey) => {
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
    },
  );

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

export function formatDuration(seconds) {
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

export function formatBytes(bytes) {
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

export function formatWavRateLabel(rate) {
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

function isSampleRateFormat(format) {
  return format === "wav" || format === "pcm";
}

function formatOptionText(value) {
  if (value === "pcm") {
    return "PCM";
  }
  return value === "opus" ? "Opus" : "WAV";
}

function rebuildFormatOptions() {
  const options = appState.availableFormats;
  const currentValue = formatInput.value;
  const preferredValue = options.includes(currentValue)
    ? currentValue
    : options.includes(appState.selectedFormat)
      ? appState.selectedFormat
      : options[0];

  formatInput.innerHTML = "";
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = formatOptionText(value);
    formatInput.appendChild(option);
  });

  formatInput.value = preferredValue;
  appState.selectedFormat = formatInput.value;
  formatInput.disabled = options.length === 1;
  refreshCustomSelect(formatInput);
}

function rebuildFormatQualityOptions() {
  const isOpus = formatInput.value === "opus";
  const isSampleRate = isSampleRateFormat(formatInput.value);
  const options = isOpus
    ? appState.availableOpusBitrates
    : appState.availableWavSampleRates;
  const preferredValue = isOpus
    ? appState.selectedOpusBitrate
    : appState.selectedWavSampleRate;

  formatQualityLabel.textContent = isOpus
    ? "Opus Bitrate"
    : isSampleRate && formatInput.value === "pcm"
      ? "PCM Sample Rate"
      : "WAV Sample Rate";
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
    appState.selectedOpusBitrate = formatQualityInput.value;
  } else {
    appState.selectedWavSampleRate = formatQualityInput.value;
  }
  refreshCustomSelect(formatQualityInput);
}

export function updateTextStats() {
  const text = textInput.value.trim();
  const charTotal = text.length;
  const wordTotal = text ? text.split(/\s+/).length : 0;
  charCount.textContent = String(charTotal);
  wordCount.textContent = String(wordTotal);
  textStats.textContent = `${charTotal}c / ${wordTotal}w`;
}

export function updatePauseLabel() {
  const pauseValue = Number(pauseMsInput.value);
  pauseMode.textContent =
    pauseValue > 0 ? `${pauseValue} ms fixed` : "0 = auto";
}

export function updateFormatControlState() {
  rebuildFormatOptions();
  rebuildFormatQualityOptions();
}

export function syncTransportModeText() {
  streamMode.textContent =
    transportInput.value === "ws" ? "WebSocket" : "NDJSON";
}
