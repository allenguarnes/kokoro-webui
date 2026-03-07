import {
  audioDuration,
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
  textInput,
  themeToggle,
  transportInput,
  voiceInput,
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

export function setSystemStatus(runtimeReady, websocketReady) {
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

export function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  exportButton.disabled = isBusy || !appState.lastExportRequest;
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
  return `kokoro-output.${format === "opus" ? "opus" : "wav"}`;
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

function rebuildFormatQualityOptions() {
  const isOpus = formatInput.value === "opus";
  const options = isOpus
    ? appState.availableOpusBitrates
    : appState.availableWavSampleRates;
  const preferredValue = isOpus
    ? appState.selectedOpusBitrate
    : appState.selectedWavSampleRate;

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
    appState.selectedOpusBitrate = formatQualityInput.value;
  } else {
    appState.selectedWavSampleRate = formatQualityInput.value;
  }
  refreshCustomSelect(formatQualityInput);
}

export function updateTextStats() {
  const text = textInput.value.trim();
  charCount.textContent = String(text.length);
  wordCount.textContent = text ? String(text.split(/\s+/).length) : "0";
}

export function updatePauseLabel() {
  const pauseValue = Number(pauseMsInput.value);
  pauseMode.textContent =
    pauseValue > 0 ? `${pauseValue} ms fixed` : "0 = auto";
}

export function updateFormatControlState() {
  rebuildFormatQualityOptions();
}

export function syncTransportModeText() {
  streamMode.textContent =
    transportInput.value === "ws" ? "WebSocket" : "NDJSON";
}
