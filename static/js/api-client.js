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

const STATUS_STREAM_RECONNECT_BASE_MS = 1000;
const STATUS_STREAM_RECONNECT_MAX_MS = 30000;
const STATUS_LEADER_LEASE_MS = 5000;
const STATUS_LEADER_HEARTBEAT_MS = 2000;
const STATUS_LEADER_TAKEOVER_DELAY_MS = STATUS_LEADER_LEASE_MS + 500;
const STATUS_MESSAGE_MAX_AGE_MS = STATUS_LEADER_LEASE_MS + 1000;
const HEALTH_REQUEST_TIMEOUT_MS = 8000;

function getSafeLocalStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function getNavigatorLocks() {
  return globalThis.navigator?.locks ?? null;
}

function supportsBroadcastChannel() {
  return typeof BroadcastChannel === "function";
}

function getStatusLeaderLeaseKey(scopeId) {
  return `kokoro-status-leader:${scopeId}`;
}

function isAuthStatusScope(scopeId) {
  return typeof scopeId === "string" && scopeId.startsWith("auth-");
}

function readStatusLeaderLease(scopeId) {
  if (isAuthStatusScope(scopeId)) {
    return null;
  }
  const localStorage = getSafeLocalStorage();
  if (!localStorage) {
    return null;
  }
  const raw = localStorage.getItem(getStatusLeaderLeaseKey(scopeId));
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    if (
      typeof parsed?.tabId !== "string" ||
      !Number.isFinite(parsed?.expiresAt)
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeStatusLeaderLease(scopeId, tabId, expiresAt) {
  if (isAuthStatusScope(scopeId)) {
    return true;
  }
  const localStorage = getSafeLocalStorage();
  if (!localStorage) {
    return false;
  }
  try {
    localStorage.setItem(
      getStatusLeaderLeaseKey(scopeId),
      JSON.stringify({ tabId, expiresAt }),
    );
    return true;
  } catch {
    return false;
  }
}

function clearStatusLeaderLease(scopeId) {
  if (!scopeId) {
    return;
  }
  if (isAuthStatusScope(scopeId)) {
    return;
  }
  const localStorage = getSafeLocalStorage();
  if (!localStorage) {
    return;
  }
  const lease = readStatusLeaderLease(scopeId);
  if (lease?.tabId !== appState.tabId) {
    return;
  }
  try {
    localStorage.removeItem(getStatusLeaderLeaseKey(scopeId));
  } catch {
    // Ignore storage cleanup failures and fall back to lease expiry.
  }
}

function getStatusScopeId() {
  if (typeof appState.statusStreamScopeId === "string" && appState.statusStreamScopeId) {
    return appState.statusStreamScopeId;
  }
  if (appState.authRequired) {
    return null;
  }
  return "public";
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
  stopHealthMonitoring();
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
    void startHealthMonitoring();
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
    void startHealthMonitoring();
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
  stopHealthMonitoring();
  resetPlaybackState();
  appState.lastExportRequest = null;
  setUiLocked(true);
  setAuthPanelState(true, "Enter the configured API key to use this server.");
  setStatus("API key cleared.");
  if (authApiKeyInput) {
    authApiKeyInput.value = "";
  }
}

function clearStatusStreamReconnect() {
  if (appState.statusStreamReconnectTimerId === null) {
    return;
  }
  window.clearTimeout(appState.statusStreamReconnectTimerId);
  appState.statusStreamReconnectTimerId = null;
}

function clearStatusLeaderHeartbeat() {
  if (appState.statusLeaderHeartbeatTimerId === null) {
    return;
  }
  window.clearInterval(appState.statusLeaderHeartbeatTimerId);
  appState.statusLeaderHeartbeatTimerId = null;
}

function releaseStatusLeaderLock() {
  appState.statusLeaderLockRelease?.();
  appState.statusLeaderLockRelease = null;
}

function clearStatusLeaderCheck() {
  if (appState.statusLeaderCheckTimerId === null) {
    return;
  }
  window.clearTimeout(appState.statusLeaderCheckTimerId);
  appState.statusLeaderCheckTimerId = null;
}

function closeStatusBroadcastChannel() {
  if (!appState.statusBroadcastChannel) {
    appState.statusStreamScopeId = null;
    return;
  }
  appState.statusBroadcastChannel.close();
  appState.statusBroadcastChannel = null;
  appState.statusLeaderTabId = null;
  appState.statusIsLeader = false;
  appState.statusStreamScopeId = null;
}

function postStatusChannelMessage(message) {
  appState.statusBroadcastChannel?.postMessage({
    ...message,
    tabId: appState.tabId,
    scopeId: appState.statusStreamScopeId,
    sentAt: Date.now(),
  });
}

function scheduleLeaderTakeoverCheck() {
  clearStatusLeaderCheck();
  if (!appState.statusBroadcastChannel || document.hidden) {
    return;
  }
  appState.statusLeaderCheckTimerId = window.setTimeout(() => {
    appState.statusLeaderCheckTimerId = null;
    void startHealthMonitoring();
  }, STATUS_LEADER_TAKEOVER_DELAY_MS);
}

function renewStatusLeadership() {
  if (!appState.statusIsLeader || !appState.statusStreamScopeId) {
    return;
  }
  const wroteLease = writeStatusLeaderLease(
    appState.statusStreamScopeId,
    appState.tabId,
    Date.now() + STATUS_LEADER_LEASE_MS,
  );
  if (!wroteLease) {
    releaseStatusLeadership();
    appState.statusStreamAbortController?.abort();
    appState.statusStreamAbortController = null;
    scheduleStatusStreamReconnect();
    return;
  }
  postStatusChannelMessage({ type: "leader-heartbeat" });
}

function releaseStatusLeadership() {
  clearStatusLeaderHeartbeat();
  releaseStatusLeaderLock();
  if (appState.statusIsLeader) {
    clearStatusLeaderLease(appState.statusStreamScopeId);
    postStatusChannelMessage({ type: "leader-release" });
  }
  appState.statusIsLeader = false;
  appState.statusLeaderTabId = null;
}

function demoteStatusLeader(remoteTabId = null) {
  appState.statusIsLeader = false;
  appState.statusLeaderTabId = remoteTabId;
  clearStatusLeaderHeartbeat();
  releaseStatusLeaderLock();
  appState.statusStreamAbortController?.abort();
  appState.statusStreamAbortController = null;
}

async function tryAcquireStatusLeadership() {
  if (!appState.statusBroadcastChannel || !appState.statusStreamScopeId || document.hidden) {
    return false;
  }
  const scopeId = appState.statusStreamScopeId;
  if (appState.statusIsLeader && appState.statusLeaderTabId === appState.tabId) {
    return true;
  }
  const lease = readStatusLeaderLease(scopeId);
  const now = Date.now();
  if (lease && lease.expiresAt > now && lease.tabId !== appState.tabId) {
    appState.statusLeaderTabId = lease.tabId;
    scheduleLeaderTakeoverCheck();
    return false;
  }
  if (!(await tryAcquireStatusLeaderLock(scopeId))) {
    scheduleLeaderTakeoverCheck();
    return false;
  }
  if (
    !appState.statusBroadcastChannel ||
    appState.statusStreamScopeId !== scopeId ||
    document.hidden
  ) {
    releaseStatusLeaderLock();
    return false;
  }
  const wroteLease = writeStatusLeaderLease(
    scopeId,
    appState.tabId,
    now + STATUS_LEADER_LEASE_MS,
  );
  if (!wroteLease) {
    releaseStatusLeaderLock();
    return false;
  }
  if (isAuthStatusScope(scopeId)) {
    clearStatusLeaderCheck();
    appState.statusIsLeader = true;
    appState.statusLeaderTabId = appState.tabId;
    clearStatusLeaderHeartbeat();
    appState.statusLeaderHeartbeatTimerId = window.setInterval(
      renewStatusLeadership,
      STATUS_LEADER_HEARTBEAT_MS,
    );
    renewStatusLeadership();
    postStatusChannelMessage({ type: "leader-announce" });
    if (appState.lastStatusSnapshot) {
      postStatusChannelMessage({
        type: "health-update",
        payload: appState.lastStatusSnapshot,
      });
    }
    return true;
  }
  const confirmedLease = readStatusLeaderLease(scopeId);
  if (confirmedLease?.tabId !== appState.tabId) {
    releaseStatusLeaderLock();
    scheduleLeaderTakeoverCheck();
    return false;
  }
  clearStatusLeaderCheck();
  appState.statusIsLeader = true;
  appState.statusLeaderTabId = appState.tabId;
  clearStatusLeaderHeartbeat();
  appState.statusLeaderHeartbeatTimerId = window.setInterval(
    renewStatusLeadership,
    STATUS_LEADER_HEARTBEAT_MS,
  );
  renewStatusLeadership();
  postStatusChannelMessage({ type: "leader-announce" });
  if (appState.lastStatusSnapshot) {
    postStatusChannelMessage({
      type: "health-update",
      payload: appState.lastStatusSnapshot,
    });
  }
  return true;
}

function handleStatusChannelMessage(event) {
  const message = event.data;
  if (
    !message ||
    message.scopeId !== appState.statusStreamScopeId ||
    message.tabId === appState.tabId
  ) {
    return;
  }
  if (
    !Number.isFinite(message.sentAt) ||
    Math.abs(Date.now() - message.sentAt) > STATUS_MESSAGE_MAX_AGE_MS
  ) {
    return;
  }
  const currentLease = readStatusLeaderLease(appState.statusStreamScopeId);
  const isLeaseOwner = isAuthStatusScope(appState.statusStreamScopeId)
    ? true
    : currentLease?.tabId === message.tabId;
  if (message.type === "leader-announce" || message.type === "leader-heartbeat") {
    if (!isLeaseOwner) {
      return;
    }
    demoteStatusLeader(message.tabId);
    scheduleLeaderTakeoverCheck();
    return;
  }
  if (message.type === "leader-release") {
    if (appState.statusLeaderTabId === message.tabId) {
      appState.statusLeaderTabId = null;
      scheduleLeaderTakeoverCheck();
    }
    return;
  }
  if (message.type === "health-update") {
    if (!isLeaseOwner) {
      return;
    }
    demoteStatusLeader(message.tabId);
    appState.lastStatusSnapshot = message.payload;
    applyHealthState(message.payload, { quiet: true });
    scheduleLeaderTakeoverCheck();
    return;
  }
  if (message.type === "request-state" && appState.statusIsLeader && appState.lastStatusSnapshot) {
    postStatusChannelMessage({
      type: "health-update",
      payload: appState.lastStatusSnapshot,
    });
  }
}

async function ensureStatusBroadcastChannel() {
  const scopeId = getStatusScopeId();
  if (
    !scopeId ||
    isAuthStatusScope(scopeId) ||
    !supportsBroadcastChannel() ||
    (!isAuthStatusScope(scopeId) && !getSafeLocalStorage()) ||
    (isAuthStatusScope(scopeId) && !getNavigatorLocks()?.request)
  ) {
    releaseStatusLeadership();
    closeStatusBroadcastChannel();
    appState.statusStreamScopeId = scopeId;
    return false;
  }
  if (
    appState.statusBroadcastChannel &&
    appState.statusStreamScopeId === scopeId
  ) {
    return true;
  }
  releaseStatusLeadership();
  closeStatusBroadcastChannel();
  appState.statusStreamScopeId = scopeId;
  appState.statusBroadcastChannel = new BroadcastChannel(`kokoro-status:${scopeId}`);
  appState.statusBroadcastChannel.onmessage = handleStatusChannelMessage;
  postStatusChannelMessage({ type: "request-state" });
  scheduleLeaderTakeoverCheck();
  return true;
}

async function tryAcquireStatusLeaderLock(scopeId) {
  const locks = getNavigatorLocks();
  if (!locks?.request) {
    return true;
  }
  return new Promise((resolve) => {
    let resolved = false;
    void locks
      .request(
        `kokoro-status:${scopeId}`,
        { ifAvailable: true, mode: "exclusive" },
        async (lock) => {
          if (!lock) {
            resolved = true;
            resolve(false);
            return;
          }
          const keepAlive = new Promise((release) => {
            appState.statusLeaderLockRelease = release;
          });
          resolved = true;
          resolve(true);
          await keepAlive;
        },
      )
      .catch(() => {
        if (!resolved) {
          resolve(false);
        }
      });
  });
}

function stopHealthMonitoring() {
  clearStatusStreamReconnect();
  clearStatusLeaderCheck();
  releaseStatusLeadership();
  appState.statusStreamReconnectAttempts = 0;
  appState.statusStreamAbortController?.abort();
  appState.statusStreamAbortController = null;
  closeStatusBroadcastChannel();
}

function requestHealthMonitoringRestart() {
  appState.statusMonitoringRestartRequested = true;
}

function scheduleStatusStreamReconnect() {
  if (
    document.hidden ||
    appState.statusStreamReconnectTimerId !== null ||
    (appState.authRequired && !appState.apiKey)
  ) {
    return;
  }
  const delayMs = Math.min(
    STATUS_STREAM_RECONNECT_BASE_MS * 2 ** appState.statusStreamReconnectAttempts,
    STATUS_STREAM_RECONNECT_MAX_MS,
  );
  appState.statusStreamReconnectTimerId = window.setTimeout(() => {
    appState.statusStreamReconnectTimerId = null;
    void startHealthMonitoring();
  }, delayMs);
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split(/\r?\n/);
  let event = "message";
  const data = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  return {
    event,
    data: data.join("\n"),
  };
}

async function consumeStatusStream(stream, signal) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() || "";
      for (const rawEvent of events) {
        const parsedEvent = parseSseEvent(rawEvent);
        if (parsedEvent.event === "health_snapshot" && parsedEvent.data) {
          applyHealthState(JSON.parse(parsedEvent.data), { quiet: true });
        }
      }
      if (signal.aborted) {
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function connectStatusStream(abortController) {
  const response = await apiFetch("/api/health/stream", {
    headers: {
      Accept: "text/event-stream",
    },
    signal: abortController.signal,
    timeoutMs: HEALTH_REQUEST_TIMEOUT_MS,
  });
  if (response.status === 401 || response.status === 429) {
    throw await buildAuthError(response);
  }
  if (!response.ok || !response.body) {
    throw new Error("Status stream unavailable.");
  }
  await consumeStatusStream(response.body, abortController.signal);
  if (!abortController.signal.aborted) {
    throw new Error("Status stream disconnected.");
  }
}

async function startHealthMonitoring() {
  if (appState.statusMonitoringPromise) {
    return appState.statusMonitoringPromise;
  }
  const monitoringPromise = (async () => {
    if (document.hidden || (appState.authRequired && !appState.apiKey)) {
      return;
    }
    const useSharedChannel = await ensureStatusBroadcastChannel();
    if (useSharedChannel && !(await tryAcquireStatusLeadership())) {
      appState.statusStreamReconnectAttempts = 0;
      appState.statusStreamAbortController?.abort();
      appState.statusStreamAbortController = null;
      return;
    }
    if (appState.statusStreamAbortController) {
      return;
    }
    clearStatusStreamReconnect();
    const abortController = new AbortController();
    appState.statusStreamAbortController = abortController;
    try {
      await connectStatusStream(abortController);
      appState.statusStreamReconnectAttempts = 0;
    } catch (error) {
      if (appState.statusStreamAbortController !== abortController) {
        return;
      }
      appState.statusStreamAbortController = null;
      if (isAuthError(error)) {
        handleAuthFailure(error.message || "Authentication failed.");
        return;
      }
      releaseStatusLeadership();
      appState.statusStreamReconnectAttempts += 1;
      scheduleStatusStreamReconnect();
    }
  })();
  appState.statusMonitoringPromise = monitoringPromise;
  try {
    await monitoringPromise;
  } finally {
    if (appState.statusMonitoringPromise === monitoringPromise) {
      appState.statusMonitoringPromise = null;
    }
    if (appState.statusMonitoringRestartRequested) {
      appState.statusMonitoringRestartRequested = false;
      if (!document.hidden && (!appState.authRequired || appState.apiKey)) {
        window.setTimeout(() => {
          void startHealthMonitoring();
        }, 0);
      }
    }
  }
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
    applyHealthState(health, { quiet, capabilities });
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

function applyHealthState(health, options = {}) {
  const quiet = options.quiet === true;
  const capabilities = options.capabilities ?? null;
  const nextStatusScopeId =
    typeof health?.status_stream_scope === "string" && health.status_stream_scope
      ? health.status_stream_scope
      : null;
  const statusScopeChanged =
    Boolean(nextStatusScopeId) &&
    appState.statusStreamScopeId !== null &&
    appState.statusStreamScopeId !== nextStatusScopeId;
  if (
    statusScopeChanged &&
    appState.statusBroadcastChannel &&
    appState.statusStreamScopeId
  ) {
    stopHealthMonitoring();
  }
  if (typeof health?.status_stream_scope === "string" && health.status_stream_scope) {
    appState.statusStreamScopeId = health.status_stream_scope;
  }
  if (
    statusScopeChanged &&
    !document.hidden &&
    (!appState.authRequired || appState.apiKey)
  ) {
    requestHealthMonitoringRestart();
    void startHealthMonitoring();
  }
  appState.lastStatusSnapshot = health;
  if (appState.statusIsLeader && appState.statusBroadcastChannel) {
    postStatusChannelMessage({
      type: "health-update",
      payload: health,
    });
  }
  if (capabilities) {
    populateVoices(capabilities.voices || []);
    if (Array.isArray(capabilities.formats) && capabilities.formats.length > 0) {
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
      if (!appState.availableOpusBitrates.includes(appState.selectedOpusBitrate)) {
        appState.selectedOpusBitrate = appState.availableOpusBitrates[0];
      }
    }
    if (
      Array.isArray(capabilities.wav_sample_rates) &&
      capabilities.wav_sample_rates.length > 0
    ) {
      appState.availableWavSampleRates = capabilities.wav_sample_rates;
      if (!appState.availableWavSampleRates.includes(appState.selectedWavSampleRate)) {
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
    typeof health.active_provider === "string" ? health.active_provider : null;
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
}

function handleVisibilityChange() {
  if (appState.authRequired && !appState.apiKey) {
    stopHealthMonitoring();
    return;
  }
  if (!document.hidden) {
    void startHealthMonitoring();
    return;
  }
  stopHealthMonitoring();
}

document.addEventListener("visibilitychange", handleVisibilityChange);
window.addEventListener("pagehide", stopHealthMonitoring);
window.addEventListener("beforeunload", stopHealthMonitoring);

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
