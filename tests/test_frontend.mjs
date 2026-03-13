import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { cp, mkdtemp, rm } from "node:fs/promises";
import { URL as NodeURL, pathToFileURL } from "node:url";

class FakeClassList {
  constructor() {
    this._values = new Set();
  }

  add(...tokens) {
    tokens.forEach((token) => this._values.add(token));
  }

  remove(...tokens) {
    tokens.forEach((token) => this._values.delete(token));
  }

  toggle(token, force) {
    if (force === true) {
      this._values.add(token);
      return true;
    }
    if (force === false) {
      this._values.delete(token);
      return false;
    }
    if (this._values.has(token)) {
      this._values.delete(token);
      return false;
    }
    this._values.add(token);
    return true;
  }

  contains(token) {
    return this._values.has(token);
  }
}

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.classList = new FakeClassList();
    this.attributes = new Map();
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.textContent = "";
    this.type = "";
    this.id = "";
    this.label = "";
    this.selected = false;
    this.max = 0;
    this.min = 0;
    this.currentTime = 0;
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  remove() {
    if (!this.parentNode) {
      return;
    }
    this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    this.parentNode = null;
  }

  querySelector(selector) {
    if (selector === ".button-label") {
      return this.children.find((child) => child.className === "button-label") ?? null;
    }
    const optionMatch = selector.match(/^option\[value="([^"]+)"\]$/);
    if (optionMatch) {
      return this.options.find((option) => option.value === optionMatch[1]) ?? null;
    }
    return null;
  }

  querySelectorAll() {
    return [];
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    this.listeners.set(
      type,
      listeners.filter((candidate) => candidate !== listener),
    );
  }

  dispatchEvent(event) {
    const listeners = this.listeners.get(event.type) ?? [];
    listeners.forEach((listener) => listener.call(this, event));
    return true;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  insertAdjacentElement(_position, element) {
    if (!this.parentNode) {
      return element;
    }
    const index = this.parentNode.children.indexOf(this);
    this.parentNode.children.splice(index + 1, 0, element);
    element.parentNode = this.parentNode;
    return element;
  }

  focus() {}

  select() {}

  pause() {}

  load() {}

  async play() {}

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  closest(selector) {
    if (selector === ".custom-select-option" && this.className.includes("custom-select-option")) {
      return this;
    }
    return null;
  }

  get options() {
    if (this.tagName !== "SELECT") {
      return [];
    }
    return this.children.flatMap((child) => {
      if (child.tagName === "OPTGROUP") {
        return child.children;
      }
      return child.tagName === "OPTION" ? [child] : [];
    });
  }

  get selectedIndex() {
    return this.options.findIndex((option) => option.selected || option.value === this.value);
  }

  set innerHTML(_value) {
    this.children = [];
  }

  get innerHTML() {
    return "";
  }
}

class FakeDocument {
  constructor() {
    this._elementsById = new Map();
    this._queryResults = new Map();
    this._listeners = new Map();
    this.body = new FakeElement("body", this);
    this.documentElement = new FakeElement("html", this);
    this.hidden = false;
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }

  getElementById(id) {
    return this._elementsById.get(id) ?? null;
  }

  querySelector(selector) {
    const values = this._queryResults.get(selector) ?? [];
    return values[0] ?? null;
  }

  querySelectorAll(selector) {
    return this._queryResults.get(selector) ?? [];
  }

  addEventListener(type, listener) {
    const listeners = this._listeners.get(type) ?? [];
    listeners.push(listener);
    this._listeners.set(type, listeners);
  }

  dispatchEvent(event) {
    const listeners = this._listeners.get(event.type) ?? [];
    listeners.forEach((listener) => listener.call(this, event));
    return true;
  }

  registerElement(id, element) {
    element.id = id;
    this._elementsById.set(id, element);
    return element;
  }

  setQueryResult(selector, values) {
    this._queryResults.set(selector, values);
  }
}

function createButton(document, id, labelText) {
  const button = document.registerElement(id, document.createElement("button"));
  const label = document.createElement("span");
  label.className = "button-label";
  label.textContent = labelText;
  button.appendChild(label);
  return button;
}

function createOption(document, value, text, { selected = false } = {}) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = text;
  option.selected = selected;
  return option;
}

function createSelect(document, id, options, value) {
  const select = document.registerElement(id, document.createElement("select"));
  options.forEach((option) => select.appendChild(option));
  select.value = value;
  return select;
}

function installDom() {
  const document = new FakeDocument();

  document.registerElement("ttsForm", document.createElement("form"));
  const textInput = document.registerElement("text", document.createElement("textarea"));
  textInput.value = "The future sounds calmer when it is rendered locally.";
  createSelect(
    document,
    "voice",
    [createOption(document, "af_heart", "Heart", { selected: true })],
    "af_heart",
  );
  createSelect(
    document,
    "lang",
    [createOption(document, "en-us", "English (US)", { selected: true })],
    "en-us",
  );
  createSelect(
    document,
    "transport",
    [
      createOption(document, "ndjson", "NDJSON", { selected: true }),
      createOption(document, "ws", "WebSocket"),
    ],
    "ndjson",
  );
  createSelect(
    document,
    "format",
    [createOption(document, "wav", "WAV", { selected: true })],
    "wav",
  );
  createSelect(
    document,
    "formatQuality",
    [createOption(document, "native", "Native", { selected: true })],
    "native",
  );
  document.registerElement("formatQualityLabel", document.createElement("label"));
  const speedInput = document.registerElement("speed", document.createElement("input"));
  speedInput.value = "1";
  document.registerElement("speedValue", document.createElement("span")).textContent = "1.00x";
  const pitchInput = document.registerElement("pitch", document.createElement("input"));
  pitchInput.value = "0";
  document.registerElement("pitchValue", document.createElement("span")).textContent = "+0.0 st";
  const chunkTargetInput = document.registerElement("chunkTarget", document.createElement("input"));
  chunkTargetInput.value = "360";
  const pauseMsInput = document.registerElement("pauseMs", document.createElement("input"));
  pauseMsInput.value = "0";
  document.registerElement("pauseMode", document.createElement("span")).textContent = "0 = auto";
  document.registerElement("statusText", document.createElement("div")).textContent = "Ready for synthesis";
  const errorText = document.registerElement("errorText", document.createElement("div"));
  errorText.hidden = true;
  createButton(document, "submitButton", "Play");
  document.registerElement("stopButton", document.createElement("button"));
  createButton(document, "exportButton", "Export Audio");
  document.registerElement("player", document.createElement("audio"));
  document.registerElement("systemStatus", document.createElement("button"));
  document.registerElement("providerBadge", document.createElement("div"));
  document.registerElement("themeToggle", document.createElement("button"));
  document.registerElement("appShell", document.createElement("div"));
  document.registerElement("appWorkspace", document.createElement("div"));
  document.registerElement("authPanel", document.createElement("div"));
  document.registerElement("authForm", document.createElement("form"));
  document.registerElement("authApiKey", document.createElement("input"));
  document.registerElement("authUnlockButton", document.createElement("button"));
  document.registerElement("authClearButton", document.createElement("button"));
  document.registerElement("authMessage", document.createElement("div"));
  document.registerElement("genTime", document.createElement("span")).textContent = "--";
  const charCard = document.createElement("div");
  const wordCard = document.createElement("div");
  const textCard = document.createElement("div");
  const vramCard = document.createElement("div");
  document.setQueryResult(".stat-card-char", [charCard]);
  document.setQueryResult(".stat-card-word", [wordCard]);
  document.setQueryResult(".stat-card-text", [textCard]);
  document.setQueryResult(".stat-card-vram", [vramCard]);
  document.registerElement("charCount", document.createElement("span")).textContent = "53";
  document.registerElement("wordCount", document.createElement("span")).textContent = "9";
  document.registerElement("textStats", document.createElement("span")).textContent = "53c / 9w";
  document.registerElement("audioDuration", document.createElement("span")).textContent = "--";
  document.registerElement("fileSize", document.createElement("span")).textContent = "--";
  document.registerElement("gpuVram", document.createElement("span")).textContent = "--";
  document.registerElement("chunkProgress", document.createElement("span")).textContent = "0 / 0";
  const chunkBar = document.registerElement("chunkBar", document.createElement("progress"));
  chunkBar.max = 1;
  chunkBar.value = 0;
  document.registerElement("streamMode", document.createElement("span")).textContent = "NDJSON";
  document.registerElement("apiExampleTitle", document.createElement("div"));
  document.registerElement("apiExampleCode", document.createElement("pre"));
  document.setQueryResult(".api-endpoint", []);
  document.setQueryResult(".chip", []);

  return document;
}

function installGlobals(document, fetchImpl, options = {}) {
  globalThis.document = document;
  globalThis.Element = FakeElement;
  globalThis.Event = class Event {
    constructor(type, options = {}) {
      this.type = type;
      this.bubbles = Boolean(options.bubbles);
      this.target = null;
    }
  };

  globalThis.window = {
    document,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    requestAnimationFrame: (callback) => setTimeout(() => callback(Date.now()), 0),
    cancelAnimationFrame: clearTimeout,
    localStorage: {
      getItem() {
        return null;
      },
      setItem() {},
    },
    matchMedia() {
      return { matches: false };
    },
    location: {
      protocol: "http:",
      host: "127.0.0.1:8000",
    },
  };

  globalThis.fetch = fetchImpl;
  globalThis.WebSocket = options.WebSocket ?? class UnsupportedWebSocket {};
  globalThis.URL = {
    createObjectURL() {
      return "blob:fake";
    },
    revokeObjectURL() {},
  };
}

async function createFrontendSandbox() {
  const sandboxRoot = await mkdtemp(path.join(os.tmpdir(), "kokoro-frontend-"));
  const sourceRoot = path.resolve("static/js");
  const targetRoot = path.join(sandboxRoot, "static", "js");
  await cp(sourceRoot, targetRoot, { recursive: true });
  return {
    importModule(relativePath) {
      return importFrontendModule(relativePath, sandboxRoot);
    },
    cleanup() {
      return rm(sandboxRoot, { recursive: true, force: true });
    },
  };
}

async function importFrontendModule(relativePath, sandboxRoot) {
  const absolutePath = sandboxRoot
    ? path.join(sandboxRoot, relativePath.replace(/^\.\//, ""))
    : path.resolve(relativePath);
  const url = new NodeURL(
    `${pathToFileURL(absolutePath).href}?t=${Date.now()}-${Math.random()}`,
  );
  return import(url.href);
}

test("synthesize resets busy state when stream startup fails", async (t) => {
  const document = installDom();
  const fetchCalls = [];
  const sandbox = await createFrontendSandbox();
  t.after(() => sandbox.cleanup());
  installGlobals(document, async (input) => {
    fetchCalls.push(String(input));
    throw new TypeError("Failed to fetch");
  });

  const apiClient = await sandbox.importModule("./static/js/api-client.js");

  await assert.doesNotReject(
    Promise.race([
      apiClient.synthesize({ preventDefault() {} }),
      new Promise((_, reject) => {
        setTimeout(() => reject(new Error("synthesize timed out")), 250);
      }),
    ]),
  );

  assert.deepEqual(fetchCalls, ["/api/speak-stream"]);
  assert.equal(document.getElementById("statusText").textContent, "Failed to fetch");
  assert.equal(document.getElementById("submitButton").disabled, false);
  assert.equal(
    document.getElementById("submitButton").querySelector(".button-label").textContent,
    "Play",
  );
});

test("loadHealth surfaces 429 detail and avoids capabilities fetch on auth failure", async (t) => {
  const document = installDom();
  const fetchCalls = [];
  const sandbox = await createFrontendSandbox();
  t.after(() => sandbox.cleanup());
  installGlobals(document, async (input) => {
    fetchCalls.push(String(input));
    return new Response(
      JSON.stringify({ detail: "Too many authentication failures. Try again later." }),
      {
        status: 429,
        headers: { "Content-Type": "application/json" },
      },
    );
  });

  const apiClient = await sandbox.importModule("./static/js/api-client.js");

  await assert.rejects(
    apiClient.loadHealth(),
    /Too many authentication failures\. Try again later\./,
  );
  assert.deepEqual(fetchCalls, ["/api/health"]);
});

test("submitApiKey keeps the UI locked when backend validation fails", async (t) => {
  const document = installDom();
  let requestCount = 0;
  const sandbox = await createFrontendSandbox();
  t.after(() => sandbox.cleanup());
  installGlobals(document, async (input) => {
    requestCount += 1;
    if (String(input) === "/api/public-config") {
      return new Response(JSON.stringify({ auth_required: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    throw new TypeError("Network down");
  });

  const apiClient = await sandbox.importModule("./static/js/api-client.js");

  await apiClient.initializeAccess();

  const unlocked = await apiClient.submitApiKey("wrong-key");

  assert.equal(unlocked, false);
  assert.equal(requestCount, 2);
  assert.equal(document.getElementById("submitButton").disabled, true);
  assert.equal(document.getElementById("statusText").textContent, "Unable to reach the backend.");
  assert.equal(document.getElementById("authMessage").textContent, "Network down");
});

test("synthesize relocks the UI after NDJSON auth failure", async (t) => {
  const document = installDom();
  const fetchCalls = [];
  const sandbox = await createFrontendSandbox();
  t.after(() => sandbox.cleanup());
  installGlobals(document, async (input) => {
    const target = String(input);
    fetchCalls.push(target);
    if (target === "/api/public-config") {
      return new Response(JSON.stringify({ auth_required: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (target === "/api/health") {
      return new Response(JSON.stringify({ ok: true, gpu: {}, runtime_activity: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (target === "/api/capabilities") {
      return new Response(
        JSON.stringify({
          voices: ["af_heart"],
          formats: ["wav"],
          opus_bitrates: ["32k"],
          wav_sample_rates: ["native"],
          websocket_streaming: true,
          pitch_shifting: true,
          max_pitch_semitones: 6,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
    return new Response(JSON.stringify({ detail: "Authentication failed." }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  });

  const apiClient = await sandbox.importModule("./static/js/api-client.js");

  await apiClient.initializeAccess();
  await apiClient.submitApiKey("bad-key");

  await apiClient.synthesize({ preventDefault() {} });

  assert.deepEqual(fetchCalls, [
    "/api/public-config",
    "/api/health",
    "/api/capabilities",
    "/api/speak-stream",
  ]);
  assert.equal(document.getElementById("submitButton").disabled, true);
  assert.equal(document.getElementById("authPanel").hidden, false);
  assert.equal(document.getElementById("authMessage").textContent, "Authentication failed.");
});

test("synthesize relocks the UI after WebSocket token auth throttling", async (t) => {
  const document = installDom();
  const fetchCalls = [];
  const sandbox = await createFrontendSandbox();
  t.after(() => sandbox.cleanup());
  class FakeWebSocket {
    constructor(url) {
      this.url = url;
      this.binaryType = "arraybuffer";
      setTimeout(() => this.onopen?.(), 0);
    }

    send() {}

    close(code = 1000, reason = "") {
      setTimeout(() => this.onclose?.({ code, reason }), 0);
    }
  }

  installGlobals(
    document,
    async () => {
      return new Response(
        JSON.stringify({ detail: "Too many authentication failures. Try again later." }),
        {
          status: 429,
          headers: { "Content-Type": "application/json" },
        },
      );
    },
    { WebSocket: FakeWebSocket },
  );

  const apiClient = await sandbox.importModule("./static/js/api-client.js");

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const target = String(input);
    fetchCalls.push(target);
    if (target === "/api/public-config") {
      return new Response(JSON.stringify({ auth_required: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (target === "/api/health") {
      return new Response(JSON.stringify({ ok: true, gpu: {}, runtime_activity: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (target === "/api/capabilities") {
      return new Response(
        JSON.stringify({
          voices: ["af_heart"],
          formats: ["wav"],
          opus_bitrates: ["32k"],
          wav_sample_rates: ["native"],
          websocket_streaming: true,
          pitch_shifting: true,
          max_pitch_semitones: 6,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
    return originalFetch(input);
  };

  await apiClient.initializeAccess();
  await apiClient.submitApiKey("bad-key");
  document.getElementById("transport").options[0].selected = false;
  document.getElementById("transport").options[1].selected = true;
  document.getElementById("transport").value = "ws";

  await apiClient.synthesize({ preventDefault() {} });

  assert.deepEqual(fetchCalls, [
    "/api/public-config",
    "/api/health",
    "/api/capabilities",
    "/api/ws-token",
  ]);
  assert.equal(document.getElementById("submitButton").disabled, true);
  assert.equal(document.getElementById("authPanel").hidden, false);
  assert.equal(
    document.getElementById("statusText").textContent,
    "Too many authentication failures. Try again later.",
  );
});
