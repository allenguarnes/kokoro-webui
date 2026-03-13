import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
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
    this.body = new FakeElement("body", this);
    this.documentElement = new FakeElement("html", this);
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

  addEventListener() {}

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

function installGlobals(document, fetchImpl) {
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
  globalThis.URL = {
    createObjectURL() {
      return "blob:fake";
    },
    revokeObjectURL() {},
  };
}

async function importFrontendModule(relativePath) {
  const absolutePath = path.resolve(relativePath);
  const url = new NodeURL(
    `${pathToFileURL(absolutePath).href}?t=${Date.now()}-${Math.random()}`,
  );
  return import(url.href);
}

test("synthesize resets busy state when stream startup fails", async () => {
  const document = installDom();
  const fetchCalls = [];
  installGlobals(document, async (input) => {
    fetchCalls.push(String(input));
    throw new TypeError("Failed to fetch");
  });

  const apiClient = await importFrontendModule("./static/js/api-client.js");

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

test("loadHealth surfaces 429 detail and avoids capabilities fetch on auth failure", async () => {
  const document = installDom();
  const fetchCalls = [];
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

  const apiClient = await importFrontendModule("./static/js/api-client.js");

  await assert.rejects(
    apiClient.loadHealth(),
    /Too many authentication failures\. Try again later\./,
  );
  assert.deepEqual(fetchCalls, ["/api/health"]);
});
