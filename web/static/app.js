"use strict";

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"];
const SUPPORTED = [".pdf", ".txt", ".md", ...IMAGE_TYPES];
const CITATION = /\[([^\[\]\n]+?\.(?:pdf|txt|md))(?: p\.(\d+))?\]|\[([WAUP]?\d{1,2})\]/gi;
const TOOL_LABELS = {
  search_documents: "Searching your documents",
  read_document: "Reading",
  web_search: "Searching the web",
  fetch_url: "Reading",
  wikipedia_search: "Searching Wikipedia",
  arxiv_search: "Searching arXiv",
  get_time: "Checking the time in",
  place_info: "Looking up",
};
// Which argument to show next to each tool's progress step.
const TOOL_SUBJECT = { query: (v) => `for “${v}”`, url: (v) => v, name: (v) => v, location: (v) => v };

const el = (id) => document.getElementById(id);
const dom = {
  messages: el("messages"),
  empty: el("empty"),
  composer: el("composer"),
  input: el("input"),
  send: el("send"),
  attach: el("attach"),
  fileInput: el("file-input"),
  docList: el("doc-list"),
  docsEmpty: el("docs-empty"),
  newChat: el("new-chat"),
  sidebar: el("sidebar"),
  scrim: el("scrim"),
  menu: el("menu"),
  overlay: el("drop-overlay"),
  toast: el("toast"),
  attachments: el("attachments"),
  usage: el("usage"),
  authDialog: el("auth-dialog"),
  authForm: el("auth-form"),
  authInput: el("auth-key"),
  authError: el("auth-error"),
};

const state = {
  sessionId: null,
  busy: false,
  abort: null,
  documents: [],
  tokensUsed: 0,
  tokenBudget: 0,
  ready: Promise.resolve(),
  attachments: [],
};

// ---------- Local storage (per-browser conveniences only; it can be unavailable) ----------

const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* ignore */ }
  },
};

// ---------- API ----------

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function describeDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  return String(detail);
}

async function apiFetch(path, options = {}) {
  let response;
  for (;;) {
    const headers = new Headers(options.headers || {});
    const key = store.get("apiKey");
    if (key) headers.set("Authorization", `Bearer ${key}`);
    try {
      response = await fetch(`/api${path}`, { ...options, headers });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new ApiError("Can't reach the server. Is it still running?", 0);
    }
    if (response.status !== 401) break;
    await askForApiKey(key ? "That key was rejected. Try again." : "");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(describeDetail(body.detail) || `Request failed (${response.status})`, response.status);
  }
  return response;
}

async function request(path, options) {
  const response = await apiFetch(path, options);
  return response.status === 204 ? null : response.json();
}

// Concurrent 401s share one prompt instead of each opening the dialog.
let pendingKeyPrompt = null;

function askForApiKey(message) {
  if (pendingKeyPrompt) return pendingKeyPrompt;
  dom.authError.textContent = message;
  dom.authError.hidden = !message;
  dom.authInput.value = "";
  dom.authDialog.showModal();
  pendingKeyPrompt = new Promise((resolve) => {
    dom.authForm.onsubmit = (event) => {
      event.preventDefault();
      store.set("apiKey", dom.authInput.value.trim());
      dom.authDialog.close();
      pendingKeyPrompt = null;
      resolve();
    };
  });
  return pendingKeyPrompt;
}

async function startSession() {
  const { session_id, token_budget } = await request("/sessions", { method: "POST" });
  state.sessionId = session_id;
  store.set("sessionId", session_id);
  setUsage(0, token_budget);
}

async function restoreSession() {
  const saved = store.get("sessionId");
  if (saved) {
    try {
      const info = await request(`/sessions/${saved}`);
      state.sessionId = saved;
      renderDocuments(info.documents);
      setUsage(info.tokens_used, info.token_budget);
      for (const turn of info.turns) {
        addUserMessage(turn.question, turn.attachments || []);
        addMessage("assistant", ...renderAssistant(turn.result));
      }
      return;
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  await startSession();
}

// Sessions expire after inactivity; recover transparently with a fresh one.
async function withSession(call) {
  await state.ready;
  try {
    return await call(state.sessionId);
  } catch (error) {
    if (error.status !== 404 || !String(error.message).startsWith("Session")) throw error;
    await startSession();
    renderDocuments([]);
    showToast("Your previous session expired, so a new one was started.");
    return call(state.sessionId);
  }
}

async function streamChat(sessionId, message, attachments, handlers, signal) {
  const response = await apiFetch(`/sessions/${sessionId}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, attachments }),
    signal,
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      handlers[event]?.(JSON.parse(data || "{}"));
    }
  }
}

// ---------- Rendering helpers ----------

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child != null) node.append(child);
  }
  return node;
}

const ICONS = {
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  send: '<path d="M5 12h14m0 0-6-6m6 6-6 6"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name];
  return svg;
}

function formatTokens(n) {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}

function renderMarkdown(text, webLinks = new Map()) {
  const container = h("div", { class: "answer" });
  if (window.marked && window.DOMPurify) {
    // Answers include text scraped from the web, so the HTML must be sanitised.
    container.innerHTML = DOMPurify.sanitize(marked.parse(text));
    container.querySelectorAll("a[href]").forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
  } else {
    container.classList.add("plain");
    container.textContent = text;
  }
  decorateCitations(container, webLinks);
  return container;
}

// Web results are numbered per search call, so [n] maps to a URL only when the turn made one search.
// Maps a citation tag ("2", "W1", "A1", "U1") to its source. Tags restart at 1 on every call
// of the same tool, so a tag is only linked when exactly one call could have produced it.
function webCitationLinks(turn) {
  const callsPerPrefix = {};
  const prefixOf = { web_search: "", wikipedia_search: "W", arxiv_search: "A", fetch_url: "U", place_info: "P" };
  for (const call of turn.tool_calls) {
    const prefix = prefixOf[call.name];
    if (prefix !== undefined) callsPerPrefix[prefix] = (callsPerPrefix[prefix] || 0) + 1;
  }
  const links = new Map();
  for (const source of turn.sources) {
    if (source.kind !== "web" || !source.ref) continue;
    const prefix = source.ref.replace(/\d+$/, "");
    if (callsPerPrefix[prefix] === 1 && !links.has(source.ref)) links.set(source.ref, source);
  }
  return links;
}

function decorateCitations(root, webLinks) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement.closest("code, pre, a") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const node of nodes) {
    const text = node.nodeValue;
    CITATION.lastIndex = 0;
    if (!CITATION.test(text)) continue;
    CITATION.lastIndex = 0;

    const fragment = document.createDocumentFragment();
    let last = 0;
    for (const match of text.matchAll(CITATION)) {
      fragment.append(text.slice(last, match.index));
      if (match[1]) {
        const label = match[2] ? `${match[1]} p.${match[2]}` : match[1];
        fragment.append(h("span", { class: "cite", title: "Document source", text: label }));
      } else {
        const source = webLinks.get(match[3].toUpperCase());
        const url = source && safeUrl(source.url);
        fragment.append(url
          ? h("a", { class: "cite", href: url.href, target: "_blank", rel: "noopener noreferrer", title: source.label, text: match[3] })
          : h("span", { class: "cite", title: "Web result", text: match[3] }));
      }
      last = match.index + match[0].length;
    }
    fragment.append(text.slice(last));
    node.replaceWith(fragment);
  }
}

function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed : null;
  } catch {
    return null;
  }
}

function renderSources(sources) {
  if (!sources.length) return null;
  const items = sources.map((source) => {
    const url = source.kind === "web" ? safeUrl(source.url) : null;
    if (url) {
      return h("a", { class: "source", href: url.href, target: "_blank", rel: "noopener noreferrer", title: url.href },
        icon("globe"), h("span", { text: source.label || url.hostname }));
    }
    return h("span", { class: "source" }, icon(source.kind === "web" ? "globe" : "file"), h("span", { text: source.label }));
  });
  return h("div", { class: "sources" },
    h("div", { class: "sources-title", text: "Sources consulted" }),
    h("div", { class: "source-list" }, items));
}

function renderTrace(turn) {
  const { analysis, tool_calls: calls, usage } = turn;
  const parts = [analysis.needs_retrieval ? "Research needed" : "Answered directly"];
  if (calls.length) parts.push(`${calls.length} tool call${calls.length > 1 ? "s" : ""}`);
  if (turn.latency_ms) parts.push(`${(turn.latency_ms / 1000).toFixed(1)} s`);
  const tokens = (usage?.input_tokens || 0) + (usage?.output_tokens || 0);
  if (tokens) parts.push(`${formatTokens(tokens)} tokens`);

  const body = h("div", { class: "trace-body" });
  if (analysis.sub_questions.length) {
    body.append(h("div", {},
      h("div", { class: "trace-title", text: "Sub-questions" }),
      h("ul", {}, analysis.sub_questions.map((q) => h("li", { text: q })))));
  }
  if (calls.length) {
    body.append(h("div", {},
      h("div", { class: "trace-title", text: "Tool calls" }),
      h("div", { class: "tool-calls" }, calls.map((call) =>
        h("div", { class: "tool-call" },
          h("div", { class: "tool-head" },
            h("code", { text: `${call.name}(${JSON.stringify(call.args)})` }),
            call.duration_ms ? h("span", { class: "muted", text: `${call.duration_ms} ms` }) : null,
            call.flagged ? h("span", { class: "flag", text: "injection flagged" }) : null),
          h("details", {}, h("summary", { class: "muted", text: "Show result" }), h("pre", { text: call.output })))))));
  }
  if (usage?.llm_calls) {
    body.append(h("div", { class: "muted small", text:
      `${usage.llm_calls} model calls · ${usage.input_tokens} input / ${usage.output_tokens} output tokens · trace ${turn.trace_id}` }));
  }
  return h("details", { class: "trace" }, h("summary", { text: parts.join(" · ") }), body);
}

function addMessage(role, ...content) {
  dom.empty.hidden = true;
  const bubble = h("div", { class: "bubble" }, content);
  const message = h("div", { class: `msg ${role}` }, bubble);
  dom.messages.append(message);
  message.scrollIntoView({ block: "end" });
  return message;
}

function renderAssistant(turn) {
  const answer = turn.answer || "_(The model returned an empty answer.)_";
  const parts = [renderMarkdown(answer, webCitationLinks(turn)), renderSources(turn.sources)];
  const notices = [...(turn.warnings || [])];
  if (turn.removed_citations.length) {
    const n = turn.removed_citations.length;
    notices.push(`Removed ${n} citation${n > 1 ? "s" : ""} that no retrieved passage supported: ${turn.removed_citations.join(", ")}`);
  }
  for (const text of notices) parts.push(h("div", { class: "notice", text }));
  parts.push(renderTrace(turn));
  return parts.filter(Boolean);
}

// ---------- Chat ----------

function setBusy(busy) {
  state.busy = busy;
  dom.newChat.disabled = busy;
  dom.send.replaceChildren(icon(busy ? "stop" : "send"));
  dom.send.classList.toggle("stop", busy);
  dom.send.setAttribute("aria-label", busy ? "Stop" : "Send");
  dom.input.placeholder = busy ? "Researching…" : "Ask anything, or attach files with +";
  updateSendState();
}

function addUserMessage(text, attachments) {
  const files = attachments.length ? h("div", { class: "msg-files" }, attachments.map((name) => fileChip(name))) : null;
  const message = addMessage("user", h("div", { class: "msg-text", text }));
  if (files) message.prepend(files);
  return message;
}

function defaultPrompt(names) {
  return names.length === 1 ? "Summarise the key points of this document." : "Summarise the key points of these documents.";
}

function setUsage(used, budget) {
  state.tokensUsed = used;
  state.tokenBudget = budget || state.tokenBudget;
  dom.usage.hidden = !state.tokenBudget;
  dom.usage.textContent = `${formatTokens(used)} of ${formatTokens(state.tokenBudget)} session tokens used`;
}

class Progress {
  constructor(bubble) {
    this.bubble = bubble;
    this.steps = h("ul", { class: "steps" });
    this.answer = h("div", { class: "answer streaming" });
    this.tools = new Map();
    this.thinking = this.addStep("Understanding the question");
    bubble.replaceChildren(this.steps, this.answer);
  }

  addStep(text) {
    const step = h("li", { class: "step" }, h("span", { class: "spinner" }), h("span", { text }));
    this.steps.append(step);
    return step;
  }

  finish(step, suffix = "") {
    step.classList.add("done");
    step.firstChild.replaceWith(icon("check"));
    if (suffix) step.append(h("span", { class: "muted", text: ` · ${suffix}` }));
  }

  analysis(data) {
    this.finish(this.thinking, data.needs_retrieval ? "research needed" : "answering directly");
  }

  toolStart(data) {
    const key = Object.keys(TOOL_SUBJECT).find((k) => data.args?.[k] != null);
    const subject = key ? ` ${TOOL_SUBJECT[key](data.args[key])}` : "";
    this.tools.set(data.id, this.addStep(`${TOOL_LABELS[data.name] || data.name}${subject}`));
  }

  toolEnd(data) {
    const step = this.tools.get(data.id);
    if (step) this.finish(step, `${data.duration_ms} ms${data.flagged ? " · injection flagged" : ""}`);
  }

  token(data) {
    this.answer.textContent += data.text;
    this.answer.scrollIntoView({ block: "end" });
  }

  reset() {
    this.answer.textContent = "";
  }
}

async function sendMessage(text, retryAttachments = null) {
  if (state.busy || state.attachments.some((a) => a.status === "uploading")) return;
  const attached = retryAttachments ?? state.attachments.filter((a) => a.status === "ready").map((a) => a.name);
  text = text.trim() || (attached.length ? defaultPrompt(attached) : "");
  if (!text) return;
  dom.input.value = "";
  resizeInput();
  state.attachments = state.attachments.filter((a) => a.status !== "ready");
  renderAttachments();
  addUserMessage(text, attached);

  const pending = addMessage("assistant");
  const bubble = pending.querySelector(".bubble");
  const progress = new Progress(bubble);
  state.abort = new AbortController();
  setBusy(true);

  let finished = false;
  const fail = (message) => {
    finished = true;
    bubble.classList.add("error");
    bubble.replaceChildren(
      h("div", { text: message }),
      h("button", { class: "btn", type: "button", text: "Retry", onclick: () => { pending.previousElementSibling?.remove(); pending.remove(); sendMessage(text, attached); } }));
  };

  try {
    await withSession((id) => streamChat(id, text, attached, {
      analysis: (data) => progress.analysis(data),
      tool_start: (data) => progress.toolStart(data),
      tool_end: (data) => progress.toolEnd(data),
      token: (data) => progress.token(data),
      reset: () => progress.reset(),
      done: (turn) => {
        finished = true;
        bubble.replaceChildren(...renderAssistant(turn));
        setUsage(state.tokensUsed + turn.usage.input_tokens + turn.usage.output_tokens);
      },
      error: (data) => fail(data.message),
    }, state.abort.signal));
    if (!finished) fail("The connection closed before the answer finished.");
  } catch (error) {
    if (error.name === "AbortError") {
      bubble.replaceChildren(h("div", { class: "muted", text: "Stopped. Nothing from this answer was saved." }));
    } else {
      fail(error.message);
    }
  } finally {
    state.abort = null;
    setBusy(false);
    dom.input.focus();
  }
}

async function newChat() {
  if (state.busy) return;
  try {
    await withSession((id) => request(`/sessions/${id}/reset`, { method: "POST" }));
  } catch (error) {
    showToast(error.message, true);
    return;
  }
  dom.messages.querySelectorAll(".msg").forEach((m) => m.remove());
  dom.empty.hidden = false;
  closeSidebar();
  dom.input.focus();
}

function resizeInput() {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 200)}px`;
}

// ---------- Documents ----------

function renderDocuments(documents, uploading = []) {
  const items = documents.map((doc) =>
    h("li", { class: "doc" },
      icon("file"),
      h("div", { class: "doc-meta" },
        h("div", { class: "doc-name", title: doc.name, text: doc.name }),
        h("div", { class: "doc-sub", text: `${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"} indexed${doc.ocr ? " · via OCR" : ""}` })),
      h("button", {
        class: "icon-btn", type: "button", "aria-label": `Remove ${doc.name}`, title: "Remove",
        onclick: () => removeDocument(doc.name),
      }, icon("close"))));
  const pending = uploading.map((name) =>
    h("li", { class: "doc uploading" }, icon("file"),
      h("div", { class: "doc-meta" }, h("div", { class: "doc-name", text: name }), h("div", { class: "doc-sub", text: "Indexing" }))));
  dom.docList.replaceChildren(...items, ...pending);
  dom.docsEmpty.hidden = items.length + pending.length > 0;
  state.documents = documents;
}

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

// ---------- Attachments (ChatGPT-style chips in the composer) ----------

function fileKind(name) {
  const ext = extensionOf(name);
  if (IMAGE_TYPES.includes(ext)) return { label: "IMG", kind: "img" };
  return { label: ext.slice(1).toUpperCase() || "FILE", kind: ext.slice(1) || "file" };
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// A static chip for sent messages (and history restored from the server).
function fileChip(name, subtitle = "") {
  const { label, kind } = fileKind(name);
  return h("div", { class: "file-chip", title: name },
    h("div", { class: `file-tile kind-${kind}`, text: label }),
    h("div", { class: "file-meta" },
      h("div", { class: "file-name", text: name }),
      subtitle ? h("div", { class: "file-sub", text: subtitle }) : null));
}

function attachFiles(files) {
  for (const file of files) {
    if (!SUPPORTED.includes(extensionOf(file.name))) {
      showToast(`${file.name}: only PDF, TXT, MD and image files are supported.`, true);
      continue;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      showToast(`${file.name} is larger than 20 MB.`, true);
      continue;
    }
    // Same name again replaces the earlier attachment, matching the server's behaviour.
    state.attachments.filter((a) => a.name === file.name).forEach((a) => dropAttachment(a, false));
    const attachment = { name: file.name, file, status: "uploading", progress: 0 };
    state.attachments.push(attachment);
    renderAttachments();
    uploadAttachment(attachment);
  }
  dom.input.focus();
}

async function uploadAttachment(attachment) {
  try {
    const { documents } = await withSession((id) => uploadWithProgress(id, attachment, (fraction) => {
      attachment.progress = fraction;
      renderAttachments();
    }));
    attachment.status = "ready";
    const info = documents.find((d) => d.name === attachment.name);
    attachment.chunks = info ? info.chunks : 0;
    attachment.ocr = Boolean(info?.ocr);
    renderDocuments(documents);
  } catch (error) {
    if (error.name === "AbortError") return;
    attachment.status = "error";
    attachment.error = error.message;
  }
  renderAttachments();
}

// fetch() can't report upload progress, so uploads use XMLHttpRequest.
function uploadWithProgress(sessionId, attachment, onProgress) {
  return new Promise((resolve, reject) => {
    const send = () => {
      const xhr = new XMLHttpRequest();
      attachment.xhr = xhr;
      xhr.open("POST", `/api/sessions/${sessionId}/documents`);
      const key = store.get("apiKey");
      if (key) xhr.setRequestHeader("Authorization", `Bearer ${key}`);
      xhr.upload.onprogress = (event) => event.lengthComputable && onProgress(event.loaded / event.total);
      xhr.onload = async () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
        if (xhr.status === 401) {
          await askForApiKey(key ? "That key was rejected. Try again." : "");
          send();
        } else if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body);
        } else {
          reject(new ApiError(describeDetail(body.detail) || `Upload failed (${xhr.status})`, xhr.status));
        }
      };
      xhr.onerror = () => reject(new ApiError("Can't reach the server. Is it still running?", 0));
      xhr.onabort = () => reject(Object.assign(new Error("Upload cancelled"), { name: "AbortError" }));
      const form = new FormData();
      form.append("file", attachment.file);
      xhr.send(form);
    };
    send();
  });
}

async function dropAttachment(attachment, deleteFromServer = true) {
  state.attachments = state.attachments.filter((a) => a !== attachment);
  if (attachment.status === "uploading") attachment.xhr?.abort();
  renderAttachments();
  // Removing an attachment before sending takes it out of the knowledge base too.
  if (deleteFromServer && attachment.status === "ready") await removeDocument(attachment.name, true);
}

function renderAttachments() {
  const chips = state.attachments.map((a) => {
    const { label, kind } = fileKind(a.name);
    const tile = h("div", { class: `file-tile kind-${kind}` }, h("span", { text: label }));
    let subtitle;
    if (a.status === "uploading") {
      const pct = Math.round(a.progress * 100);
      subtitle = pct < 100 ? `Uploading ${pct}%` : "Reading and indexing…";
      const ring = h("div", { class: "file-ring" });
      ring.style.setProperty("--p", pct < 100 ? pct : 100);
      ring.classList.toggle("indeterminate", pct >= 100);
      tile.append(ring);
    } else if (a.status === "ready") {
      subtitle = `${formatSize(a.file.size)} · ${a.chunks} chunk${a.chunks === 1 ? "" : "s"}${a.ocr ? " · OCR" : ""}`;
    } else {
      subtitle = a.error || "Upload failed";
    }
    return h("div", { class: `file-chip removable status-${a.status}`, title: a.error || a.name },
      tile,
      h("div", { class: "file-meta" },
        h("div", { class: "file-name", text: a.name }),
        h("div", { class: "file-sub", text: subtitle })),
      h("button", {
        class: "file-remove", type: "button", "aria-label": `Remove ${a.name}`, title: "Remove",
        onclick: () => dropAttachment(a),
      }, icon("close")));
  });
  dom.attachments.replaceChildren(...chips);
  dom.attachments.hidden = chips.length === 0;
  updateSendState();
}

function updateSendState() {
  if (state.busy) {
    dom.send.disabled = false;  // it's the stop button while busy
    return;
  }
  const uploading = state.attachments.some((a) => a.status === "uploading");
  const hasContent = dom.input.value.trim() || state.attachments.some((a) => a.status === "ready");
  dom.send.disabled = uploading || !hasContent;
  dom.send.title = uploading ? "Waiting for uploads to finish" : "";
}

async function removeDocument(name, quiet = false) {
  try {
    const { documents } = await withSession((id) =>
      request(`/sessions/${id}/documents/${encodeURIComponent(name)}`, { method: "DELETE" }));
    renderDocuments(documents);
  } catch (error) {
    if (!(quiet && error.status === 404)) showToast(error.message, true);
  }
  // Keep the composer in sync if the file was removed from the sidebar.
  state.attachments.filter((a) => a.name === name && a.status === "ready").forEach((a) => {
    state.attachments = state.attachments.filter((x) => x !== a);
  });
  renderAttachments();
}

// ---------- UI chrome ----------

let toastTimer;
function showToast(message, isError = false) {
  dom.toast.textContent = message;
  dom.toast.classList.toggle("error", isError);
  dom.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { dom.toast.hidden = true; }, isError ? 6000 : 3000);
}

function openSidebar() {
  dom.sidebar.classList.add("open");
  dom.scrim.hidden = false;
}

function closeSidebar() {
  dom.sidebar.classList.remove("open");
  dom.scrim.hidden = true;
}

function setupDragAndDrop() {
  let depth = 0;
  const hasFiles = (event) => event.dataTransfer && [...event.dataTransfer.types].includes("Files");

  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth += 1;
    dom.overlay.hidden = false;
  });
  window.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) dom.overlay.hidden = true;
  });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    dom.overlay.hidden = true;
    attachFiles([...event.dataTransfer.files]);
  });
}

function bindEvents() {
  dom.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy) state.abort?.abort();
    else sendMessage(dom.input.value);
  });
  dom.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage(dom.input.value);
    }
  });
  dom.input.addEventListener("input", () => {
    resizeInput();
    updateSendState();
  });

  dom.attach.addEventListener("click", () => dom.fileInput.click());
  dom.fileInput.addEventListener("change", () => {
    attachFiles([...dom.fileInput.files]);
    dom.fileInput.value = "";
  });

  dom.newChat.addEventListener("click", newChat);
  dom.menu.addEventListener("click", openSidebar);
  dom.scrim.addEventListener("click", closeSidebar);
  document.querySelectorAll("#suggestions button").forEach((button) =>
    button.addEventListener("click", () => sendMessage(button.textContent)));
  // The key prompt must be answered; Escape would leave a request hanging.
  dom.authDialog.addEventListener("cancel", (event) => event.preventDefault());

  setupDragAndDrop();
}

function init() {
  bindEvents();
  renderDocuments([]);
  setBusy(false);
  state.ready = restoreSession().catch((error) => showToast(error.message, true));
  dom.input.focus();
}

init();
