import http from "node:http";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai";

const env = process.env;
const host = env.HOST || "127.0.0.1";
const port = Number(env.PORT || 8787);
const localProviderKey = env.LOCAL_PROVIDER_KEY || "";
const defaultModel = env.DEFAULT_MODEL || "gemini-flash-latest";
const forceDefaultModel = String(env.FORCE_DEFAULT_MODEL || "false").toLowerCase() === "true";
const defaultReasoningEffort = (env.REASONING_EFFORT || "").trim().toLowerCase();
const forceReasoningEffort = String(env.FORCE_REASONING_EFFORT || "false").toLowerCase() === "true";
const maxAttemptsPerRequest = Math.max(1, Number(env.MAX_ATTEMPTS_PER_REQUEST || 3));
const keyCooldownMs = Math.max(1000, Number(env.KEY_COOLDOWN_MS || 60_000));
const upstreamTimeoutMs = Math.max(10_000, Number(env.UPSTREAM_TIMEOUT_MS || 300_000));
const thoughtSignatureDummyFallback = String(env.THOUGHT_SIGNATURE_DUMMY_FALLBACK || "true").toLowerCase() === "true";
const thoughtSignatureCacheTtlMs = Math.max(1000, Number(env.THOUGHT_SIGNATURE_CACHE_TTL_MS || 1_800_000));
const thoughtSignatureCacheMaxEntries = Math.max(1, Number(env.THOUGHT_SIGNATURE_CACHE_MAX_ENTRIES || 10_000));

const keys = (env.GEMINI_API_KEYS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

if (keys.length === 0) {
  console.error("GEMINI_API_KEYS is empty. Set it to a comma-separated list of Gemini API keys.");
  process.exit(1);
}

function createKeyState(inputKeys) {
  return inputKeys.map((key, index) => ({
    key,
    index,
    cooldownUntil: 0,
    disabled: false,
    lastError: null,
  }));
}

const keyState = createKeyState(keys);

let cursor = 0;
const serverStartedAtMs = Date.now();

function createThoughtSignatureCache({ ttlMs, maxEntries }) {
  const entries = new Map();

  function evictExpired(atMs) {
    for (const [id, item] of entries) {
      if (item.expiresAt <= atMs) entries.delete(id);
    }
  }

  return {
    get size() {
      return entries.size;
    },
    clear() {
      entries.clear();
    },
    get(id, atMs = Date.now()) {
      evictExpired(atMs);
      const item = entries.get(id);
      if (!item) return null;
      if (item.expiresAt <= atMs) {
        entries.delete(id);
        return null;
      }
      return item.signature;
    },
    set(id, signature, atMs = Date.now()) {
      evictExpired(atMs);
      entries.set(id, { signature, expiresAt: atMs + ttlMs });

      if (entries.size <= maxEntries) return;

      let excess = entries.size - maxEntries;
      for (const key of entries.keys()) {
        entries.delete(key);
        excess -= 1;
        if (excess <= 0) break;
      }
    },
  };
}

const thoughtSignatureByToolCallId = createThoughtSignatureCache({
  ttlMs: thoughtSignatureCacheTtlMs,
  maxEntries: thoughtSignatureCacheMaxEntries,
});

const DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator";

function maskKey(key) {
  if (key.length <= 10) return "***";
  return `${key.slice(0, 6)}…${key.slice(-4)}`;
}

function now() {
  return Date.now();
}

function getNextKey() {
  const t = now();
  for (let i = 0; i < keyState.length; i++) {
    const idx = (cursor + i) % keyState.length;
    const state = keyState[idx];
    if (!state.disabled && state.cooldownUntil <= t) {
      cursor = (idx + 1) % keyState.length;
      return state;
    }
  }
  return null;
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 100 * 1024 * 1024) {
        reject(new Error("Request body is too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        reject(new Error(`Invalid JSON: ${e.message}`));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, status, value) {
  const data = JSON.stringify(value, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(data),
  });
  res.end(data);
}

function isAuthorized(req) {
  if (!localProviderKey) return true;
  const auth = req.headers.authorization || "";
  // Soft auth: treat missing or empty-token Bearer as "no key provided".
  // VS Code Copilot Chat 0.48.x's Ollama BYOK sends `Authorization: Bearer` with
  // an empty token because that provider has no API key field. Real Ollama
  // clients also typically don't authenticate.
  if (!auth) return true;
  const token = auth.replace(/^Bearer\s*/i, "").trim();
  if (!token) return true;
  return auth === `Bearer ${localProviderKey}`;
}

function retryAfterMs(headers) {
  const raw = headers.get("retry-after");
  if (!raw) return keyCooldownMs;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return Math.max(keyCooldownMs, seconds * 1000);
  const dateMs = Date.parse(raw);
  if (Number.isFinite(dateMs)) return Math.max(keyCooldownMs, dateMs - now());
  return keyCooldownMs;
}

function shouldFailover(status, errorPayload) {
  // Failover only on key/provider/billing/quota/transient errors.
  // Do not retry normal client errors like bad request, safety block, unsupported model, etc.
  if ([401, 402, 403, 408, 409, 429, 500, 502, 503, 504].includes(status)) return true;

  const text = JSON.stringify(errorPayload || {}).toLowerCase();
  return (
    text.includes("api key") ||
    text.includes("quota") ||
    text.includes("rate limit") ||
    text.includes("billing") ||
    text.includes("overloaded")
  );
}

async function parseErrorPayload(response) {
  const contentType = response.headers.get("content-type") || "";
  try {
    if (contentType.includes("application/json")) return await response.json();
    return { error: { message: await response.text() } };
  } catch {
    return { error: { message: `HTTP ${response.status}` } };
  }
}

function stringifyInstruction(value) {
  if (value == null) return "";
  if (typeof value === "string") return value.trim();
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item : item?.text || item?.content || JSON.stringify(item)))
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (typeof value === "object") return value.text || value.content || JSON.stringify(value);
  return String(value);
}

function normalizeContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;

  // Keep OpenAI-style multimodal parts, but remove extra fields that Gemini may reject.
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return { type: "text", text: part };
        if (!part || typeof part !== "object") return null;

        if (part.type === "text") return { type: "text", text: part.text || "" };
        if (part.type === "image_url") return { type: "image_url", image_url: part.image_url };

        // Some clients emit { type: "input_text" } / { type: "input_image" }.
        // Convert the text form and drop unsupported binary/file parts for chat/completions.
        if (part.type === "input_text") return { type: "text", text: part.text || "" };
        if (part.type === "input_image" && part.image_url) {
          return { type: "image_url", image_url: part.image_url };
        }

        return null;
      })
      .filter(Boolean);
  }

  return String(content);
}

function normalizeMessages(messages, instructions) {
  const out = [];
  const instructionText = stringifyInstruction(instructions);

  if (instructionText) {
    out.push({ role: "system", content: instructionText });
  }

  for (const message of Array.isArray(messages) ? messages : []) {
    if (!message || typeof message !== "object") continue;

    const next = {
      role: message.role === "developer" ? "system" : message.role,
      content: normalizeContent(message.content),
    };

    if (message.name) next.name = message.name;
    if (message.tool_call_id) next.tool_call_id = message.tool_call_id;
    if (Array.isArray(message.tool_calls)) {
      next.tool_calls = repairToolCalls(message.tool_calls);
    }

    out.push(next);
  }

  return out;
}

function readThoughtSignatureFromToolCall(toolCall) {
  return toolCall?.extra_content?.google?.thought_signature;
}

function cacheThoughtSignature(toolCall) {
  const toolCallId = toolCall?.id;
  const signature = readThoughtSignatureFromToolCall(toolCall);
  if (!toolCallId || typeof signature !== "string" || !signature) return;
  thoughtSignatureByToolCallId.set(toolCallId, signature);
}

function withThoughtSignature(toolCall, signature) {
  const extraContent = toolCall.extra_content && typeof toolCall.extra_content === "object"
    ? toolCall.extra_content
    : {};
  const google = extraContent.google && typeof extraContent.google === "object"
    ? extraContent.google
    : {};

  return {
    ...toolCall,
    extra_content: {
      ...extraContent,
      google: {
        ...google,
        thought_signature: signature,
      },
    },
  };
}

function repairToolCalls(toolCalls) {
  return toolCalls.map((toolCall, index) => {
    if (!toolCall || typeof toolCall !== "object") return toolCall;

    cacheThoughtSignature(toolCall);

    // Gemini expects thought_signature on the first tool call of the message.
    if (index !== 0) return toolCall;

    const existing = readThoughtSignatureFromToolCall(toolCall);
    if (typeof existing === "string" && existing) return toolCall;

    const cached = toolCall.id ? thoughtSignatureByToolCallId.get(toolCall.id) : null;
    if (cached) return withThoughtSignature(toolCall, cached);

    if (thoughtSignatureDummyFallback) {
      return withThoughtSignature(toolCall, DUMMY_THOUGHT_SIGNATURE);
    }

    return toolCall;
  });
}

function cacheToolCallsFromCompletionPayload(payload) {
  for (const choice of Array.isArray(payload?.choices) ? payload.choices : []) {
    const messageCalls = choice?.message?.tool_calls;
    if (Array.isArray(messageCalls)) {
      for (const toolCall of messageCalls) cacheThoughtSignature(toolCall);
    }

    const deltaCalls = choice?.delta?.tool_calls;
    if (Array.isArray(deltaCalls)) {
      for (const toolCall of deltaCalls) cacheThoughtSignature(toolCall);
    }
  }
}

const ALLOWED_REASONING_EFFORTS = new Set(["none", "minimal", "low", "medium", "high"]);

const MODEL_BASES = [
  "gemini-flash-latest",
  "gemini-3-flash-preview",
  "gemini-3.1-flash-lite",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
];

function normalizeRequestedModel(rawModel) {
  // Strip Ollama-style ":latest" / ":tag" suffix — Copilot Chat's Ollama BYOK
  // sends model names like "gemini-flash-latest-high:latest" to /v1/chat/completions.
  let model = stripModelTag(rawModel || defaultModel);
  let effortFromModel = undefined;

  // Optional convenience model ids for clients that cannot display variants well:
  //   gemini-flash-latest-minimal -> model=gemini-flash-latest, reasoning_effort=minimal
  //   gemini-2.5-flash-none      -> model=gemini-2.5-flash, reasoning_effort=none
  for (const base of MODEL_BASES) {
    for (const effort of ALLOWED_REASONING_EFFORTS) {
      const suffix = `-${effort}`;
      if (model === `${base}${suffix}`) {
        return { model: base, effortFromModel: effort };
      }
    }
  }

  return { model, effortFromModel };
}

function normalizeReasoningEffort(payload, effortFromModel) {
  // Gemini OpenAI-compatible chat/completions accepts OpenAI-style
  // `reasoning_effort`: none|minimal|low|medium|high.
  // OpenCode config uses camelCase `reasoningEffort`; accept both.
  const allowed = ALLOWED_REASONING_EFFORTS;

  let incoming = payload.reasoning_effort || payload.reasoningEffort || effortFromModel;

  if (!incoming && payload.reasoning && typeof payload.reasoning === "object") {
    incoming = payload.reasoning.effort || payload.reasoning.reasoningEffort;
    if (!incoming && payload.reasoning.enabled === true) incoming = "medium";
  }

  if (typeof incoming === "string") incoming = incoming.trim().toLowerCase();

  let finalEffort = incoming;
  if (defaultReasoningEffort && (forceReasoningEffort || !finalEffort)) {
    finalEffort = defaultReasoningEffort;
  }

  if (!finalEffort) return undefined;
  if (!allowed.has(finalEffort)) {
    console.warn(`[reasoning] ignored unsupported reasoning effort: ${finalEffort}`);
    return undefined;
  }

  return finalEffort;
}

function normalizePayload(payload) {
  // Gemini's OpenAI-compatible chat endpoint is stricter than OpenAI's SDK types.
  // OpenCode / AI SDK may send top-level fields like `instructions`; Gemini rejects
  // them with: "Unknown name \"instructions\": Cannot find field.".
  // So we convert what is useful and forward only common chat.completions fields.
  const next = {};

  const requested = normalizeRequestedModel(payload.model);
  const fallback = normalizeRequestedModel(defaultModel);

  next.model = requested.model;
  let effortFromModel = requested.effortFromModel;

  if (forceDefaultModel || !next.model || next.model === "gemini-flash") {
    next.model = fallback.model;
    // Preserve effort selected by a convenience model id unless the env explicitly forces it.
    effortFromModel = effortFromModel || fallback.effortFromModel;
  }

  next.messages = normalizeMessages(payload.messages, payload.instructions);

  const reasoningEffort = normalizeReasoningEffort(payload, effortFromModel);
  if (reasoningEffort) next.reasoning_effort = reasoningEffort;

  const passthrough = [
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "tools",
    "tool_choice",
    "response_format",
    "seed",
    "stream_options",
  ];

  for (const key of passthrough) {
    if (payload[key] !== undefined) next[key] = payload[key];
  }

  return next;
}

async function upstreamChatCompletion(payload, streamMode) {
  const attemptsLimit = Math.min(maxAttemptsPerRequest, keyState.length);
  const errors = [];

  for (let attempt = 1; attempt <= attemptsLimit; attempt++) {
    let key = getNextKey();

    if (!key) {
      const soonest = Math.min(...keyState.filter((k) => !k.disabled).map((k) => k.cooldownUntil));
      const wait = Number.isFinite(soonest) ? Math.max(250, soonest - now()) : 1000;
      await delay(Math.min(wait, 2000));
      key = getNextKey();
      if (!key) break;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), upstreamTimeoutMs);

    try {
      const response = await fetch(`${GEMINI_OPENAI_BASE}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "authorization": `Bearer ${key.key}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.ok) {
        console.log(`[ok] key=${key.index + 1}/${keyState.length} ${maskKey(key.key)} model=${payload.model} stream=${streamMode}`);
        return { response, key };
      }

      const errorPayload = await parseErrorPayload(response);
      const message = errorPayload?.error?.message || `HTTP ${response.status}`;
      key.lastError = `${response.status}: ${message}`;
      errors.push({ key: key.index + 1, status: response.status, message });

      console.warn(`[fail] key=${key.index + 1}/${keyState.length} status=${response.status} ${message}`);

      if (!shouldFailover(response.status, errorPayload)) {
        return { response: null, errorStatus: response.status, errorPayload };
      }

      applyKeyFailureState(key, response.status, response.headers, keyCooldownMs);
    } catch (e) {
      clearTimeout(timeout);
      key.lastError = e.message;
      key.cooldownUntil = now() + keyCooldownMs;
      errors.push({ key: key.index + 1, status: "network", message: e.message });
      console.warn(`[network] key=${key.index + 1}/${keyState.length} ${e.message}`);
    }
  }

  return {
    response: null,
    errorStatus: 503,
    errorPayload: {
      error: {
        message: "All Gemini keys failed or are cooling down",
        attempts: errors,
      },
    },
  };
}

function applyKeyFailureState(key, status, headers, defaultCooldownMs) {
  const totalKeysForLog = Math.max(keyState.length, key.index + 1);

  if (status === 401) {
    key.disabled = true;
    console.warn(`[disabled] key=${key.index + 1}/${totalKeysForLog} ${maskKey(key.key)}`);
    return;
  }

  const cooldownMs = retryAfterMs(headers) || defaultCooldownMs;
  key.cooldownUntil = now() + Math.max(defaultCooldownMs, cooldownMs);
  console.warn(`[cooldown] key=${key.index + 1}/${totalKeysForLog} until=${new Date(key.cooldownUntil).toISOString()}`);
}

async function handleChatCompletions(req, res) {
  if (!isAuthorized(req)) {
    return sendJson(res, 401, { error: { message: "Unauthorized local provider key" } });
  }

  let payload;
  try {
    payload = normalizePayload(await readJsonBody(req));
  } catch (e) {
    return sendJson(res, 400, { error: { message: e.message } });
  }

  const streamMode = payload.stream === true;
  const result = await upstreamChatCompletion(payload, streamMode);

  if (!result.response) {
    return sendJson(res, result.errorStatus || 503, result.errorPayload);
  }

  const upstream = result.response;
  const headers = Object.fromEntries(upstream.headers.entries());

  // Avoid forwarding hop-by-hop/problematic headers.
  delete headers["content-encoding"];
  delete headers["content-length"];
  delete headers["transfer-encoding"];
  delete headers["connection"];

  const contentType = headers["content-type"] || "";
  const isSse = contentType.includes("text/event-stream");

  if (!streamMode && !isSse) {
    const text = await upstream.text();
    try {
      const parsed = text ? JSON.parse(text) : {};
      cacheToolCallsFromCompletionPayload(parsed);
    } catch {
      // Keep passthrough behavior for non-JSON payloads.
    }
    headers["content-length"] = Buffer.byteLength(text);
    res.writeHead(upstream.status, headers);
    res.end(text);
    return;
  }

  res.writeHead(upstream.status, headers);

  if (!upstream.body) {
    res.end();
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let sseBuffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));

      if (isSse) {
        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split("\n");
        sseBuffer = lines.pop() || "";
        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          try {
            const parsed = JSON.parse(data);
            cacheToolCallsFromCompletionPayload(parsed);
          } catch {
            // Ignore partial/non-JSON events.
          }
        }
      }
    }
  } catch (e) {
    console.warn(`[stream] client/upstream stream ended: ${e.message}`);
  } finally {
    res.end();
  }
}

function handleModels(req, res) {
  if (!isAuthorized(req)) {
    return sendJson(res, 401, { error: { message: "Unauthorized local provider key" } });
  }

  sendJson(res, 200, {
    object: "list",
    data: [
      { id: defaultModel, object: "model", owned_by: "google" },
      { id: "gemini-flash-latest", object: "model", owned_by: "google" },
      { id: "gemini-flash-latest-minimal", object: "model", owned_by: "google" },
      { id: "gemini-flash-latest-low", object: "model", owned_by: "google" },
      { id: "gemini-flash-latest-medium", object: "model", owned_by: "google" },
      { id: "gemini-flash-latest-high", object: "model", owned_by: "google" },
      { id: "gemini-3-flash-preview", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-none", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-minimal", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-low", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-medium", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-high", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-lite", object: "model", owned_by: "google" },
      { id: "gemini-2.5-flash-lite-none", object: "model", owned_by: "google" }
    ],
  });
}

function handleHealth(req, res) {
  sendJson(res, 200, {
    ok: true,
    provider: "gemini-openai-failover",
    uptimeSec: Math.floor((Date.now() - serverStartedAtMs) / 1000),
  });
}

// =============================================================================
// Ollama-compatible API surface
//
// GitHub Copilot Chat's "Ollama" BYOK provider verifies the server with native
// Ollama endpoints (`/api/version`, `/api/tags`, `/api/show`) and sends chat
// requests in Ollama format on `/api/chat` — not on `/v1/chat/completions`.
// These handlers translate Ollama requests/responses to the existing failover
// pipeline that targets Gemini's OpenAI-compatible endpoint.
// =============================================================================

const OLLAMA_FAKE_VERSION = "0.11.0";
const OLLAMA_MODEL_NAMES = [
  "gemini-flash-latest",
  "gemini-flash-latest-minimal",
  "gemini-flash-latest-low",
  "gemini-flash-latest-medium",
  "gemini-flash-latest-high",
  "gemini-3-flash-preview",
  "gemini-2.5-flash",
  "gemini-2.5-flash-minimal",
  "gemini-2.5-flash-low",
  "gemini-2.5-flash-medium",
  "gemini-2.5-flash-high",
  "gemini-2.5-flash-lite",
];

function stripModelTag(name) {
  if (!name) return name;
  return String(name).replace(/:[^:]*$/, "");
}

function withLatestTag(name) {
  return /:[^:]+$/.test(name) ? name : `${name}:latest`;
}

function ollamaModelEntry(name) {
  return {
    name: withLatestTag(name),
    model: withLatestTag(name),
    modified_at: new Date(serverStartedAtMs).toISOString(),
    size: 0,
    digest: `sha256:${Buffer.from(name).toString("hex").padEnd(64, "0").slice(0, 64)}`,
    details: {
      parent_model: "",
      format: "gguf",
      family: "gemini",
      families: ["gemini"],
      parameter_size: "unknown",
      quantization_level: "none",
    },
  };
}

function handleOllamaVersion(_req, res) {
  sendJson(res, 200, { version: OLLAMA_FAKE_VERSION });
}

function handleOllamaTags(req, res) {
  if (!isAuthorized(req)) {
    return sendJson(res, 401, { error: "unauthorized" });
  }
  sendJson(res, 200, {
    models: OLLAMA_MODEL_NAMES.map(ollamaModelEntry),
  });
}

async function handleOllamaShow(req, res) {
  if (!isAuthorized(req)) {
    return sendJson(res, 401, { error: "unauthorized" });
  }
  let body;
  try {
    body = await readJsonBody(req);
  } catch (e) {
    return sendJson(res, 400, { error: e.message });
  }
  const name = stripModelTag(body?.model || body?.name || defaultModel);
  sendJson(res, 200, {
    modelfile: "",
    parameters: "",
    template: "",
    details: ollamaModelEntry(name).details,
    model_info: {
      "general.architecture": "gemini",
      "general.parameter_count": 0,
      "gemini.context_length": 1048576,
    },
    capabilities: ["completion", "tools"],
  });
}

function ollamaMessagesToOpenAI(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map((m) => {
    if (!m || typeof m !== "object") return m;

    const out = { role: m.role === "developer" ? "system" : m.role };

    const hasImages = Array.isArray(m.images) && m.images.length > 0;
    if (hasImages) {
      const parts = [];
      if (m.content != null && String(m.content).length > 0) {
        parts.push({ type: "text", text: String(m.content) });
      }
      for (const img of m.images) {
        if (typeof img !== "string" || !img) continue;
        const url = img.startsWith("data:") ? img : `data:image/png;base64,${img}`;
        parts.push({ type: "image_url", image_url: { url } });
      }
      out.content = parts;
    } else {
      out.content = m.content == null ? "" : String(m.content);
    }

    if (m.name) out.name = m.name;
    if (m.tool_call_id) out.tool_call_id = m.tool_call_id;

    if (Array.isArray(m.tool_calls)) {
      out.tool_calls = m.tool_calls.map((tc, i) => {
        const fn = (tc && tc.function) || {};
        let argsStr = "{}";
        try {
          argsStr = typeof fn.arguments === "string"
            ? fn.arguments
            : JSON.stringify(fn.arguments ?? {});
        } catch {
          argsStr = "{}";
        }
        return {
          id: tc.id || `ollama_tc_${Date.now()}_${i}`,
          type: "function",
          function: { name: fn.name || "", arguments: argsStr },
        };
      });
    }

    return out;
  });
}

function ollamaToolsToOpenAI(tools) {
  if (!Array.isArray(tools)) return undefined;
  return tools.map((t) => {
    if (t && t.type === "function" && t.function) return t;
    if (t && t.function) return { type: "function", function: t.function };
    return t;
  });
}

function ollamaOptionsToOpenAI(options) {
  const out = {};
  if (!options || typeof options !== "object") return out;
  if (typeof options.temperature === "number") out.temperature = options.temperature;
  if (typeof options.top_p === "number") out.top_p = options.top_p;
  if (typeof options.num_predict === "number") out.max_tokens = options.num_predict;
  if (typeof options.seed === "number") out.seed = options.seed;
  if (options.stop !== undefined) out.stop = options.stop;
  return out;
}

function openAiToolCallsToOllama(toolCalls) {
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return undefined;
  return toolCalls.map((tc) => {
    const fn = (tc && tc.function) || {};
    let argsObj = {};
    try {
      argsObj = typeof fn.arguments === "string"
        ? (fn.arguments ? JSON.parse(fn.arguments) : {})
        : (fn.arguments || {});
    } catch {
      argsObj = {};
    }
    return { function: { name: fn.name || "", arguments: argsObj } };
  });
}

async function handleOllamaChat(req, res) {
  if (!isAuthorized(req)) {
    return sendJson(res, 401, { error: "unauthorized" });
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (e) {
    return sendJson(res, 400, { error: e.message });
  }

  const requestedModel = String(body.model || defaultModel);
  const cleanModel = stripModelTag(requestedModel);
  const streamMode = body.stream !== false; // Ollama default is stream=true

  const openAiPayload = normalizePayload({
    model: cleanModel,
    messages: ollamaMessagesToOpenAI(body.messages),
    tools: ollamaToolsToOpenAI(body.tools),
    stream: streamMode,
    ...ollamaOptionsToOpenAI(body.options),
  });

  const result = await upstreamChatCompletion(openAiPayload, streamMode);

  if (!result.response) {
    const status = result.errorStatus || 503;
    const message = result.errorPayload?.error?.message || `HTTP ${status}`;
    return sendJson(res, status, { error: message });
  }

  const upstream = result.response;
  const contentType = upstream.headers.get("content-type") || "";
  const isSse = contentType.includes("text/event-stream");

  if (!streamMode && !isSse) {
    const text = await upstream.text();
    let parsed = {};
    try {
      parsed = text ? JSON.parse(text) : {};
      cacheToolCallsFromCompletionPayload(parsed);
    } catch {
      // fallthrough: empty parsed
    }

    const choice = (parsed.choices && parsed.choices[0]) || {};
    const msg = choice.message || {};
    const usage = parsed.usage || {};

    const message = {
      role: msg.role || "assistant",
      content: typeof msg.content === "string" ? msg.content : "",
    };
    const ollamaToolCalls = openAiToolCallsToOllama(msg.tool_calls);
    if (ollamaToolCalls) message.tool_calls = ollamaToolCalls;

    return sendJson(res, upstream.status, {
      model: requestedModel,
      created_at: new Date().toISOString(),
      message,
      done: true,
      done_reason: choice.finish_reason || "stop",
      total_duration: 0,
      load_duration: 0,
      prompt_eval_count: usage.prompt_tokens || 0,
      prompt_eval_duration: 0,
      eval_count: usage.completion_tokens || 0,
      eval_duration: 0,
    });
  }

  res.writeHead(upstream.status, {
    "content-type": "application/x-ndjson; charset=utf-8",
    "cache-control": "no-cache",
  });

  if (!upstream.body) {
    res.write(JSON.stringify({
      model: requestedModel,
      created_at: new Date().toISOString(),
      message: { role: "assistant", content: "" },
      done: true,
      done_reason: "stop",
    }) + "\n");
    return res.end();
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const toolCallAccum = new Map();
  let role = "assistant";
  let finishReason = null;
  let usage = null;

  const writeFrame = (obj) => {
    res.write(JSON.stringify(obj) + "\n");
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;

        let event;
        try {
          event = JSON.parse(data);
        } catch {
          continue;
        }
        cacheToolCallsFromCompletionPayload(event);

        if (event.usage) usage = event.usage;

        const choice = (event.choices && event.choices[0]) || {};
        const delta = choice.delta || {};
        if (delta.role) role = delta.role;
        if (choice.finish_reason) finishReason = choice.finish_reason;

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = typeof tc.index === "number" ? tc.index : 0;
            const acc = toolCallAccum.get(idx) || { id: "", name: "", arguments: "" };
            if (tc.id) acc.id = tc.id;
            if (tc.function && tc.function.name) acc.name = tc.function.name;
            if (tc.function && typeof tc.function.arguments === "string") {
              acc.arguments += tc.function.arguments;
            }
            toolCallAccum.set(idx, acc);
          }
        }

        if (typeof delta.content === "string" && delta.content.length > 0) {
          writeFrame({
            model: requestedModel,
            created_at: new Date().toISOString(),
            message: { role, content: delta.content },
            done: false,
          });
        }
      }
    }
  } catch (e) {
    console.warn(`[ollama-stream] ${e.message}`);
  }

  const finalMessage = { role, content: "" };
  if (toolCallAccum.size > 0) {
    finalMessage.tool_calls = Array.from(toolCallAccum.values()).map((acc) => {
      let argsObj = {};
      try {
        argsObj = acc.arguments ? JSON.parse(acc.arguments) : {};
      } catch {
        argsObj = {};
      }
      return { function: { name: acc.name, arguments: argsObj } };
    });
  }

  writeFrame({
    model: requestedModel,
    created_at: new Date().toISOString(),
    message: finalMessage,
    done: true,
    done_reason: finishReason || "stop",
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: usage?.prompt_tokens || 0,
    prompt_eval_duration: 0,
    eval_count: usage?.completion_tokens || 0,
    eval_duration: 0,
  });
  res.end();
}

export const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

    const normalizedPath = url.pathname.replace(/\/+$|^$/, "") || "/";

    if (normalizedPath !== "/health") {
      console.log(`[req] ${req.method} ${normalizedPath} ua=${req.headers["user-agent"] || "?"}`);
    }

    if (req.method === "GET" && normalizedPath === "/health") return handleHealth(req, res);

    // Some OpenAI-compatible clients append /chat/completions themselves.
    // Therefore support both baseURL styles:
    //   http://host:port/v1 -> /v1/chat/completions
    //   http://host:port    -> /chat/completions
    if (req.method === "GET" && ["/v1/models", "/models"].includes(normalizedPath)) {
      return handleModels(req, res);
    }

    if (req.method === "POST" && ["/v1/chat/completions", "/chat/completions"].includes(normalizedPath)) {
      return handleChatCompletions(req, res);
    }

    // Ollama-compatible surface (for VS Code Copilot Chat BYOK).
    if (req.method === "GET" && normalizedPath === "/api/version") {
      return handleOllamaVersion(req, res);
    }
    if (req.method === "GET" && normalizedPath === "/api/tags") {
      return handleOllamaTags(req, res);
    }
    if (req.method === "POST" && normalizedPath === "/api/show") {
      return handleOllamaShow(req, res);
    }
    if (req.method === "POST" && normalizedPath === "/api/chat") {
      return handleOllamaChat(req, res);
    }

    sendJson(res, 404, {
      error: {
        message: `Not found: ${req.method} ${url.pathname}`,
        hint: "Use baseURL http://127.0.0.1:8787/v1 or http://127.0.0.1:8787; both /v1/chat/completions and /chat/completions are supported in this build. Ollama-compatible /api/version, /api/tags, /api/show, /api/chat are also exposed."
      }
    });
  } catch (e) {
    console.error(e);
    sendJson(res, 500, { error: { message: e.message } });
  }
});

export {
  normalizeMessages,
  repairToolCalls,
  cacheToolCallsFromCompletionPayload,
  readThoughtSignatureFromToolCall,
  thoughtSignatureByToolCallId,
  createThoughtSignatureCache,
  createKeyState,
  applyKeyFailureState,
  ollamaMessagesToOpenAI,
  ollamaToolsToOpenAI,
  ollamaOptionsToOpenAI,
  openAiToolCallsToOllama,
  stripModelTag,
  OLLAMA_FAKE_VERSION,
};

function startServer() {
  server.listen(port, host, () => {
    console.log(`Gemini failover provider listening on http://${host}:${port}`);
    console.log(`Loaded ${keyState.length} Gemini API key(s). Default model: ${defaultModel}`);
    console.log(`Reasoning effort: ${defaultReasoningEffort || "client/default"}${forceReasoningEffort ? " (forced)" : ""}`);
    console.log(`Thought signature fallback: ${thoughtSignatureDummyFallback ? "dummy-enabled" : "disabled"}`);
  });
}

const isMainModule = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMainModule) {
  startServer();
}
