import { config } from "./config.js";

export function logDebugRequestShape(route, streamMode, model, upstreamModel, normalizedPayload) {
  if (!config.debugRequestShapes) return;

  const msgs = normalizedPayload.messages || [];
  const shape = {
    route,
    stream: streamMode,
    model,
    upstreamModel,
    numMessages: msgs.length,
    messages: msgs.map((m) => {
      const summary = {
        role: m.role || "?",
        contentType: typeof m.content === "string" ? "string" : "array",
        numTextParts: 0,
        numImageParts: 0,
        numFileParts: 0,
        estimatedTextLength: 0,
        attachmentMimeTypes: [],
        attachmentByteSizes: [],
      };

      if (typeof m.content === "string") {
        summary.numTextParts = m.content.length > 0 ? 1 : 0;
        summary.estimatedTextLength = m.content.length;
      } else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (!part || typeof part !== "object") continue;
          if (part.type === "text") {
            summary.numTextParts++;
            summary.estimatedTextLength += (part.text || "").length;
          } else if (part.type === "image_url") {
            summary.numImageParts++;
            const url = part.image_url?.url || "";
            const mime = extractMimeFromDataUrl(url);
            if (mime) summary.attachmentMimeTypes.push(mime);
            summary.attachmentByteSizes.push(estimateBase64Size(url));
          }
        }
      }

      return summary;
    }),
    numTools: Array.isArray(normalizedPayload.tools) ? normalizedPayload.tools.length : 0,
    hasToolChoice: normalizedPayload.tool_choice !== undefined,
  };

  console.log("[debug-request-shape]", JSON.stringify(shape));
}

export function logDebugNormalizedPayload(payload) {
  if (!config.debugNormalizedPayloads) return;

  const safe = {
    model: payload.model,
    stream: payload.stream,
    numMessages: payload.messages?.length,
    hasTools: !!payload.tools,
    hasToolChoice: payload.tool_choice !== undefined,
    reasoningEffort: payload.reasoning_effort,
    hasInstructions: false,
    messages: [],
  };

  if (Array.isArray(payload.messages)) {
    safe.messages = payload.messages.map((m) => {
      const s = { role: m.role, contentShape: typeof m.content === "string" ? "string" : "array" };
      if (m.name) s.name = m.name;
      if (m.tool_call_id) s.tool_call_id = m.tool_call_id;
      if (Array.isArray(m.content)) {
        s.contentParts = m.content.map((p) => {
          if (p?.type === "text") return { type: "text", len: (p.text || "").length };
          if (p?.type === "image_url") {
            const url = p.image_url?.url || "";
            return { type: "image_url", mime: extractMimeFromDataUrl(url), approxBytes: estimateBase64Size(url) };
          }
          return { type: p?.type || "unknown" };
        });
      }
      return s;
    });
  }

  console.log("[debug-normalized-payload]", JSON.stringify(safe));
}

export function logRequestStart(method, path, ua) {
  console.log(`[req] ${method} ${path} ua=${ua || "?"}`);
}

export function logUpstreamOk(keyIndex, totalKeys, maskedKey, model, streamMode) {
  console.log(`[ok] key=${keyIndex}/${totalKeys} ${maskedKey} model=${model} stream=${streamMode}`);
}

const UPSTREAM_ERROR_LOG_SNIPPET_CHARS = Math.max(
  1,
  Number(process.env.UPSTREAM_ERROR_LOG_SNIPPET_CHARS || 2000)
);

export function logUpstreamFail(keyIndex, totalKeys, status, message) {
  console.warn(`[fail] key=${keyIndex}/${totalKeys} status=${status} ${message}`);
}

export function logUpstreamFailDetailed(keyIndex, totalKeys, {
  status,
  model,
  kind,
  code,
  reason,
  bodySnippet,
  contentType,
  providerCode,
  providerStatus,
  requestShape,
}) {
  const parts = [`[fail] key=${keyIndex}/${totalKeys}`];
  if (status != null) parts.push(`status=${status}`);
  if (model) parts.push(`model=${model}`);
  if (kind) parts.push(`kind=${kind}`);
  if (code) parts.push(`code=${code}`);
  if (contentType) parts.push(`contentType=${contentType}`);
  if (providerCode) parts.push(`providerCode=${providerCode}`);
  if (providerStatus) parts.push(`providerStatus=${providerStatus}`);
  if (reason) parts.push(`reason=${reason}`);
  if (requestShape) {
    const shapeSummary = JSON.stringify(requestShape);
    parts.push(`shape=${shapeSummary}`);
  }
  console.warn(parts.join(" "));

  if (bodySnippet) {
    console.warn(`[fail-body] key=${keyIndex}/${totalKeys} ${bodySnippet}`);
  }
}

export function getUpstreamErrorSnippet(errorPayload) {
  if (!errorPayload) return "";
  if (typeof errorPayload === "string") {
    return String(errorPayload).slice(0, UPSTREAM_ERROR_LOG_SNIPPET_CHARS);
  }
  try {
    const safe = {
      error: {
        message: String(errorPayload?.error?.message || "").slice(0, UPSTREAM_ERROR_LOG_SNIPPET_CHARS),
        code: errorPayload?.error?.code,
        status: errorPayload?.error?.status,
        type: errorPayload?.error?.type,
      },
    };
    return JSON.stringify(safe);
  } catch {
    return String(errorPayload || "").slice(0, UPSTREAM_ERROR_LOG_SNIPPET_CHARS);
  }
}

export function logNetworkError(keyIndex, totalKeys, message) {
  console.warn(`[network] key=${keyIndex}/${totalKeys} ${message}`);
}

export function logKeyDisabled(keyIndex, totalKeys, maskedKey) {
  console.warn(`[disabled] key=${keyIndex}/${totalKeys} ${maskedKey}`);
}

export function logKeyCooldown(keyIndex, totalKeys, until) {
  console.warn(`[cooldown] key=${keyIndex}/${totalKeys} until=${new Date(until).toISOString()}`);
}

export function logStreamEnd(message) {
  console.warn(`[stream] client/upstream stream ended: ${message}`);
}

export function logOllamaStreamEnd(message) {
  console.warn(`[ollama-stream] ${message}`);
}

export function logStartup(host, port, keyCount, defaultModel, reasoningEffort, forceReasoning, thoughtSigFallback) {
  console.log(`Gemini failover provider listening on http://${host}:${port}`);
  console.log(`Loaded ${keyCount} Gemini API key(s). Default model: ${defaultModel}`);
  console.log(`Reasoning effort: ${reasoningEffort || "client/default"}${forceReasoning ? " (forced)" : ""}`);
  console.log(`Thought signature fallback: ${thoughtSigFallback ? "dummy-enabled" : "disabled"}`);
}

function extractMimeFromDataUrl(url) {
  const match = String(url || "").match(/^data:([^;]*)/);
  return match ? match[1] : null;
}

function estimateBase64Size(url) {
  const data = String(url || "");
  const b64 = data.includes("base64,") ? data.split("base64,")[1] : "";
  return Math.floor((b64.length * 3) / 4);
}
