import { createHash } from "node:crypto";

const env = process.env;

export const MAX_UPSTREAM_500_FAILOVER_ATTEMPTS = Math.max(
  1,
  Number(env.MAX_UPSTREAM_500_FAILOVER_ATTEMPTS || 2)
);
export const UPSTREAM_500_FINGERPRINT_TTL_MS = Math.max(
  1000,
  Number(env.UPSTREAM_500_FINGERPRINT_TTL_MS || 300000)
);

/**
 * Classify an upstream HTTP error into a structured error kind.
 *
 * @param {number} status - HTTP status from upstream
 * @param {object} errorPayload - Parsed error body
 * @param {{ model?: string, stream?: boolean, hasImages?: boolean, hasFiles?: boolean, hasTools?: boolean, hasToolChoice?: boolean, reasoningEffort?: string, numMessages?: number }} context - Request shape context (no secrets)
 * @returns {{ kind: string, shouldFailover: boolean, shouldCooldownKey: boolean, clientStatus: number, code: string, retryAfterMs?: number, reason: string }}
 */
export function classifyUpstreamError(status, errorPayload, context = {}) {
  const errorText = extractErrorText(errorPayload);

  if (status === 401) {
    return {
      kind: "key_auth",
      shouldFailover: true,
      shouldCooldownKey: false, // disable, don't cooldown
      clientStatus: 401,
      code: "upstream_unauthorized",
      reason: "Upstream API key invalid or revoked (401)",
    };
  }

  if (status === 402) {
    return {
      kind: "billing",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 402,
      code: "upstream_billing",
      reason: "Upstream billing required (402)",
    };
  }

  if (status === 403) {
    if (errorText.includes("quota") || errorText.includes("billing") || errorText.includes("key") || errorText.includes("api key") || errorText.includes("permission")) {
      return {
        kind: "quota",
        shouldFailover: true,
        shouldCooldownKey: true,
        clientStatus: 403,
        code: "upstream_quota_or_permission",
        reason: "Upstream quota, permission or key issue (403)",
      };
    }
    return {
      kind: "safety_or_policy",
      shouldFailover: false,
      shouldCooldownKey: false,
      clientStatus: 400,
      code: "upstream_safety_policy",
      reason: "Upstream safety/policy rejection (403)",
    };
  }

  if (status === 429) {
    return {
      kind: "rate_limit",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 429,
      code: "upstream_rate_limit",
      reason: "Upstream rate limit (429)",
    };
  }

  if ([408, 409].includes(status)) {
    return {
      kind: "transient_provider",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 502,
      code: "upstream_transient",
      reason: `Upstream transient error (${status})`,
    };
  }

  if ([502, 503, 504].includes(status)) {
    return {
      kind: "transient_provider",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 502,
      code: "upstream_transient",
      reason: `Upstream gateway/timeout error (${status})`,
    };
  }

  if (status === 500) {
    // Check suspicious unsupported combinations first
    if (looksLikelyPayloadModelError(context)) {
      return {
        kind: "payload_or_model",
        shouldFailover: false,
        shouldCooldownKey: false,
        clientStatus: 502,
        code: "upstream_model_payload_error",
        reason: "Upstream 500 likely caused by unsupported model/capability combination",
      };
    }

    const transientIndicators = [
      "overloaded",
      "temporarily unavailable",
      "try again",
      "unavailable",
      "backend error",
      "service error",
      "server error",
    ];

    const looksTransient = transientIndicators.some((ind) =>
      errorText.toLowerCase().includes(ind)
    );

    if (looksTransient) {
      return {
        kind: "transient_provider",
        shouldFailover: true,
        shouldCooldownKey: true,
        clientStatus: 502,
        code: "upstream_transient_500",
        reason: "Upstream overloaded or temporarily unavailable (500)",
      };
    }

    // Generic 500 — allow limited failover, guarded by fingerprint
    return {
      kind: "unknown_upstream",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 502,
      code: "upstream_unknown_500",
      reason: "Upstream returned 500 with no clear error classification",
    };
  }

  // For unknown status codes, check error text
  const text = errorText.toLowerCase();
  if (text.includes("api key") || text.includes("unauthorized")) {
    return {
      kind: "key_auth",
      shouldFailover: true,
      shouldCooldownKey: false,
      clientStatus: 401,
      code: "upstream_key_auth_text",
      reason: "Upstream error text indicates auth/key problem",
    };
  }
  if (text.includes("quota") || text.includes("rate limit")) {
    return {
      kind: "quota",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 429,
      code: "upstream_quota_text",
      reason: "Upstream error text indicates quota/rate limit",
    };
  }
  if (text.includes("billing")) {
    return {
      kind: "billing",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 402,
      code: "upstream_billing_text",
      reason: "Upstream error text indicates billing issue",
    };
  }
  if (text.includes("overloaded")) {
    return {
      kind: "transient_provider",
      shouldFailover: true,
      shouldCooldownKey: true,
      clientStatus: 502,
      code: "upstream_overloaded_text",
      reason: "Upstream error text indicates overloaded",
    };
  }

  return {
    kind: "unknown_upstream",
    shouldFailover: false,
    shouldCooldownKey: false,
    clientStatus: 502,
    code: "upstream_unknown",
    reason: `Unclassified upstream error (${status})`,
  };
}

/**
 * Check if request context suggests a likely payload/model incompatibility.
 */
function looksLikelyPayloadModelError(context) {
  const { hasImages, hasFiles, hasTools, hasToolChoice, reasoningEffort } = context;
  let score = 0;

  if (hasImages) score += 2;
  if (hasFiles) score += 1;
  if (hasTools) score += 2;
  if (hasToolChoice) score += 1;
  if (reasoningEffort === "high" || reasoningEffort === "medium") score += 2;

  return score >= 4;
}

function extractErrorText(errorPayload) {
  if (!errorPayload) return "";
  if (typeof errorPayload === "string") return errorPayload;
  if (errorPayload.error) {
    if (typeof errorPayload.error === "string") return errorPayload.error;
    if (errorPayload.error.message) return String(errorPayload.error.message);
    return JSON.stringify(errorPayload.error);
  }
  try {
    return JSON.stringify(errorPayload);
  } catch {
    return String(errorPayload || "");
  }
}

/**
 * Create a safe request fingerprint from shape-level fields only.
 * Does NOT include prompt text, base64, file content, or API keys.
 */
export function createRequestFingerprint(requestShape) {
  const fingerprintData = {
    model: requestShape.model || "",
    stream: !!requestShape.stream,
    hasImages: !!requestShape.hasImages,
    hasFiles: !!requestShape.hasFiles,
    hasTools: !!requestShape.hasTools,
    hasToolChoice: !!requestShape.hasToolChoice,
    reasoningEffort: requestShape.reasoningEffort || "",
    numMessages: requestShape.numMessages || 0,
    messageRoles: requestShape.messageRoles || [],
    contentPartTypes: requestShape.contentPartTypes || [],
    attachmentMimeTypes: requestShape.attachmentMimeTypes || [],
    attachmentSizeBuckets: requestShape.attachmentSizeBuckets || [],
  };

  return createHash("sha256")
    .update(JSON.stringify(fingerprintData))
    .digest("hex");
}

/**
 * Build a safe request shape summary from a normalized payload.
 * Never includes prompt text, base64 data, or API keys.
 */
export function buildRequestShape(payload, originalMessages) {
  const msgs = originalMessages || payload.messages || [];
  let hasImages = false;
  let hasFiles = false;
  const messageRoles = [];
  const contentPartTypes = new Set();
  const attachmentMimeTypes = new Set();
  let approxAttachmentBytes = 0;

  for (const m of msgs) {
    if (!m || typeof m !== "object") continue;
    messageRoles.push(m.role || "?");

    if (typeof m.content === "string") {
      contentPartTypes.add("text_string");
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (!part || typeof part !== "object") continue;
        contentPartTypes.add(part.type || "unknown");

        if (part.type === "image_url" || part.type === "input_image") {
          hasImages = true;
          const url = part.image_url?.url || "";
          if (url.startsWith("data:")) {
            const mime = extractMime(url);
            if (mime) attachmentMimeTypes.add(mime);
            approxAttachmentBytes += estimateBase64Size(url);
          }
        }

        if (part.type === "file" || part.type === "input_file") {
          hasFiles = true;
          const mime = part.mime_type || part.file?.mime_type || guessFromFilename(part.filename) || "application/octet-stream";
          attachmentMimeTypes.add(mime);
          if (part.base64) approxAttachmentBytes += estimateBase64Size(part.base64);
        }
      }
    }
  }

  return {
    model: payload.model || "",
    stream: !!payload.stream,
    hasImages,
    hasFiles,
    hasTools: !!(payload.tools && Array.isArray(payload.tools) && payload.tools.length > 0),
    hasToolChoice: payload.tool_choice !== undefined,
    reasoningEffort: payload.reasoning_effort || "",
    numMessages: msgs.length,
    messageRoles,
    contentPartTypes: [...contentPartTypes].sort(),
    attachmentMimeTypes: [...attachmentMimeTypes].sort(),
    attachmentSizeBuckets: [bucketSize(approxAttachmentBytes)],
  };
}

function extractMime(dataUrl) {
  const match = String(dataUrl).match(/^data:([^;]*)/);
  return match ? match[1] : null;
}

function guessFromFilename(filename) {
  if (!filename) return null;
  const ext = String(filename).split(".").pop()?.toLowerCase();
  const map = {
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    webp: "image/webp",
    gif: "image/gif",
    pdf: "application/pdf",
    txt: "text/plain",
    json: "application/json",
  };
  return map[ext] || null;
}

function estimateBase64Size(b64) {
  return Math.floor((String(b64 || "").length * 3) / 4);
}

function bucketSize(bytes) {
  if (bytes <= 0) return "0";
  if (bytes < 1024) return "<1KB";
  if (bytes < 1024 * 1024) return "<1MB";
  if (bytes < 10 * 1024 * 1024) return "<10MB";
  if (bytes < 50 * 1024 * 1024) return "<50MB";
  return ">=50MB";
}

/**
 * In-memory tracker for repeated upstream 500 failures by request fingerprint.
 */
export function createFingerprintTracker({ maxAttempts, ttlMs } = {}) {
  const entries = new Map();

  const max = maxAttempts || MAX_UPSTREAM_500_FAILOVER_ATTEMPTS;
  const ttl = ttlMs || UPSTREAM_500_FINGERPRINT_TTL_MS;

  function evict(atMs) {
    for (const [fp, entry] of entries) {
      if (entry.expiresAt <= atMs) entries.delete(fp);
    }
  }

  return {
    get size() {
      return entries.size;
    },
    clear() {
      entries.clear();
    },
    recordAttempt(fingerprint, keyIndex, classification) {
      const atMs = Date.now();
      evict(atMs);

      let entry = entries.get(fingerprint);
      if (!entry) {
        entry = {
          attempts: [],
          firstSeenAt: atMs,
          expiresAt: atMs + ttl,
        };
        entries.set(fingerprint, entry);
      } else {
        // Extend TTL
        entry.expiresAt = atMs + ttl;
      }

      entry.attempts.push({
        keyIndex,
        kind: classification.kind,
        status: 500,
        atMs,
      });

      return entry;
    },
    isBlocked(fingerprint) {
      const atMs = Date.now();
      evict(atMs);
      const entry = entries.get(fingerprint);
      if (!entry) return false;
      return entry.attempts.length >= max;
    },
    get(fingerprint) {
      const atMs = Date.now();
      evict(atMs);
      return entries.get(fingerprint) || null;
    },
  };
}
