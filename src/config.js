const env = process.env;

export const config = {
  host: env.HOST || "127.0.0.1",
  port: Number(env.PORT || 8787),
  localProviderKey: env.LOCAL_PROVIDER_KEY || "",
  defaultModel: env.DEFAULT_MODEL || "gemini-flash-latest",
  forceDefaultModel: String(env.FORCE_DEFAULT_MODEL || "false").toLowerCase() === "true",
  defaultReasoningEffort: (env.REASONING_EFFORT || "").trim().toLowerCase(),
  forceReasoningEffort: String(env.FORCE_REASONING_EFFORT || "false").toLowerCase() === "true",
  maxAttemptsPerRequest: Math.max(1, Number(env.MAX_ATTEMPTS_PER_REQUEST || 3)),
  keyCooldownMs: Math.max(1000, Number(env.KEY_COOLDOWN_MS || 60_000)),
  upstreamTimeoutMs: Math.max(10_000, Number(env.UPSTREAM_TIMEOUT_MS || 300_000)),
  thoughtSignatureDummyFallback: String(env.THOUGHT_SIGNATURE_DUMMY_FALLBACK || "true").toLowerCase() === "true",
  thoughtSignatureCacheTtlMs: Math.max(1000, Number(env.THOUGHT_SIGNATURE_CACHE_TTL_MS || 1_800_000)),
  thoughtSignatureCacheMaxEntries: Math.max(1, Number(env.THOUGHT_SIGNATURE_CACHE_MAX_ENTRIES || 10_000)),

  // Debugging / observability
  debugRequestShapes: String(env.DEBUG_REQUEST_SHAPES || "false").toLowerCase() === "true",
  debugNormalizedPayloads: String(env.DEBUG_NORMALIZED_PAYLOADS || "false").toLowerCase() === "true",
  debugUpstreamModelVersion: String(env.DEBUG_UPSTREAM_MODEL_VERSION || "false").toLowerCase() === "true",

  // Multimodal feature flags
  enableVision: String(env.ENABLE_VISION || "true").toLowerCase() === "true",
  advertiseVision: String(env.ADVERTISE_VISION || "true").toLowerCase() === "true",
  enableFileTextExtraction: String(env.ENABLE_FILE_TEXT_EXTRACTION || "true").toLowerCase() === "true",
  enableRemoteFileFetch: String(env.ENABLE_REMOTE_FILE_FETCH || "false").toLowerCase() === "true",
  enableNativeGeminiFiles: String(env.ENABLE_NATIVE_GEMINI_FILES || "false").toLowerCase() === "true",

  // Safety limits
  maxRequestBodyBytes: Math.max(1, Number(env.MAX_REQUEST_BODY_BYTES || 104857600)), // 100MB default
  maxImageBytes: Math.max(1, Number(env.MAX_IMAGE_BYTES || 20971520)), // 20MB default
  maxFileBytes: Math.max(1, Number(env.MAX_FILE_BYTES || 52428800)), // 50MB default
  maxTotalAttachmentBytes: Math.max(1, Number(env.MAX_TOTAL_ATTACHMENT_BYTES || 104857600)), // 100MB default
  fileTextMaxChars: Math.max(1, Number(env.FILE_TEXT_MAX_CHARS || 200000)),
  remoteFetchTimeoutMs: Math.max(1000, Number(env.REMOTE_FETCH_TIMEOUT_MS || 10000)),
};

export const SUPPORTED_IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);

export const TEXT_MIMES = new Set([
  "text/plain",
  "text/markdown",
  "text/csv",
  "text/html",
  "application/json",
]);

export const EXTRACTABLE_MIMES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);
