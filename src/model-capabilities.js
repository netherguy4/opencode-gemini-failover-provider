const env = process.env;

export const MODEL_CAPABILITIES = {
  "gemini-flash-latest": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },
  "gemini-3.5-flash": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },
  "gemini-3.1-flash-lite": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },
  "gemini-3-flash-preview": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },
  "gemini-2.5-flash": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },
  "gemini-2.5-flash-lite": {
    vision: true,
    tools: true,
    reasoningEffort: true,
    fileText: true,
  },

  "gemma-4-31b-it": {
    vision: false,
    tools: false,
    reasoningEffort: false,
    fileText: true,
  },
};

export const UNKNOWN_MODEL_CAPABILITY_MODE =
  (env.UNKNOWN_MODEL_CAPABILITY_MODE || "permissive").toLowerCase();

export function getGemmaAllowVision() {
  return String(process.env.GEMMA_ALLOW_VISION || "false").toLowerCase() === "true";
}
export function getGemmaAllowTools() {
  return String(process.env.GEMMA_ALLOW_TOOLS || "false").toLowerCase() === "true";
}
export function getGemmaAllowReasoningEffort() {
  return String(process.env.GEMMA_ALLOW_REASONING_EFFORT || "false").toLowerCase() === "true";
}

export class ProviderRequestError extends Error {
  constructor(message, { status = 400, code, details, hint } = {}) {
    super(message);
    this.name = "ProviderRequestError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.hint = hint;
  }
}

/**
 * Get model capabilities, accounting for overrides.
 * @param {string} model
 * @returns {{ vision: boolean, tools: boolean, reasoningEffort: boolean, fileText: boolean }}
 */
export function getModelCapabilities(model) {
  const normalized = String(model || "").trim().toLowerCase();

  // Check exact match first
  if (MODEL_CAPABILITIES[normalized]) {
    const caps = { ...MODEL_CAPABILITIES[normalized] };
    applyGemmaOverrides(normalized, caps);
    return caps;
  }

  // Check prefix match for variant models (e.g. gemini-flash-latest-minimal -> gemini-flash-latest)
  for (const [knownModel, caps] of Object.entries(MODEL_CAPABILITIES)) {
    if (normalized.startsWith(knownModel)) {
      const result = { ...caps };
      applyGemmaOverrides(normalized, result);
      return result;
    }
  }

  // Unknown model
  return getUnknownModelCapabilities(normalized);
}

function getUnknownModelCapabilities(model) {
  const isGemma = model.startsWith("gemma");
  const isGemini = model.startsWith("gemini");

  if (UNKNOWN_MODEL_CAPABILITY_MODE === "conservative") {
    if (isGemma) {
      const caps = { vision: false, tools: false, reasoningEffort: false, fileText: true };
      applyGemmaOverrides(model, caps);
      return caps;
    }
    return { vision: false, tools: false, reasoningEffort: false, fileText: true };
  }

  // Permissive mode
  if (isGemini) {
    return { vision: true, tools: true, reasoningEffort: true, fileText: true };
  }

  if (isGemma) {
    const caps = { vision: false, tools: false, reasoningEffort: false, fileText: true };
    applyGemmaOverrides(model, caps);
    return caps;
  }

  // Other unknown: permissive, let 500 fingerprint guard handle
  return { vision: true, tools: true, reasoningEffort: true, fileText: true };
}

function applyGemmaOverrides(model, caps) {
  if (model.startsWith("gemma")) {
    if (getGemmaAllowVision()) caps.vision = true;
    if (getGemmaAllowTools()) caps.tools = true;
    if (getGemmaAllowReasoningEffort()) caps.reasoningEffort = true;
  }
}

/**
 * Validate model capabilities against the request shape.
 * Throws ProviderRequestError on local capability mismatch (status 400).
 * May modify payload to strip unsupported fields (reasoning_effort).
 *
 * @param {object} payload - Normalized payload (may be modified)
 * @param {object} requestShape - from buildRequestShape()
 * @param {{ stripReasoning?: boolean }} options
 * @returns {object} - Possibly modified payload
 */
export function validateModelCapabilities(payload, requestShape, options = {}) {
  const model = payload.model || "";
  const caps = getModelCapabilities(model);

  if (requestShape.hasImages && !caps.vision) {
    throw new ProviderRequestError(
      `Model "${model}" does not support vision/image input.`,
      {
        status: 400,
        code: "model_capability_mismatch",
        details: { model, capability: "vision", hasImages: true },
        hint: `Try a Gemini Flash model (gemini-flash-latest, gemini-3.1-flash-lite) for image requests, or set GEMMA_ALLOW_VISION=true to override.`,
      }
    );
  }

  if (requestShape.hasTools && !caps.tools) {
    throw new ProviderRequestError(
      `Model "${model}" does not support tools/function calling.`,
      {
        status: 400,
        code: "model_capability_mismatch",
        details: { model, capability: "tools", hasTools: true },
        hint: `Try a Gemini Flash model for tool/function calling, or set GEMMA_ALLOW_TOOLS=true to override.`,
      }
    );
  }

  if (requestShape.hasToolChoice && !caps.tools) {
    throw new ProviderRequestError(
      `Model "${model}" does not support tools/function calling (tool_choice was set).`,
      {
        status: 400,
        code: "model_capability_mismatch",
        details: { model, capability: "tools", hasToolChoice: true },
        hint: `Try a Gemini Flash model for tool/function calling, or set GEMMA_ALLOW_TOOLS=true to override.`,
      }
    );
  }

  if (requestShape.reasoningEffort && !caps.reasoningEffort) {
    if (options.stripReasoning !== false) {
      delete payload.reasoning_effort;
    } else {
      throw new ProviderRequestError(
        `Model "${model}" does not support reasoning_effort.`,
        {
          status: 400,
          code: "model_capability_mismatch",
          details: { model, capability: "reasoningEffort", reasoningEffort: requestShape.reasoningEffort },
          hint: `Use a Gemini Flash model for reasoning effort, or set GEMMA_ALLOW_REASONING_EFFORT=true to override.`,
        }
      );
    }
  }

  return payload;
}

/**
 * Get a debug-friendly report of all capabilities and settings.
 */
export function getCapabilitiesReport() {
  const allModels = { ...MODEL_CAPABILITIES };

  return {
    models: Object.fromEntries(
      Object.entries(allModels).map(([model, caps]) => {
        const effective = getModelCapabilities(model);
        return [model, effective];
      })
    ),
    unknownModelCapabilityMode: UNKNOWN_MODEL_CAPABILITY_MODE,
    gemmaOverrides: {
      allowVision: getGemmaAllowVision(),
      allowTools: getGemmaAllowTools(),
      allowReasoningEffort: getGemmaAllowReasoningEffort(),
    },
  };
}
