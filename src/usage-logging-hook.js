const enabled = String(process.env.USAGE_LOGGING || "true").toLowerCase() !== "false";
const originalFetch = globalThis.fetch;

function getUrl(input) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input?.url || "";
}

function parseBody(init) {
  if (typeof init?.body !== "string") return {};
  try {
    return JSON.parse(init.body);
  } catch {
    return {};
  }
}

function readUsageFromSse(text) {
  let usage = {};
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    try {
      const event = JSON.parse(data);
      if (event?.usage && typeof event.usage === "object") usage = event.usage;
    } catch {}
  }
  return usage;
}

function usageNumber(usage, ...keys) {
  for (const key of keys) {
    const value = usage?.[key];
    if (Number.isFinite(value)) return value;
  }
  return 0;
}

function rate(tokens, seconds) {
  return tokens && seconds > 0 ? (tokens / seconds).toFixed(2) : "0.00";
}

async function logUsage(response, request, startedAtNs) {
  try {
    const contentType = response.headers.get("content-type") || "";
    const text = await response.text();
    let usage = {};

    if (contentType.includes("text/event-stream")) {
      usage = readUsageFromSse(text);
    } else {
      try {
        usage = JSON.parse(text || "{}").usage || {};
      } catch {}
    }

    const elapsedMs = Number(process.hrtime.bigint() - startedAtNs) / 1e6;
    const seconds = elapsedMs / 1000;
    const prompt = usageNumber(usage, "prompt_tokens", "input_tokens");
    const completion = usageNumber(usage, "completion_tokens", "output_tokens");
    const total = usageNumber(usage, "total_tokens") || prompt + completion;

    console.log(
      `[usage] model=${request.model || "?"} stream=${request.stream === true} status=${response.status} ` +
      `elapsedMs=${elapsedMs.toFixed(0)} prompt=${prompt} completion=${completion} total=${total} ` +
      `outputTps=${rate(completion, seconds)} totalTps=${rate(total, seconds)}`
    );
  } catch (error) {
    const elapsedMs = Number(process.hrtime.bigint() - startedAtNs) / 1e6;
    console.warn(`[usage] model=${request.model || "?"} elapsedMs=${elapsedMs.toFixed(0)} error=${error.message}`);
  }
}

if (enabled && typeof originalFetch === "function") {
  globalThis.fetch = async function fetchWithUsageLogging(input, init) {
    const url = getUrl(input);
    if (!url.includes("generativelanguage.googleapis.com") || !url.includes("/openai/chat/completions")) {
      return originalFetch(input, init);
    }

    const request = parseBody(init);
    const startedAtNs = process.hrtime.bigint();
    const response = await originalFetch(input, init);
    if (response.ok) void logUsage(response.clone(), request, startedAtNs);
    return response;
  };
}
