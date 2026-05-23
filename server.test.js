import assert from "node:assert/strict";
import test from "node:test";

process.env.GEMINI_API_KEYS = process.env.GEMINI_API_KEYS || "test-key";
process.env.LOCAL_PROVIDER_KEY = "local-test-key";

const mod = await import("./server.js");
const {
  server,
  normalizeMessages,
  cacheToolCallsFromCompletionPayload,
  createThoughtSignatureCache,
  applyKeyFailureState,
  createKeyState,
  thoughtSignatureByToolCallId,
  readThoughtSignatureFromToolCall,
  ollamaMessagesToOpenAI,
  ollamaToolsToOpenAI,
  ollamaOptionsToOpenAI,
  openAiToolCallsToOllama,
  stripModelTag,
  OLLAMA_FAKE_VERSION,
} = mod;

// Import multimodal module for direct unit tests
const multimodal = await import("./src/multimodal.js");
const config = await import("./src/config.js");

// =============================================================================
// Existing tests (preserved from original)
// =============================================================================

test("preserves existing thought signature", () => {
  thoughtSignatureByToolCallId.clear();
  const messages = normalizeMessages([
    {
      role: "assistant",
      tool_calls: [
        {
          id: "call-1",
          type: "function",
          function: { name: "default_api:skill", arguments: "{}" },
          extra_content: { google: { thought_signature: "sig-1" } },
        },
      ],
    },
  ]);

  const toolCall = messages[0].tool_calls[0];
  assert.equal(readThoughtSignatureFromToolCall(toolCall), "sig-1");
});

test("restores thought signature from cache", () => {
  thoughtSignatureByToolCallId.clear();
  cacheToolCallsFromCompletionPayload({
    choices: [
      {
        message: {
          tool_calls: [
            {
              id: "call-2",
              extra_content: { google: { thought_signature: "sig-2" } },
            },
          ],
        },
      },
    ],
  });

  const messages = normalizeMessages([
    {
      role: "assistant",
      tool_calls: [
        {
          id: "call-2",
          type: "function",
          function: { name: "default_api:skill", arguments: "{}" },
        },
      ],
    },
  ]);

  const toolCall = messages[0].tool_calls[0];
  assert.equal(readThoughtSignatureFromToolCall(toolCall), "sig-2");
});

test("repairs only first tool call in a message", () => {
  thoughtSignatureByToolCallId.clear();
  const messages = normalizeMessages([
    {
      role: "assistant",
      tool_calls: [
        {
          id: "call-3a",
          type: "function",
          function: { name: "a", arguments: "{}" },
        },
        {
          id: "call-3b",
          type: "function",
          function: { name: "b", arguments: "{}" },
        },
      ],
    },
  ]);

  assert.equal(
    readThoughtSignatureFromToolCall(messages[0].tool_calls[0]),
    "skip_thought_signature_validator"
  );
  assert.equal(readThoughtSignatureFromToolCall(messages[0].tool_calls[1]), undefined);
});

test("health response is minimal and does not expose internal fields", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/health`);
    assert.equal(response.status, 200);

    const payload = await response.json();
    assert.deepEqual(Object.keys(payload).sort(), ["ok", "provider", "uptimeSec"].sort());
    assert.equal(payload.ok, true);
    assert.equal(payload.provider, "gemini-openai-failover");
    assert.equal(typeof payload.uptimeSec, "number");
    assert.equal("keys" in payload, false);
    assert.equal("defaultModel" in payload, false);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("models endpoint uses soft auth: missing/empty bearer allowed, wrong key rejected", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

  try {
    const { port } = server.address();

    const noHeader = await fetch(`http://127.0.0.1:${port}/v1/models`);
    assert.equal(noHeader.status, 200);

    const emptyBearer = await fetch(`http://127.0.0.1:${port}/v1/models`, {
      headers: { authorization: "Bearer" },
    });
    assert.equal(emptyBearer.status, 200);

    const wrong = await fetch(`http://127.0.0.1:${port}/v1/models`, {
      headers: { authorization: "Bearer not-the-key" },
    });
    assert.equal(wrong.status, 401);

    const ok = await fetch(`http://127.0.0.1:${port}/v1/models`, {
      headers: { authorization: "Bearer local-test-key" },
    });
    assert.equal(ok.status, 200);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("thought signature cache supports ttl expiration and max entries", () => {
  const cache = createThoughtSignatureCache({ ttlMs: 10, maxEntries: 2 });
  cache.set("a", "sig-a", 0);
  cache.set("b", "sig-b", 0);
  cache.set("c", "sig-c", 0);

  assert.equal(cache.get("a", 1), null);
  assert.equal(cache.get("b", 1), "sig-b");
  assert.equal(cache.get("c", 1), "sig-c");

  assert.equal(cache.get("b", 11), null);
  assert.equal(cache.get("c", 11), null);
});

test("403 puts key on cooldown while 401 disables key", () => {
  const keyState = createKeyState(["k1", "k2"]);

  applyKeyFailureState(keyState[0], 403, new Headers(), 60_000);
  assert.equal(keyState[0].disabled, false);
  assert.equal(keyState[0].cooldownUntil > Date.now(), true);

  applyKeyFailureState(keyState[1], 401, new Headers(), 60_000);
  assert.equal(keyState[1].disabled, true);
});

test("/v1/chat/completions strips Ollama-style :latest from model name", async () => {
  let upstreamModel = null;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("generativelanguage.googleapis.com")) {
      const payload = JSON.parse(init.body);
      upstreamModel = payload.model;
      return new Response(
        JSON.stringify({
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer" },
      body: JSON.stringify({
        model: "gemini-flash-latest-high:latest",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    });
    assert.equal(response.status, 200);
    assert.equal(upstreamModel, "gemini-flash-latest");
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

test("stripModelTag drops :latest and :tag suffix", () => {
  assert.equal(stripModelTag("gemini-flash-latest:latest"), "gemini-flash-latest");
  assert.equal(stripModelTag("gemini-2.5-flash:foo"), "gemini-2.5-flash");
  assert.equal(stripModelTag("gemini-flash-latest"), "gemini-flash-latest");
});

test("ollamaMessagesToOpenAI handles plain text, images, and tool_calls", () => {
  const out = ollamaMessagesToOpenAI([
    { role: "user", content: "hello" },
    {
      role: "user",
      content: "look at this",
      images: ["base64imgdata"],
    },
    {
      role: "assistant",
      tool_calls: [
        { function: { name: "search", arguments: { q: "node" } } },
      ],
    },
    { role: "tool", content: "result", tool_call_id: "tc-1" },
  ]);

  assert.equal(out[0].role, "user");
  assert.equal(out[0].content, "hello");

  assert.equal(out[1].role, "user");
  assert.equal(Array.isArray(out[1].content), true);
  assert.equal(out[1].content[0].type, "text");
  assert.equal(out[1].content[1].type, "image_url");
  assert.match(out[1].content[1].image_url.url, /^data:image\/png;base64,base64imgdata$/);

  assert.equal(out[2].role, "assistant");
  assert.equal(out[2].tool_calls[0].type, "function");
  assert.equal(out[2].tool_calls[0].function.name, "search");
  assert.equal(typeof out[2].tool_calls[0].function.arguments, "string");
  assert.deepEqual(JSON.parse(out[2].tool_calls[0].function.arguments), { q: "node" });

  assert.equal(out[3].tool_call_id, "tc-1");
});

test("ollamaToolsToOpenAI normalizes both wrapped and bare function entries", () => {
  const tools = [
    { type: "function", function: { name: "a", parameters: {} } },
    { function: { name: "b", parameters: {} } },
  ];
  const out = ollamaToolsToOpenAI(tools);
  assert.equal(out[0].type, "function");
  assert.equal(out[0].function.name, "a");
  assert.equal(out[1].type, "function");
  assert.equal(out[1].function.name, "b");
});

test("ollamaOptionsToOpenAI maps known options and ignores unknown", () => {
  const out = ollamaOptionsToOpenAI({
    temperature: 0.5,
    top_p: 0.9,
    num_predict: 256,
    seed: 42,
    stop: ["END"],
    num_ctx: 4096,
    keep_alive: "5m",
  });
  assert.deepEqual(out, {
    temperature: 0.5,
    top_p: 0.9,
    max_tokens: 256,
    seed: 42,
    stop: ["END"],
  });
});

test("openAiToolCallsToOllama parses arguments string into object", () => {
  const out = openAiToolCallsToOllama([
    {
      id: "tc1",
      type: "function",
      function: { name: "search", arguments: '{"q":"node"}' },
    },
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].function.name, "search");
  assert.deepEqual(out[0].function.arguments, { q: "node" });
});

test("openAiToolCallsToOllama returns undefined for empty or missing input", () => {
  assert.equal(openAiToolCallsToOllama(undefined), undefined);
  assert.equal(openAiToolCallsToOllama([]), undefined);
});

test("/api/version is reachable without auth and reports advertised version", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/version`);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.version, OLLAMA_FAKE_VERSION);
    assert.match(body.version, /^\d+\.\d+\.\d+$/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/tags lists models with :latest suffix and Ollama-style details", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/tags`);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(Array.isArray(body.models), true);
    const names = body.models.map((m) => m.name);
    assert.ok(names.includes("gemini-flash-latest:latest"), `expected gemini-flash-latest:latest in ${names.join(", ")}`);
    const first = body.models[0];
    assert.equal(first.details.family, "gemini");
    assert.match(first.digest, /^sha256:[0-9a-f]{64}$/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/show advertises tool capability for a requested model", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/show`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gemini-flash-latest:latest" }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.ok(Array.isArray(body.capabilities));
    assert.ok(body.capabilities.includes("tools"));
    assert.equal(body.details.family, "gemini");
    assert.equal(body.model_info["gemini.context_length"], 1048576);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/show advertises vision capability when ADVERTISE_VISION is enabled", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/show`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gemini-flash-latest:latest" }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    // ADVERTISE_VISION is true by default
    assert.ok(body.capabilities.includes("vision"), `expected vision in capabilities: ${JSON.stringify(body.capabilities)}`);
    assert.equal(body.model_info["gemini.vision"], true);
    assert.equal(body.model_info["gemini.image_input"], true);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/chat returns Ollama-shaped JSON (non-stream) by translating upstream completion", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("/chat/completions")) {
      const body = {
        id: "cmpl-1",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gemini-flash-latest",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hello world" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gemini-flash-latest:latest",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.model, "gemini-flash-latest:latest");
    assert.equal(body.message.role, "assistant");
    assert.equal(body.message.content, "hello world");
    assert.equal(body.done, true);
    assert.equal(body.done_reason, "stop");
    assert.equal(body.prompt_eval_count, 5);
    assert.equal(body.eval_count, 2);
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/chat (stream) emits NDJSON with final done frame", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("/chat/completions")) {
      const sse = [
        `data: ${JSON.stringify({ id: "1", choices: [{ index: 0, delta: { role: "assistant", content: "he" } }] })}\n\n`,
        `data: ${JSON.stringify({ id: "1", choices: [{ index: 0, delta: { content: "llo" } }] })}\n\n`,
        `data: ${JSON.stringify({ id: "1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 1 } })}\n\n`,
        `data: [DONE]\n\n`,
      ].join("");
      return new Response(sse, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gemini-flash-latest:latest",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    assert.equal(response.status, 200);
    assert.match(response.headers.get("content-type") || "", /application\/x-ndjson/);

    const text = await response.text();
    const lines = text.trim().split("\n").filter(Boolean);
    const frames = lines.map((l) => JSON.parse(l));

    const contentFrames = frames.filter((f) => f.done === false);
    assert.ok(contentFrames.length >= 1, "expected at least one non-final frame");
    assert.equal(contentFrames[0].message.role, "assistant");

    const finalFrame = frames[frames.length - 1];
    assert.equal(finalFrame.done, true);
    assert.equal(finalFrame.done_reason, "stop");
    assert.equal(finalFrame.prompt_eval_count, 3);

    const combined = contentFrames.map((f) => f.message.content).join("");
    assert.equal(combined, "hello");
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/api/chat with auth header still works when LOCAL_PROVIDER_KEY matches", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("/chat/completions")) {
      return new Response(
        JSON.stringify({
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const okResp = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": "Bearer local-test-key",
      },
      body: JSON.stringify({
        model: "gemini-flash-latest:latest",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    });
    assert.equal(okResp.status, 200);

    const badResp = await fetch(`http://127.0.0.1:${port}/api/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": "Bearer wrong-key",
      },
      body: JSON.stringify({
        model: "gemini-flash-latest:latest",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    });
    assert.equal(badResp.status, 401);
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

// =============================================================================
// Multimodal normalization tests
// =============================================================================

test("normalizeContent: string content returns text part", () => {
  const result = multimodal.normalizeContent("hello world");
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[0].text, "hello world");
});

test("normalizeContent: array of text parts preserved", () => {
  const result = multimodal.normalizeContent([
    { type: "text", text: "part 1" },
    { type: "text", text: "part 2" },
  ]);
  assert.equal(result.parts.length, 2);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[0].text, "part 1");
  assert.equal(result.parts[1].kind, "text");
  assert.equal(result.parts[1].text, "part 2");
});

test("normalizeContent: image_url object shape preserved", () => {
  const dataUrl = "data:image/png;base64,iVBORw0KGgo=";
  const result = multimodal.normalizeContent([
    { type: "image_url", image_url: { url: dataUrl } },
  ]);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "image");
  assert.equal(result.parts[0].mimeType, "image/png");
  assert.equal(result.parts[0].dataUrl, dataUrl);
});

test("normalizeContent: input_image with image_url converted", () => {
  const dataUrl = "data:image/jpeg;base64,iVBORw0KGgo=";
  const result = multimodal.normalizeContent([
    { type: "input_image", image_url: { url: dataUrl } },
  ]);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "image");
  assert.equal(result.parts[0].mimeType, "image/jpeg");
});

test("normalizeContent: input_text converted to text", () => {
  const result = multimodal.normalizeContent([
    { type: "input_text", text: "hello from input_text" },
  ]);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[0].text, "hello from input_text");
});

test("normalizeContent: unsupported part produces unsupported kind", () => {
  const result = multimodal.normalizeContent([
    { type: "unknown_binary", data: "..." },
  ]);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "unsupported");
});

test("normalizeContent: mixed text and image parts", () => {
  const dataUrl = "data:image/png;base64,abc=";
  const result = multimodal.normalizeContent([
    { type: "text", text: "Look at this:" },
    { type: "image_url", image_url: { url: dataUrl } },
  ]);
  assert.equal(result.parts.length, 2);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[1].kind, "image");
});

test("normalizeContent: null content returns empty text", () => {
  const result = multimodal.normalizeContent(null);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[0].text, "");
});

test("normalizeContent: non-array non-string content stringified", () => {
  const result = multimodal.normalizeContent(42);
  assert.equal(result.parts.length, 1);
  assert.equal(result.parts[0].kind, "text");
  assert.equal(result.parts[0].text, "42");
});

// =============================================================================
// Image handling tests
// =============================================================================

test("normalizeImagePart: rejects images over MAX_IMAGE_BYTES", () => {
  // Create a base64 string that's too large
  const hugeBase64 = Buffer.alloc(30 * 1024 * 1024).toString("base64");
  const dataUrl = `data:image/png;base64,${hugeBase64}`;
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: dataUrl },
  });
  assert.equal(result.kind, "unsupported");
  assert.match(result.reason, /too large/);
});

test("normalizeImagePart: rejects unsupported MIME types", () => {
  const dataUrl = "data:image/bmp;base64,iVBORw0KGgo=";
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: dataUrl },
  });
  assert.equal(result.kind, "unsupported");
  assert.match(result.reason, /unsupported image MIME/);
});

test("normalizeImagePart: rejects invalid base64", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: "data:image/png;base64,!!!invalid!!!" },
  });
  assert.equal(result.kind, "unsupported");
});

test("normalizeImagePart: accepts valid image/png", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/png");
});

test("normalizeImagePart: accepts valid image/jpeg", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: "data:image/jpeg;base64,iVBORw0KGgo=" },
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/jpeg");
});

test("normalizeImagePart: accepts valid image/webp", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: "data:image/webp;base64,iVBORw0KGgo=" },
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/webp");
});

test("normalizeImagePart: accepts valid image/gif", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: { url: "data:image/gif;base64,iVBORw0KGgo=" },
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/gif");
});

test("normalizeImagePart: handles input_image with base64 + mime_type", () => {
  const result = multimodal.normalizeImagePart({
    type: "input_image",
    base64: "iVBORw0KGgo=",
    mime_type: "image/png",
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/png");
  assert.ok(result.dataUrl.startsWith("data:image/png;base64,"));
});

test("normalizeImagePart: handles image_url as string", () => {
  const result = multimodal.normalizeImagePart({
    type: "image_url",
    image_url: "data:image/png;base64,iVBORw0KGgo=",
  });
  assert.equal(result.kind, "image");
  assert.equal(result.mimeType, "image/png");
});

// =============================================================================
// buildUpstreamContent tests
// =============================================================================

test("buildUpstreamContent: text-only content", async () => {
  const result = await multimodal.buildUpstreamContent([{ type: "text", text: "hello" }]);
  assert.equal(result.openAiContent.length, 1);
  assert.equal(result.openAiContent[0].type, "text");
  assert.equal(result.openAiContent[0].text, "hello");
  assert.equal(result.hasAnyImage, false);
});

test("buildUpstreamContent: image content produces image_url parts", async () => {
  const result = await multimodal.buildUpstreamContent([
    { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
  ]);
  assert.equal(result.openAiContent.length, 1);
  assert.equal(result.openAiContent[0].type, "image_url");
  assert.ok(result.openAiContent[0].image_url.url.startsWith("data:image/png;base64,"));
  assert.equal(result.hasAnyImage, true);
});

test("buildUpstreamContent: throws on unsupported part", async () => {
  await assert.rejects(
    multimodal.buildUpstreamContent([{ type: "bad_type", data: "..." }]),
    (err) => err.name === "MultimodalError"
  );
});

test("buildUpstreamContent: string content works", async () => {
  const result = await multimodal.buildUpstreamContent("hello");
  assert.equal(result.openAiContent.length, 1);
  assert.equal(result.openAiContent[0].text, "hello");
});

// =============================================================================
// Data URL helper tests
// =============================================================================

test("parseDataUrl: parses valid data URL", () => {
  const result = multimodal.parseDataUrl("data:image/png;base64,iVBORw0KGgo=");
  assert.equal(result.mimeType, "image/png");
  assert.equal(result.isBase64, true);
  assert.equal(result.data, "iVBORw0KGgo=");
});

test("parseDataUrl: parses non-base64 data URL", () => {
  const result = multimodal.parseDataUrl("data:text/plain,hello%20world");
  assert.equal(result.mimeType, "text/plain");
  assert.equal(result.isBase64, false);
  assert.equal(result.data, "hello%20world");
});

test("parseDataUrl: returns null for non-data URLs", () => {
  assert.equal(multimodal.parseDataUrl("https://example.com/image.png"), null);
  assert.equal(multimodal.parseDataUrl("not a url"), null);
});

test("estimateDataUrlBytes: estimates size from base64", () => {
  // "iVBORw0KGgo=" is 12 chars → ~9 bytes
  const bytes = multimodal.estimateDataUrlBytes("data:image/png;base64,iVBORw0KGgo=");
  assert.ok(bytes > 0 && bytes < 20);
});

test("validateBase64: accepts valid base64", () => {
  assert.equal(multimodal.validateBase64("iVBORw0KGgo="), true);
  assert.equal(multimodal.validateBase64("abcd"), true);
});

test("validateBase64: rejects invalid base64", () => {
  assert.equal(multimodal.validateBase64("!!!invalid"), false);
});

// =============================================================================
// MIME type helper tests
// =============================================================================

test("guessMimeFromFilename: maps extensions correctly", () => {
  assert.equal(multimodal.guessMimeFromFilename("test.pdf"), "application/pdf");
  assert.equal(multimodal.guessMimeFromFilename("test.docx"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
  assert.equal(multimodal.guessMimeFromFilename("test.xlsx"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
  assert.equal(multimodal.guessMimeFromFilename("test.txt"), "text/plain");
  assert.equal(multimodal.guessMimeFromFilename("test.md"), "text/markdown");
  assert.equal(multimodal.guessMimeFromFilename("test.json"), "application/json");
  assert.equal(multimodal.guessMimeFromFilename("test.csv"), "text/csv");
  assert.equal(multimodal.guessMimeFromFilename("test.html"), "text/html");
  assert.equal(multimodal.guessMimeFromFilename("test.png"), "image/png");
  assert.equal(multimodal.guessMimeFromFilename("test.jpg"), "image/jpeg");
  assert.equal(multimodal.guessMimeFromFilename("test.jpeg"), "image/jpeg");
  assert.equal(multimodal.guessMimeFromFilename("test.webp"), "image/webp");
  assert.equal(multimodal.guessMimeFromFilename("test.gif"), "image/gif");
});

test("guessMimeFromFilename: returns null for unknown extensions", () => {
  assert.equal(multimodal.guessMimeFromFilename("test.xyz"), null);
  assert.equal(multimodal.guessMimeFromFilename("noextension"), null);
});

// =============================================================================
// Config tests
// =============================================================================

test("config: default feature flags are set correctly", () => {
  assert.equal(config.config.enableVision, true);
  assert.equal(config.config.advertiseVision, true);
  assert.equal(config.config.enableFileTextExtraction, true);
  assert.equal(config.config.enableRemoteFileFetch, false);
  assert.equal(config.config.enableNativeGeminiFiles, false);
});

test("config: safety limits have reasonable defaults", () => {
  assert.ok(config.config.maxImageBytes > 0);
  assert.ok(config.config.maxFileBytes > 0);
  assert.ok(config.config.maxTotalAttachmentBytes > 0);
  assert.ok(config.config.fileTextMaxChars > 0);
});

test("config: supported MIME sets contain expected types", () => {
  assert.ok(config.SUPPORTED_IMAGE_MIMES.has("image/png"));
  assert.ok(config.SUPPORTED_IMAGE_MIMES.has("image/jpeg"));
  assert.ok(config.SUPPORTED_IMAGE_MIMES.has("image/webp"));
  assert.ok(config.TEXT_MIMES.has("text/plain"));
  assert.ok(config.EXTRACTABLE_MIMES.has("application/pdf"));
});

// =============================================================================
// File normalization tests
// =============================================================================

test("normalizeFilePart: extracts file metadata from input_file", () => {
  const result = multimodal.normalizeFilePart({
    type: "input_file",
    filename: "test.pdf",
    base64: "iVBORw0KGgo=",
    mime_type: "application/pdf",
  });
  assert.equal(result.kind, "file");
  assert.equal(result.mimeType, "application/pdf");
  assert.equal(result.filename, "test.pdf");
});

test("normalizeFilePart: extracts file with file_data data URL", () => {
  const result = multimodal.normalizeFilePart({
    type: "input_file",
    filename: "doc.txt",
    file_data: "data:text/plain;base64,SGVsbG8gV29ybGQ=",
  });
  assert.equal(result.kind, "file");
  assert.equal(result.mimeType, "text/plain");
  assert.equal(result.filename, "doc.txt");
});

test("normalizeFilePart: extracts text from text file part", () => {
  const result = multimodal.normalizeFilePart({
    type: "input_file",
    filename: "notes.md",
    text: "# Hello\nWorld",
  });
  assert.equal(result.kind, "file");
  assert.equal(result.text, "# Hello\nWorld");
  assert.equal(result.filename, "notes.md");
});

test("normalizeFilePart: guesses MIME from filename when missing", () => {
  const result = multimodal.normalizeFilePart({
    type: "input_file",
    filename: "report.pdf",
    base64: "abc",
  });
  assert.equal(result.mimeType, "application/pdf");
});

test("normalizeFilePart: rejects file over MAX_FILE_BYTES", () => {
  const hugeBase64 = Buffer.alloc(60 * 1024 * 1024).toString("base64");
  const result = multimodal.normalizeFilePart({
    type: "input_file",
    filename: "big.pdf",
    base64: hugeBase64,
    mime_type: "application/pdf",
  });
  assert.equal(result.kind, "unsupported");
  assert.match(result.reason, /too large/);
});

// =============================================================================
// Ollama multimodal tests
// =============================================================================

test("ollamaImagesToParts: converts plain base64 to data URL image parts", () => {
  const parts = multimodal.ollamaImagesToParts(["base64data123"]);
  assert.equal(parts.length, 1);
  assert.equal(parts[0].type, "image_url");
  assert.equal(parts[0].image_url.url, "data:image/png;base64,base64data123");
});

test("ollamaImagesToParts: preserves existing data URLs", () => {
  const parts = multimodal.ollamaImagesToParts(["data:image/jpeg;base64,abcd"]);
  assert.equal(parts.length, 1);
  assert.equal(parts[0].image_url.url, "data:image/jpeg;base64,abcd");
});

test("ollamaImagesToParts: filters out empty/invalid entries", () => {
  const parts = multimodal.ollamaImagesToParts(["", null, undefined]);
  assert.equal(parts.length, 0);
});

test("ollamaMessagesToOpenAI: images converted with vision support", () => {
  const out = ollamaMessagesToOpenAI([
    {
      role: "user",
      content: "What's in this image?",
      images: ["data:image/png;base64,iVBORw0KGgo="],
    },
  ]);
  assert.equal(out[0].role, "user");
  assert.equal(Array.isArray(out[0].content), true);
  assert.equal(out[0].content.length, 2);
  assert.equal(out[0].content[0].type, "text");
  assert.equal(out[0].content[0].text, "What's in this image?");
  assert.equal(out[0].content[1].type, "image_url");
});

// =============================================================================
// Image proxying integration test
// =============================================================================

test("/v1/chat/completions: image_url parts reach upstream as image_url", async () => {
  let upstreamPayload = null;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("generativelanguage.googleapis.com")) {
      upstreamPayload = JSON.parse(init.body);
      return new Response(
        JSON.stringify({
          choices: [{ index: 0, message: { role: "assistant", content: "I see a test image" }, finish_reason: "stop" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer" },
      body: JSON.stringify({
        model: "gemini-flash-latest",
        messages: [{
          role: "user",
          content: [
            { type: "text", text: "Describe this" },
            { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
          ],
        }],
        stream: false,
      }),
    });
    assert.equal(response.status, 200);

    // Verify the image was preserved in the upstream payload
    const userMsg = upstreamPayload.messages.find((m) => m.role === "user");
    assert.ok(Array.isArray(userMsg.content));
    const imagePart = userMsg.content.find((p) => p.type === "image_url");
    assert.ok(imagePart, "expected image_url part in upstream payload");
    assert.ok(imagePart.image_url.url.includes("data:image/png;base64,"));
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

// =============================================================================
// Unsupported attachment rejection test
// =============================================================================

test("/v1/chat/completions: unsupported attachment type returns 400", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("generativelanguage.googleapis.com")) {
      return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer" },
      body: JSON.stringify({
        model: "gemini-flash-latest",
        messages: [{
          role: "user",
          content: [
            { type: "text", text: "Check this file" },
            { type: "input_file", filename: "archive.zip", base64: "UEsDBBQAAAA=", mime_type: "application/zip" },
          ],
        }],
        stream: false,
      }),
    });
    assert.equal(response.status, 400);
    const body = await response.json();
    assert.ok(body.error.message.includes("Unsupported"), `expected 'Unsupported' in error: ${body.error.message}`);
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

// =============================================================================
// Large image rejection test
// =============================================================================

test("/v1/chat/completions: oversized image returns 400", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (typeof url === "string" && url.includes("generativelanguage.googleapis.com")) {
      return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
    }
    return originalFetch(url, init);
  };

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    // Create a base64 string ~25MB
    const hugeBase64 = Buffer.alloc(25 * 1024 * 1024).toString("base64");
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer" },
      body: JSON.stringify({
        model: "gemini-flash-latest",
        messages: [{
          role: "user",
          content: [
            { type: "text", text: "Look" },
            { type: "image_url", image_url: { url: `data:image/png;base64,${hugeBase64}` } },
          ],
        }],
        stream: false,
      }),
    });
    assert.equal(response.status, 400);
    const body = await response.json();
    assert.ok(body.error.message.includes("too large") || body.error.message.includes("Request body"),
      `expected size error: ${JSON.stringify(body)}`);
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise((resolve) => server.close(resolve));
  }
});

// =============================================================================
// Debug model-version endpoint tests
// =============================================================================

test("/debug/model-version returns 403 when DEBUG_UPSTREAM_MODEL_VERSION is disabled", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/debug/model-version?model=gemini-flash-latest`, {
      headers: { authorization: "Bearer local-test-key" },
    });
    // Should be 403 because DEBUG_UPSTREAM_MODEL_VERSION defaults to false
    assert.equal(response.status, 403);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("/debug/model-version requires auth", async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/debug/model-version?model=gemini-flash-latest`);
    // Should be 401 (wrong auth) or 403 (no auth + disabled)
    assert.ok(response.status === 401 || response.status === 403);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
