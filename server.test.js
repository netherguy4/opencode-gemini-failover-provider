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

    // No header at all: soft-allowed (matches real Ollama, which doesn't auth).
    const noHeader = await fetch(`http://127.0.0.1:${port}/v1/models`);
    assert.equal(noHeader.status, 200);

    // Empty bearer (Copilot Chat 0.48.x sends this): soft-allowed.
    const emptyBearer = await fetch(`http://127.0.0.1:${port}/v1/models`, {
      headers: { authorization: "Bearer" },
    });
    assert.equal(emptyBearer.status, 200);

    // Wrong non-empty key: rejected.
    const wrong = await fetch(`http://127.0.0.1:${port}/v1/models`, {
      headers: { authorization: "Bearer not-the-key" },
    });
    assert.equal(wrong.status, 401);

    // Correct key: allowed.
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
    // Only intercept calls to the upstream Gemini host, not the test->proxy hop.
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
    // The ":latest" suffix must be stripped AND the "-high" convenience suffix
    // must be unpacked into base model + reasoning_effort.
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
    num_ctx: 4096, // ignored
    keep_alive: "5m", // ignored
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

test("/api/chat returns Ollama-shaped JSON (non-stream) by translating upstream completion", async () => {
  // Mock upstream Gemini OpenAI-compatible endpoint via fetch shim
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
