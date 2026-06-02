import test from "node:test";
import assert from "node:assert/strict";

process.env.GEMINI_API_KEYS = process.env.GEMINI_API_KEYS || "test-key-1,test-key-2";
process.env.LOCAL_PROVIDER_KEY = process.env.LOCAL_PROVIDER_KEY || "local-dev-key";
process.env.USAGE_LOGGING = "false";
process.env.HOST = "127.0.0.1";
process.env.PORT = "0";

const nativeFetch = globalThis.fetch.bind(globalThis);
const upstreamRequests = [];

globalThis.fetch = async function fetchStub(input, init = {}) {
  const url = typeof input === "string" ? input : input?.url || String(input);
  if (!url.includes("generativelanguage.googleapis.com")) {
    return nativeFetch(input, init);
  }

  const body = JSON.parse(init.body || "{}");
  upstreamRequests.push({ url, body, headers: init.headers || {} });

  if (body.stream === true) {
    return new Response(
      [
        `data: ${JSON.stringify({ id: "chatcmpl_stream", object: "chat.completion.chunk", choices: [{ delta: { role: "assistant" } }] })}\n\n`,
        `data: ${JSON.stringify({ id: "chatcmpl_stream", object: "chat.completion.chunk", choices: [{ delta: { content: "Hel" } }] })}\n\n`,
        `data: ${JSON.stringify({ id: "chatcmpl_stream", object: "chat.completion.chunk", choices: [{ delta: { content: "lo" }, finish_reason: "stop" }], usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 } })}\n\n`,
        "data: [DONE]\n\n",
      ].join(""),
      { status: 200, headers: { "content-type": "text/event-stream" } }
    );
  }

  return new Response(
    JSON.stringify({
      id: "chatcmpl_test",
      object: "chat.completion",
      created: 1710000000,
      model: body.model || "gemini-flash-latest",
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: "Hello from Gemini" },
        },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
};

const {
  responsesToChatPayload,
  normalizeResponsesInput,
  normalizeResponsesTools,
  normalizeResponsesToolChoice,
  normalizeResponsesContentPart,
  chatCompletionToResponse,
} = await import("../src/responses-api-hook.js");
const { server } = await import("../server.js");

function listen() {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

async function request(port, path, body, headers = {}) {
  const response = await nativeFetch(`http://127.0.0.1:${port}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined
      ? headers
      : { "content-type": "application/json", "authorization": "Bearer local-dev-key", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  const parsed = text && response.headers.get("content-type")?.includes("application/json")
    ? JSON.parse(text)
    : text;
  return { response, text, parsed };
}

test.after(() => {
  if (server.listening) server.close();
});

test("Responses string input converts to one user chat message", () => {
  assert.deepEqual(normalizeResponsesInput("Hello"), [{ role: "user", content: "Hello" }]);
});

test("Responses array input maps input_text/input_image/input_file content parts", () => {
  const messages = normalizeResponsesInput([
    {
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: "Describe" },
        { type: "input_image", image_url: "data:image/png;base64,AAAA" },
        { type: "input_file", filename: "notes.txt", text: "note text" },
      ],
    },
  ]);

  assert.equal(messages[0].role, "user");
  assert.deepEqual(messages[0].content[0], { type: "text", text: "Describe" });
  assert.deepEqual(messages[0].content[1], { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } });
  assert.equal(messages[0].content[2].type, "input_file");
  assert.equal(messages[0].content[2].filename, "notes.txt");
});

test("Responses function tools convert to chat-completions function tools", () => {
  assert.deepEqual(
    normalizeResponsesTools([{ type: "function", name: "lookup", parameters: { type: "object" } }]),
    [{ type: "function", function: { name: "lookup", description: undefined, parameters: { type: "object" }, strict: undefined } }]
  );
});

test("Responses built-in tools fail fast instead of silently dropping behavior", () => {
  assert.throws(
    () => normalizeResponsesTools([{ type: "web_search_preview" }]),
    /not supported/
  );
});

test("Responses request normalizes to chat payload without mutating existing chat route contract", () => {
  const payload = responsesToChatPayload({
    model: "gemini-flash-latest",
    instructions: "Be concise",
    input: "Ping",
    max_output_tokens: 64,
    stream: false,
  });

  assert.equal(payload.model, "gemini-flash-latest");
  assert.equal(payload.instructions, "Be concise");
  assert.deepEqual(payload.messages, [{ role: "user", content: "Ping" }]);
  assert.equal(payload.max_completion_tokens, 64);
  assert.equal(payload.stream, false);
});

test("HTTP routes preserve /health and /v1/chat/completions while adding /v1/responses aliases", async () => {
  const port = server.listening ? server.address().port : await listen();

  const health = await request(port, "/health");
  assert.equal(health.response.status, 200);
  assert.equal(health.parsed.ok, true);

  const chat = await request(port, "/v1/chat/completions", {
    model: "gemini-flash-latest",
    messages: [{ role: "user", content: "hello" }],
    stream: false,
  });
  assert.equal(chat.response.status, 200);
  assert.equal(chat.parsed.object, "chat.completion");
  assert.equal(chat.parsed.choices[0].message.content, "Hello from Gemini");

  const responses = await request(port, "/v1/responses", {
    model: "gemini-flash-latest",
    input: "hello",
    stream: false,
  });
  assert.equal(responses.response.status, 200);
  assert.equal(responses.parsed.object, "response");
  assert.equal(responses.parsed.status, "completed");
  assert.equal(responses.parsed.output_text, "Hello from Gemini");
  assert.equal(responses.parsed.output[0].content[0].type, "output_text");
  assert.equal(responses.parsed.usage.input_tokens, 5);
  assert.equal(responses.parsed.usage.output_tokens, 3);

  const alias = await request(port, "/responses", {
    model: "gemini-flash-latest",
    input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hello" }] }],
    stream: false,
  });
  assert.equal(alias.response.status, 200);
  assert.equal(alias.parsed.output_text, "Hello from Gemini");

  const lastRequest = upstreamRequests.at(-1).body;
  assert.equal(lastRequest.messages[0].role, "user");
  assert.deepEqual(lastRequest.messages[0].content, [{ type: "text", text: "hello" }]);
});

test("/v1/responses rejects unsupported built-in tools with 400", async () => {
  const port = server.listening ? server.address().port : await listen();
  const result = await request(port, "/v1/responses", {
    model: "gemini-flash-latest",
    input: "search the web",
    tools: [{ type: "web_search_preview" }],
  });

  assert.equal(result.response.status, 400);
  assert.match(result.parsed.error.message, /not supported/);
});

test("/v1/responses streaming translates chat SSE into Responses SSE events", async () => {
  const port = server.listening ? server.address().port : await listen();
  const result = await request(port, "/v1/responses", {
    model: "gemini-flash-latest",
    input: "stream hello",
    stream: true,
  });

  assert.equal(result.response.status, 200);
  assert.match(result.response.headers.get("content-type") || "", /text\/event-stream/);
  assert.match(result.text, /event: response\.created/);
  assert.match(result.text, /event: response\.output_text\.delta/);
  assert.match(result.text, /"delta":"Hel"/);
  assert.match(result.text, /event: response\.completed/);
  assert.match(result.text, /"output_text":"Hello"/);
});

test("previous_response_id is rejected with 400", async () => {
  const port = server.listening ? server.address().port : await listen();
  const result = await request(port, "/v1/responses", {
    model: "gemini-flash-latest",
    input: "hello",
    previous_response_id: "resp_abc123",
  });

  assert.equal(result.response.status, 400);
  assert.match(result.parsed.error.message, /previous_response_id/);
});

test("function_call_output maps to tool role message", () => {
  const messages = normalizeResponsesInput([
    { type: "message", role: "assistant", content: "I will look that up" },
    { type: "function_call_output", call_id: "call_123", output: "Order found: #456" },
  ]);

  assert.equal(messages.length, 2);
  assert.equal(messages[1].role, "tool");
  assert.equal(messages[1].tool_call_id, "call_123");
  assert.equal(messages[1].content, "Order found: #456");
});

test("normalizeResponsesToolChoice handles string and function object", () => {
  assert.equal(normalizeResponsesToolChoice("auto"), "auto");
  assert.equal(normalizeResponsesToolChoice("none"), "none");
  assert.deepEqual(
    normalizeResponsesToolChoice({ type: "function", name: "lookup" }),
    { type: "function", function: { name: "lookup" } }
  );
  assert.deepEqual(
    normalizeResponsesToolChoice({ type: "function", function: { name: "search" } }),
    { type: "function", function: { name: "search" } }
  );
  assert.equal(normalizeResponsesToolChoice(null), undefined);
  assert.equal(normalizeResponsesToolChoice(undefined), undefined);
});

test("Bare content parts accumulate into a single user message", () => {
  const messages = normalizeResponsesInput([
    { type: "input_text", text: "Describe this" },
    { type: "input_image", image_url: "data:image/png;base64,AAAA" },
  ]);

  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "user");
  assert.equal(messages[0].content.length, 2);
  assert.deepEqual(messages[0].content[0], { type: "text", text: "Describe this" });
  assert.deepEqual(messages[0].content[1], { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } });
});

test("Invalid content parts are filtered out", () => {
  assert.equal(normalizeResponsesContentPart(null), null);
  assert.equal(normalizeResponsesContentPart(undefined), null);
  assert.equal(normalizeResponsesContentPart(42), null);
  assert.equal(normalizeResponsesContentPart(false), null);
});

test("tool_calls in non-streaming response are mapped to function_call output", () => {
  const chat = {
    id: "chatcmpl_test",
    object: "chat.completion",
    created: 1710000000,
    model: "gemini-flash-latest",
    choices: [{
      index: 0,
      finish_reason: "tool_calls",
      message: {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_abc",
          type: "function",
          function: { name: "lookup_order", arguments: '{"id":"123"}' },
        }],
      },
    }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };

  const response = chatCompletionToResponse(chat, { model: "gemini-flash-latest" });

  assert.equal(response.status, "completed");
  assert.equal(response.output.length, 2);
  assert.equal(response.output[0].type, "message");
  assert.equal(response.output[1].type, "function_call");
  assert.equal(response.output[1].call_id, "call_abc");
  assert.equal(response.output[1].name, "lookup_order");
  assert.equal(response.output[1].arguments, '{"id":"123"}');
});
