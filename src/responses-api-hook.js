import { Readable, Writable } from "node:stream";
import http from "node:http";

const RESPONSES_PATHS = new Set(["/v1/responses", "/responses"]);
const UNSUPPORTED_RESPONSES_TOOL_TYPES = new Set([
  "web_search_preview",
  "web_search_preview_2025_03_11",
  "file_search",
  "computer_use_preview",
  "code_interpreter",
  "image_generation",
  "mcp",
]);

const originalCreateServer = http.createServer.bind(http);

function getPath(req) {
  const host = req.headers?.host || "localhost";
  return new URL(req.url || "/", `http://${host}`).pathname.replace(/\/+$/g, "") || "/";
}

function createResponseId() {
  return `resp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function jsonError(res, status, message, code = "invalid_request_error", details = undefined) {
  const body = JSON.stringify({ error: { message, code, details } }, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(new Error(`Invalid JSON: ${error.message}`));
      }
    });
    req.on("error", reject);
  });
}

function contentText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (!part || typeof part !== "object") return "";
        return part.text || part.content || "";
      })
      .filter(Boolean)
      .join("\n");
  }
  if (typeof content === "object") return content.text || content.content || JSON.stringify(content);
  return String(content);
}

function normalizeResponsesContentPart(part) {
  if (typeof part === "string") return { type: "text", text: part };
  if (!part || typeof part !== "object") return { type: "text", text: "" };

  if (part.type === "input_text" || part.type === "output_text" || part.type === "text") {
    return { type: "text", text: part.text || "" };
  }

  if (part.type === "input_image" || part.type === "image_url") {
    const imageUrl = part.image_url || part.url || part.file_data;
    if (typeof imageUrl === "string") {
      return { type: "image_url", image_url: { url: imageUrl } };
    }
    if (imageUrl && typeof imageUrl === "object") {
      return { type: "image_url", image_url: imageUrl };
    }
    return { ...part, type: "input_image" };
  }

  if (part.type === "input_file" || part.type === "file") {
    return {
      type: "input_file",
      filename: part.filename || part.file?.filename,
      mime_type: part.mime_type || part.file?.mime_type,
      file_data: part.file_data || part.file?.file_data,
      base64: part.base64,
      text: part.text,
      file: part.file,
    };
  }

  return part;
}

function normalizeResponsesContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return contentText(content);
  return content.map(normalizeResponsesContentPart).filter(Boolean);
}

function normalizeResponsesInput(input) {
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }

  if (!Array.isArray(input)) {
    throw new Error("Responses API requires `input` to be a string or an array.");
  }

  const messages = [];
  for (const item of input) {
    if (typeof item === "string") {
      messages.push({ role: "user", content: item });
      continue;
    }
    if (!item || typeof item !== "object") continue;

    if (item.type === "message" || item.role) {
      const message = {
        role: item.role || "user",
        content: normalizeResponsesContent(item.content),
      };
      if (item.name) message.name = item.name;
      if (item.tool_call_id) message.tool_call_id = item.tool_call_id;
      if (Array.isArray(item.tool_calls)) message.tool_calls = item.tool_calls;
      messages.push(message);
      continue;
    }

    if (item.type === "function_call_output") {
      messages.push({
        role: "tool",
        tool_call_id: item.call_id || item.id,
        content: item.output == null ? "" : String(item.output),
      });
      continue;
    }

    if (item.type === "input_text" || item.type === "input_image" || item.type === "input_file") {
      messages.push({ role: "user", content: [normalizeResponsesContentPart(item)] });
    }
  }

  if (messages.length === 0) {
    throw new Error("Responses API `input` did not contain any supported message items.");
  }
  return messages;
}

function normalizeResponsesTools(tools) {
  if (tools == null) return undefined;
  if (!Array.isArray(tools)) throw new Error("Responses API `tools` must be an array when provided.");

  const out = [];
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") continue;

    if (tool.type === "function") {
      if (tool.function) {
        out.push({ type: "function", function: tool.function });
      } else {
        out.push({
          type: "function",
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
            strict: tool.strict,
          },
        });
      }
      continue;
    }

    if (UNSUPPORTED_RESPONSES_TOOL_TYPES.has(tool.type)) {
      throw new Error(`Responses built-in tool is not supported by this Gemini adapter: ${tool.type}`);
    }

    throw new Error(`Unsupported Responses API tool type: ${tool.type || "unknown"}`);
  }

  return out.length > 0 ? out : undefined;
}

function normalizeResponsesToolChoice(toolChoice) {
  if (toolChoice == null) return undefined;
  if (typeof toolChoice === "string") return toolChoice;
  if (toolChoice.type === "function") {
    const name = toolChoice.name || toolChoice.function?.name;
    if (name) return { type: "function", function: { name } };
  }
  return toolChoice;
}

function normalizeResponsesTextFormat(text) {
  const format = text?.format;
  if (!format || typeof format !== "object") return undefined;
  if (format.type === "json_schema") {
    return {
      type: "json_schema",
      json_schema: {
        name: format.name || format.json_schema?.name || "response_schema",
        schema: format.schema || format.json_schema?.schema || {},
        strict: format.strict ?? format.json_schema?.strict,
      },
    };
  }
  if (format.type === "json_object") return { type: "json_object" };
  return undefined;
}

function responsesToChatPayload(body) {
  if (body.previous_response_id) {
    throw new Error("previous_response_id is not supported by this stateless Gemini adapter.");
  }

  const payload = {
    model: body.model,
    messages: normalizeResponsesInput(body.input),
    instructions: body.instructions,
    stream: body.stream === true,
  };

  const passthrough = [
    "temperature",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "reasoning",
    "reasoning_effort",
    "reasoningEffort",
  ];
  for (const key of passthrough) {
    if (body[key] !== undefined) payload[key] = body[key];
  }

  if (body.max_output_tokens !== undefined) payload.max_completion_tokens = body.max_output_tokens;
  if (body.max_completion_tokens !== undefined) payload.max_completion_tokens = body.max_completion_tokens;
  if (body.max_tokens !== undefined) payload.max_tokens = body.max_tokens;

  const tools = normalizeResponsesTools(body.tools);
  if (tools) payload.tools = tools;

  const toolChoice = normalizeResponsesToolChoice(body.tool_choice);
  if (toolChoice !== undefined) payload.tool_choice = toolChoice;

  const responseFormat = body.response_format || normalizeResponsesTextFormat(body.text);
  if (responseFormat) payload.response_format = responseFormat;

  return payload;
}

function createInternalRequest(originalReq, payload) {
  const body = JSON.stringify(payload);
  const req = Readable.from([body]);
  req.method = "POST";
  req.url = "/v1/chat/completions";
  req.headers = {
    ...originalReq.headers,
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  };
  return req;
}

class CaptureResponse extends Writable {
  constructor() {
    super();
    this.statusCode = 200;
    this.headers = {};
    this.chunks = [];
    this.done = new Promise((resolve) => {
      this._resolveDone = resolve;
    });
  }

  writeHead(statusCode, headers = {}) {
    this.statusCode = statusCode;
    this.headers = { ...this.headers, ...headers };
    return this;
  }

  setHeader(name, value) {
    this.headers[String(name).toLowerCase()] = value;
  }

  getHeader(name) {
    return this.headers[String(name).toLowerCase()];
  }

  _write(chunk, _encoding, callback) {
    this.chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
    callback();
  }

  end(chunk, encoding, callback) {
    if (chunk) this.chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk), encoding));
    super.end(callback);
    this._resolveDone();
    return this;
  }

  text() {
    return Buffer.concat(this.chunks).toString("utf8");
  }
}

async function callChatCompletions(listener, originalReq, chatPayload) {
  const internalReq = createInternalRequest(originalReq, chatPayload);
  const internalRes = new CaptureResponse();
  await Promise.resolve(listener(internalReq, internalRes));
  await internalRes.done;
  return internalRes;
}

function extractMessageText(message) {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : part?.text || part?.content || ""))
      .filter(Boolean)
      .join("\n");
  }
  return content == null ? "" : String(content);
}

function mapUsage(usage = {}) {
  const inputTokens = usage.input_tokens ?? usage.prompt_tokens ?? 0;
  const outputTokens = usage.output_tokens ?? usage.completion_tokens ?? 0;
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: usage.total_tokens ?? inputTokens + outputTokens,
    input_tokens_details: usage.prompt_tokens_details || usage.input_tokens_details || {},
    output_tokens_details: usage.completion_tokens_details || usage.output_tokens_details || {},
  };
}

function chatCompletionToResponse(chat, requestBody, responseId = createResponseId()) {
  const choice = chat.choices?.[0] || {};
  const message = choice.message || {};
  const text = extractMessageText(message);
  const output = [];

  output.push({
    id: `msg_${responseId.slice(5)}`,
    type: "message",
    status: "completed",
    role: "assistant",
    content: [
      {
        type: "output_text",
        text,
        annotations: [],
      },
    ],
  });

  for (const toolCall of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (toolCall?.type !== "function") continue;
    output.push({
      id: toolCall.id,
      type: "function_call",
      status: "completed",
      call_id: toolCall.id,
      name: toolCall.function?.name || "",
      arguments: toolCall.function?.arguments || "{}",
    });
  }

  return {
    id: responseId,
    object: "response",
    created_at: chat.created || Math.floor(Date.now() / 1000),
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: requestBody.instructions ?? null,
    max_output_tokens: requestBody.max_output_tokens ?? requestBody.max_completion_tokens ?? null,
    model: chat.model || requestBody.model,
    output,
    output_text: text,
    parallel_tool_calls: requestBody.parallel_tool_calls ?? true,
    previous_response_id: requestBody.previous_response_id ?? null,
    reasoning: requestBody.reasoning ?? null,
    store: requestBody.store ?? false,
    temperature: requestBody.temperature ?? null,
    text: requestBody.text ?? null,
    tool_choice: requestBody.tool_choice ?? "auto",
    tools: requestBody.tools ?? [],
    top_p: requestBody.top_p ?? null,
    truncation: requestBody.truncation ?? "disabled",
    usage: mapUsage(chat.usage),
    user: requestBody.user,
    metadata: requestBody.metadata ?? null,
  };
}

function writeSse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function parseChatSse(text) {
  const events = [];
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    try {
      events.push(JSON.parse(data));
    } catch {}
  }
  return events;
}

function streamChatSseAsResponses(res, chatSseText, requestBody, responseId = createResponseId()) {
  const createdAt = Math.floor(Date.now() / 1000);
  const responseBase = {
    id: responseId,
    object: "response",
    created_at: createdAt,
    status: "in_progress",
    model: requestBody.model,
    output: [],
    output_text: "",
  };
  const itemId = `msg_${responseId.slice(5)}`;
  const outputIndex = 0;
  const contentIndex = 0;
  let text = "";
  let usage = {};

  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    "connection": "keep-alive",
  });

  writeSse(res, "response.created", { response: responseBase });
  writeSse(res, "response.in_progress", { response: responseBase });
  writeSse(res, "response.output_item.added", {
    output_index: outputIndex,
    item: { id: itemId, type: "message", status: "in_progress", role: "assistant", content: [] },
  });
  writeSse(res, "response.content_part.added", {
    item_id: itemId,
    output_index: outputIndex,
    content_index: contentIndex,
    part: { type: "output_text", text: "", annotations: [] },
  });

  for (const event of parseChatSse(chatSseText)) {
    if (event.usage) usage = event.usage;
    const delta = event.choices?.[0]?.delta;
    if (typeof delta?.content !== "string" || delta.content.length === 0) continue;
    text += delta.content;
    writeSse(res, "response.output_text.delta", {
      item_id: itemId,
      output_index: outputIndex,
      content_index: contentIndex,
      delta: delta.content,
    });
  }

  writeSse(res, "response.output_text.done", {
    item_id: itemId,
    output_index: outputIndex,
    content_index: contentIndex,
    text,
  });
  writeSse(res, "response.content_part.done", {
    item_id: itemId,
    output_index: outputIndex,
    content_index: contentIndex,
    part: { type: "output_text", text, annotations: [] },
  });
  writeSse(res, "response.output_item.done", {
    output_index: outputIndex,
    item: {
      id: itemId,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text, annotations: [] }],
    },
  });
  writeSse(res, "response.completed", {
    response: {
      ...responseBase,
      status: "completed",
      output_text: text,
      output: [{
        id: itemId,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text, annotations: [] }],
      }],
      usage: mapUsage(usage),
    },
  });
  res.end();
}

async function handleResponses(listener, req, res) {
  let body;
  let chatPayload;
  try {
    body = await readJsonBody(req);
    chatPayload = responsesToChatPayload(body);
  } catch (error) {
    return jsonError(res, 400, error.message);
  }

  const chatRes = await callChatCompletions(listener, req, chatPayload);
  const text = chatRes.text();

  if (chatRes.statusCode < 200 || chatRes.statusCode >= 300) {
    const headers = { ...chatRes.headers };
    delete headers["content-length"];
    res.writeHead(chatRes.statusCode, headers);
    res.end(text);
    return;
  }

  if (chatPayload.stream === true) {
    return streamChatSseAsResponses(res, text, body);
  }

  let chat;
  try {
    chat = text ? JSON.parse(text) : {};
  } catch (error) {
    return jsonError(res, 502, `Failed to parse upstream chat completion response: ${error.message}`, "upstream_parse_error");
  }

  const response = chatCompletionToResponse(chat, body);
  const data = JSON.stringify(response, null, 2);
  res.writeHead(chatRes.statusCode, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(data),
  });
  res.end(data);
}

if (!globalThis.__geminiResponsesApiHookInstalled) {
  globalThis.__geminiResponsesApiHookInstalled = true;

  http.createServer = function createServerWithResponsesApi(options, requestListener) {
    const listener = typeof options === "function" ? options : requestListener;
    const wrappedListener = async (req, res) => {
      if (req.method === "POST" && RESPONSES_PATHS.has(getPath(req))) {
        return handleResponses(listener, req, res);
      }
      return listener(req, res);
    };

    if (typeof options === "function") {
      return originalCreateServer(wrappedListener);
    }
    return originalCreateServer(options, wrappedListener);
  };
}

export {
  responsesToChatPayload,
  chatCompletionToResponse,
  normalizeResponsesInput,
  normalizeResponsesContentPart,
  normalizeResponsesTools,
};
