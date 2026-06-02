# Responses API compatibility

This provider exposes a basic OpenAI Responses API adapter in addition to the existing Chat Completions and Ollama-compatible routes.

## Routes

Both prefixed and unprefixed paths are supported:

```txt
POST /v1/responses
POST /responses
```

The adapter does not call a separate Gemini `/responses` upstream. It translates Responses API requests into the existing OpenAI-compatible chat-completions flow, so it reuses the same Gemini key rotation, cooldown, timeout, multimodal normalization, capability validation, and failover behavior.

Existing routes continue to work unchanged:

```txt
POST /v1/chat/completions
POST /chat/completions
GET  /v1/models
GET  /models
GET  /health
GET  /api/version
GET  /api/tags
POST /api/show
POST /api/chat
```

## Supported request shape

### Text input

```bash
curl -s http://127.0.0.1:8787/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "gemini-flash-latest",
    "input": "Say hello in one sentence",
    "stream": false
  }'
```

The non-streaming response includes a Responses-style `output` array and a convenience `output_text` field.

### Message-array input

```json
{
  "model": "gemini-flash-latest",
  "instructions": "Be concise.",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "Describe this image." },
        { "type": "input_image", "image_url": "data:image/png;base64,..." }
      ]
    }
  ]
}
```

The adapter maps:

- `input: "text"` → one user chat message.
- `input[].type: "message"` → chat messages.
- `input_text` / `output_text` / `text` → OpenAI chat text parts.
- `input_image` → OpenAI chat `image_url` parts.
- `input_file` → the existing file extraction path.
- `function_call_output` → chat `tool` messages.
- `instructions` → the same system-message normalization used by `/v1/chat/completions`.

### Function tools

Responses-style function tools are converted to chat-completions function tools:

```json
{
  "tools": [
    {
      "type": "function",
      "name": "lookup_order",
      "description": "Look up an order by id",
      "parameters": { "type": "object", "properties": { "id": { "type": "string" } } }
    }
  ]
}
```

Built-in Responses tools that require OpenAI-side infrastructure are rejected with a clear `400` instead of being silently ignored. Currently unsupported examples include `web_search_preview`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, and `mcp`.

## Streaming

When `stream: true`, the adapter asks the existing chat-completions route for SSE and returns Responses-style SSE events such as:

```txt
event: response.created
event: response.in_progress
event: response.output_item.added
event: response.content_part.added
event: response.output_text.delta
event: response.output_text.done
event: response.content_part.done
event: response.output_item.done
event: response.completed
```

The stream is translated from the upstream chat-completions stream. Existing `/v1/chat/completions` streaming behavior is not changed.

## Limitations

- `previous_response_id` is not supported because the provider is stateless.
- OpenAI-hosted built-in tools are not implemented locally.
- Responses API support is an adapter over chat completions, so unsupported Gemini/chat-completions model-capability combinations still return the existing provider errors.
