# OpenCode Gemini Local Failover Provider

A local OpenAI-compatible proxy that forwards `/v1/chat/completions` to the Gemini OpenAI-compatible endpoint and rotates across multiple Gemini API keys on auth, billing, quota, and transient errors. Also exposes a minimal Ollama-compatible API so VS Code's GitHub Copilot Chat can use Gemini as a BYOK model.

Supports text, tool/function calling, images/vision, and common document attachments (PDF, DOCX, XLSX, TXT, MD, JSON, CSV, HTML).

## Feature matrix

| Feature | Status |
| --- | --- |
| Text chat | Supported |
| Tools / function calling | Supported |
| Streaming (SSE / NDJSON) | Supported |
| Images via OpenAI `image_url` | Supported (PNG, JPEG, WebP, GIF) |
| Images via Ollama `images[]` | Supported |
| PDF | Supported via text extraction |
| DOCX | Supported via text extraction |
| XLSX | Supported via sheet extraction |
| TXT, MD, JSON, CSV | Supported (injected as text) |
| HTML | Supported (tags stripped) |
| Audio / video | Unsupported (returns clear error) |
| Remote file URLs | Disabled by default (security) |
| Native Gemini Files API | Disabled by default |

## What it does and what it doesn't

This proxy is meant to improve reliability, not to bypass per-key limits.

- One upstream request per client request — no fan-out across keys at the same time.
- Honors `Retry-After` if the provider returns it.
- Puts a key on cooldown after `429`, `5xx`, or timeout.
- Disables a key after `401`. `403` goes on cooldown.
- Does not retry ordinary `400` responses (bad request, unsupported model, etc.).
- Extracts text from file attachments locally and injects it into the prompt.
- Validates image and file sizes; rejects oversized/unsupported attachments with clear errors.
- Never silently drops user-provided attachments.

## Quick start (Docker)

```bash
cp .env.example .env
```

Fill in `.env`:

```env
GEMINI_API_KEYS=key1,key2,key3
LOCAL_PROVIDER_KEY=local-dev-key
DEFAULT_MODEL=gemini-flash-latest
FORCE_DEFAULT_MODEL=false
HOST=0.0.0.0
PORT=8787
PUBLISHED_HOST=127.0.0.1
```

Build and run:

```bash
docker compose up -d --build
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

Logs and stop:

```bash
docker compose logs -f
docker compose down
```

## Configuration

All settings come from `.env`. See `.env.example` for the full set.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEYS` | — | Comma-separated Gemini API keys. Required. |
| `LOCAL_PROVIDER_KEY` | `local-dev-key` | Bearer token clients must send to this proxy. |
| `DEFAULT_MODEL` | `gemini-flash-latest` | Model used when the client doesn't specify one. |
| `FORCE_DEFAULT_MODEL` | `false` | If `true`, override the client's model with `DEFAULT_MODEL`. |
| `HOST` | `0.0.0.0` | Interface the Node server binds to inside the container. |
| `PORT` | `8787` | Server port. |
| `PUBLISHED_HOST` | `127.0.0.1` | Host interface Docker publishes the port on (`127.0.0.1` = local only, `0.0.0.0` = LAN). |
| `MAX_ATTEMPTS_PER_REQUEST` | `3` | How many keys to try per request before giving up. |
| `KEY_COOLDOWN_MS` | `60000` | Cooldown duration after `429`/`5xx`/timeout. |
| `UPSTREAM_TIMEOUT_MS` | `300000` | Per-attempt upstream timeout. |
| `REASONING_EFFORT` | `` | Default reasoning effort: `none`, `minimal`, `low`, `medium`, `high`. |
| `FORCE_REASONING_EFFORT` | `false` | If `true`, always replace the client's reasoning effort with `REASONING_EFFORT`. |
| `THOUGHT_SIGNATURE_DUMMY_FALLBACK` | `true` | If a Gemini `thought_signature` is missing, inject Google's documented dummy value. |
| `THOUGHT_SIGNATURE_CACHE_TTL_MS` | `1800000` | TTL for cached thought signatures. |
| `THOUGHT_SIGNATURE_CACHE_MAX_ENTRIES` | `10000` | Max entries in the thought-signature cache. |
| `DEBUG_REQUEST_SHAPES` | `false` | Log request metadata (route, model, attachment counts/types). |
| `DEBUG_NORMALIZED_PAYLOADS` | `false` | Log normalized payload structure (content shapes, image sizes). |
| `DEBUG_UPSTREAM_MODEL_VERSION` | `false` | Enable `GET /debug/model-version` endpoint. |
| `ENABLE_VISION` | `true` | Enable image/vision processing. |
| `ADVERTISE_VISION` | `true` | Advertise `vision` capability in `/api/show`. |
| `ENABLE_FILE_TEXT_EXTRACTION` | `true` | Enable text extraction from file attachments. |
| `ENABLE_REMOTE_FILE_FETCH` | `false` | Enable remote file URL fetching (security risk). |
| `ENABLE_NATIVE_GEMINI_FILES` | `false` | Enable native Gemini Files API path (experimental). |
| `MAX_REQUEST_BODY_BYTES` | `104857600` | Max request body size (100 MB). |
| `MAX_IMAGE_BYTES` | `20971520` | Max per-image attachment size (20 MB). |
| `MAX_FILE_BYTES` | `52428800` | Max per-file attachment size (50 MB). |
| `MAX_TOTAL_ATTACHMENT_BYTES` | `104857600` | Max combined attachment size (100 MB). |
| `FILE_TEXT_MAX_CHARS` | `200000` | Max characters of extracted file text. |
| `REMOTE_FETCH_TIMEOUT_MS` | `10000` | Timeout for remote file fetching. |

Notes:

- For Docker, keep `HOST=0.0.0.0`. For non-Docker local runs, prefer `HOST=127.0.0.1`.
- `PUBLISHED_HOST=0.0.0.0` exposes the proxy on your LAN — see the security section.

## OpenAI API surface

Both prefixed and unprefixed paths work:

```txt
/v1/chat/completions
/chat/completions
/v1/models
/models
```

So either `baseURL` is valid in clients:

```txt
http://127.0.0.1:8787/v1
http://127.0.0.1:8787
```

### Payload normalization

OpenCode can send extra top-level fields that Gemini's `/chat/completions` rejects (for example `instructions`, producing `Invalid JSON payload received. Unknown name "instructions"...`). The proxy normalizes the payload before forwarding:

- `instructions` → folded into a `system` message.
- `developer` role → `system`.
- Unknown top-level fields that Gemini rejects are stripped.
- Standard fields (`messages`, `tools`, `tool_choice`, `temperature`, `stream`, `max_tokens`, …) are preserved.
- `image_url` and file attachment parts are validated, size-checked, and normalized.
- File attachments are text-extracted and injected as text blocks.

### Gemini tool calling: `thought_signature`

Gemini (especially Gemini 3) requires `thought_signature` to be echoed back in the next assistant tool-call message. The proxy:

- Caches `tool_calls[].extra_content.google.thought_signature` from Gemini responses.
- Reattaches the signature to subsequent assistant `tool_calls` by `tool_call.id`.
- Falls back to Google's documented `skip_thought_signature_validator` value if `THOUGHT_SIGNATURE_DUMMY_FALLBACK=true` and the cache is empty.

Reference: [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures).

### OpenAI-compatible examples

#### Text chat

```bash
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "gemini-flash-latest",
    "messages": [{"role":"user","content":"Say hello in one sentence"}],
    "stream": false
  }'
```

#### Image (data URL)

```bash
IMAGE_B64="$(base64 -w0 test.png)"

curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d "{
    \"model\": \"gemini-flash-latest\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image briefly.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_B64\"}}
      ]
    }],
    \"stream\": false
  }" | jq
```

#### PDF as base64 file

```bash
PDF_B64="$(base64 -w0 sample.pdf)"

curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d "{
    \"model\": \"gemini-flash-latest\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Summarize the attached PDF.\"},
        {
          \"type\": \"input_file\",
          \"filename\": \"sample.pdf\",
          \"mime_type\": \"application/pdf\",
          \"base64\": \"$PDF_B64\"
        }
      ]
    }],
    \"stream\": false
  }" | jq
```

## Ollama-compatible API surface

Copilot Chat can use custom models through its **Ollama** provider. The proxy exposes a minimal Ollama-compatible API:

```txt
GET  /api/version    fake Ollama version
GET  /api/tags       model list in Ollama schema
POST /api/show       model capabilities (advertises completion, tools, vision)
POST /api/chat       Ollama-native chat, NDJSON streaming
```

`/api/chat` translates the request body to OpenAI format, runs it through the same Gemini-key failover, and translates the response back to Ollama format (including `tool_calls`).

### Copilot Chat BYOK / Ollama notes

- `/api/show` advertises `vision` when `ADVERTISE_VISION=true`.
- Copilot may or may not send attachments depending on its BYOK/Ollama implementation.
- Inline completions do not use BYOK — only Chat and Edits/Agent modes.
- If an API Key field is shown in Copilot, set it to your `LOCAL_PROVIDER_KEY`.

### Ollama image example

```bash
IMAGE_B64="$(base64 -w0 test.png)"

curl -s http://127.0.0.1:8787/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemini-flash-latest:latest\",
    \"stream\": false,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": \"Describe this image briefly.\",
      \"images\": [\"$IMAGE_B64\"]
    }]
  }" | jq
```

### Expose to your LAN

In `.env`:

```env
HOST=0.0.0.0
PORT=8787
PUBLISHED_HOST=0.0.0.0
```

Open `8787/tcp` in the firewall, then from a client machine:

```bash
curl http://<LAN_IP>:8787/api/version
```

### Configure Copilot Chat

1. Open Copilot Chat → model picker → **Manage Models…** (or Command Palette: `Chat: Manage Language Models`).
2. **Add Models…** → **Ollama**.
3. URL: `http://<LAN_IP>:8787` (no `/v1` — the Ollama client targets `/api/*` itself).
4. If an API Key field is shown, set it to your `LOCAL_PROVIDER_KEY`. If there's no field, leave it blank.
5. Pick a model from the list, e.g. `gemini-flash-latest:latest` or one of the reasoning variants like `gemini-flash-latest-medium:latest`.

## Try a chat completion

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "gemini-flash-latest",
    "messages": [{"role":"user","content":"Say hello in one sentence"}],
    "stream": false
  }'
```

## OpenCode setup

Copy `opencode.example.json` into your OpenCode config:

- Linux/macOS: `~/.config/opencode/opencode.json`
- Windows (PowerShell): `$env:USERPROFILE\.config\opencode\opencode.json`

Set the local bearer token in your shell environment so OpenCode can authenticate against the proxy.

Bash/Zsh:

```bash
export LOCAL_PROVIDER_KEY="local-dev-key"
```

PowerShell:

```powershell
setx LOCAL_PROVIDER_KEY "local-dev-key"
```

Restart your shell/OpenCode, open `/models`, and pick:

```txt
gemini-local/gemini-flash-latest
```

### Context limit and reasoning variants

OpenCode doesn't know context limits for a custom provider automatically. `opencode.example.json` provides:

```json
{
  "models": {
    "gemini-flash-latest": {
      "name": "Gemini Flash Latest",
      "limit": { "context": 1048576, "output": 65536 },
      "variants": {
        "minimal": { "reasoningEffort": "minimal" },
        "low":     { "reasoningEffort": "low" },
        "medium":  { "reasoningEffort": "medium" },
        "high":    { "reasoningEffort": "high" }
      }
    }
  }
}
```

If your OpenCode build picks up variants, you can pick them directly:

```txt
gemini-local/gemini-flash-latest:minimal
gemini-local/gemini-flash-latest:low
gemini-local/gemini-flash-latest:medium
gemini-local/gemini-flash-latest:high
```

If variants don't show up in the UI, use the dedicated model IDs that the proxy maps to `gemini-flash-latest` + the matching `reasoning_effort`:

```txt
gemini-local/gemini-flash-latest-minimal
gemini-local/gemini-flash-latest-low
gemini-local/gemini-flash-latest-medium
gemini-local/gemini-flash-latest-high
```

For Gemini 2.5 Flash, `none` is also available (`gemini-local/gemini-2.5-flash-none`). For `gemini-flash-latest` / Gemini 3, prefer `minimal` — thinking can't be fully disabled.

The proxy accepts both `reasoningEffort` (OpenCode-style) and `reasoning_effort` (OpenAI-style).

## Reasoning / thinking level

Control Gemini thinking via env:

```env
# Empty = let OpenCode / Gemini choose the default
REASONING_EFFORT=

# Allowed: none, minimal, low, medium, high
# Recommended fast setting for gemini-flash-latest / Gemini 3 Flash:
# REASONING_EFFORT=minimal

# false: clients can still override per request
# true:  the proxy always replaces the client value with REASONING_EFFORT
FORCE_REASONING_EFFORT=false
```

Examples:

```env
# Cheapest/fastest for Gemini 3 Flash-style models
REASONING_EFFORT=minimal
FORCE_REASONING_EFFORT=true
```

```env
# More careful reasoning
REASONING_EFFORT=medium
FORCE_REASONING_EFFORT=true
```

```env
# Disable thinking where supported (mostly Gemini 2.5 Flash/Lite)
REASONING_EFFORT=none
FORCE_REASONING_EFFORT=true
```

Google documents `reasoning_effort` on the Gemini OpenAI-compatible API. Gemini 3 can't fully disable thinking — use `minimal` instead of `none`.

## Models

Default model:

```txt
gemini-flash-latest
```

Pin a stable version if you prefer:

```env
DEFAULT_MODEL=gemini-2.5-flash
```

If a client keeps sending the wrong model, force the override:

```env
FORCE_DEFAULT_MODEL=true
```

## Debugging / troubleshooting

### Check if the client is sending images/files

Set `DEBUG_REQUEST_SHAPES=true` in `.env` and check logs:

```bash
docker compose logs -f | grep debug-request-shape
```

Output shows for each request: route, model, message count, and per-message: role, content type, text part count, image part count, file part count, attachment MIME types, attachment byte sizes. Full prompts and base64 data are never logged.

### Check normalized payloads

Set `DEBUG_NORMALIZED_PAYLOADS=true` for more detail on the payload sent upstream.

### Check which model version Gemini resolves

Set `DEBUG_UPSTREAM_MODEL_VERSION=true` and query:

```bash
curl -s "http://127.0.0.1:8787/debug/model-version?model=gemini-flash-latest" \
  -H "Authorization: Bearer local-dev-key" | jq
```

### Error: "Unsupported attachment type"

The client sent a file type the proxy doesn't understand (e.g. `.zip`, `.mp4`). Check the error message for supported types. If the file type should be supported, check that `ENABLE_FILE_TEXT_EXTRACTION=true`.

### Error: "attachment too large"

Increase `MAX_FILE_BYTES`, `MAX_IMAGE_BYTES`, or `MAX_TOTAL_ATTACHMENT_BYTES` in `.env`.

### Running without Docker

```bash
npm install
cp .env.example .env
npm start
```

For non-Docker runs, prefer:

```env
HOST=127.0.0.1
```

## Security notes

- `.env` is gitignored — keep it that way. Never commit real keys.
- Debug logging (`DEBUG_REQUEST_SHAPES`, `DEBUG_NORMALIZED_PAYLOADS`) never logs full prompts, file contents, or base64 data. It logs only metadata: attachment counts, types, and byte sizes.
- Remote file URL fetching is **disabled by default** (`ENABLE_REMOTE_FILE_FETCH=false`). Remote URLs return a clear error.
- With `PUBLISHED_HOST=0.0.0.0`, the proxy is reachable from your LAN. Restrict `8787/tcp` at the firewall.
- The Ollama API surface allows unauthenticated requests by design (Copilot Chat doesn't always send a key). Rely on firewall rules if running on a LAN.
- The OpenAI surface (`/v1/chat/completions` and `/chat/completions`) always requires `Authorization: Bearer $LOCAL_PROVIDER_KEY`.
- File size limits are enforced: `MAX_IMAGE_BYTES` (20 MB default), `MAX_FILE_BYTES` (50 MB default), `MAX_TOTAL_ATTACHMENT_BYTES` (100 MB default).
- `MAX_REQUEST_BODY_BYTES` (100 MB default) prevents memory exhaustion from oversized requests.

## Upstream 500 handling

The proxy distinguishes provider transient errors from likely model/payload incompatibilities.
It will not burn every API key on repeated same-payload 500s.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `upstream_model_payload_error` | Unsupported model/capability combo | Use Gemini Flash, disable images/tools/reasoning, or enable capability override |
| Repeated 429 | quota/rate limit | Add quota, wait cooldown, reduce request rate |
| Gemma fails with images/tools | Gemma capability mismatch | Use Gemini model or opt in with env flags after testing |

## Troubleshooting

#### `model_capability_mismatch` (400)

The request uses a model with features it doesn't support. For example, `gemma-4-31b-it` does not support images, tools, or reasoning effort by default.

Fix:
- Switch to a Gemini Flash model (`gemini-flash-latest`, `gemini-3.1-flash-lite`)
- Or disable images/tools/reasoning in your client
- Or override: `GEMMA_ALLOW_VISION=true`, `GEMMA_ALLOW_TOOLS=true`, `GEMMA_ALLOW_REASONING_EFFORT=true`

#### `upstream_model_payload_error` (502)

The upstream returned repeated 500s for the same request shape across multiple keys. The proxy stops retrying to avoid burning all keys.

Fix:
- Check if the model supports all requested features (images, tools, reasoning)
- Try a different model
- Increase `MAX_UPSTREAM_500_FAILOVER_ATTEMPTS` if you want more retries

#### Debugging model capabilities

```bash
# List all known model capabilities
curl -s http://127.0.0.1:8787/debug/model-capabilities | jq

# Check a specific model
curl -s "http://127.0.0.1:8787/debug/model-capabilities?model=gemma-4-31b-it" | jq
```

## License

[MIT](./LICENSE)
