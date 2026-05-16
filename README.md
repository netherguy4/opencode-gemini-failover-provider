# OpenCode Gemini Local Failover Provider

A local OpenAI-compatible proxy that forwards `/v1/chat/completions` to the Gemini OpenAI-compatible endpoint and rotates across multiple Gemini API keys on auth, billing, quota, and transient errors. Also exposes a minimal Ollama-compatible API so VS Code's GitHub Copilot Chat can use Gemini as a BYOK model.

## What it does and what it doesn't

This proxy is meant to improve reliability, not to bypass per-key limits.

- One upstream request per client request — no fan-out across keys at the same time.
- Honors `Retry-After` if the provider returns it.
- Puts a key on cooldown after `429`, `5xx`, or timeout.
- Disables a key after `401`. `403` goes on cooldown.
- Does not retry ordinary `400` responses (bad request, unsupported model, etc.).

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

If you see `Not found: POST /chat/completions`, you're running an old container — rebuild with `docker compose up -d --build`.

### Payload normalization

OpenCode can send extra top-level fields that Gemini's `/chat/completions` rejects (for example `instructions`, producing `Invalid JSON payload received. Unknown name "instructions"...`). The proxy normalizes the payload before forwarding:

- `instructions` → folded into a `system` message.
- `developer` role → `system`.
- Unknown top-level fields that Gemini rejects are stripped.
- Standard fields (`messages`, `tools`, `tool_choice`, `temperature`, `stream`, `max_tokens`, …) are preserved.

### Gemini tool calling: `thought_signature`

Gemini (especially Gemini 3) requires `thought_signature` to be echoed back in the next assistant tool-call message. If a client drops the nonstandard `extra_content` field, you get errors like `Function call is missing a thought_signature ...`.

The proxy:

- Caches `tool_calls[].extra_content.google.thought_signature` from Gemini responses.
- Reattaches the signature to subsequent assistant `tool_calls` by `tool_call.id`.
- Falls back to Google's documented `skip_thought_signature_validator` value if `THOUGHT_SIGNATURE_DUMMY_FALLBACK=true` and the cache is empty.

Reference: [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures).

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

## VS Code · GitHub Copilot Chat (Ollama BYOK)

Copilot Chat can use custom models through its **Ollama** provider. The proxy exposes a minimal Ollama-compatible API alongside the OpenAI endpoints:

```txt
GET  /api/version    fake Ollama version, satisfies Copilot's check
GET  /api/tags       model list in Ollama schema (with :latest suffix)
POST /api/show       model capabilities (advertises tools)
POST /api/chat       Ollama-native chat, NDJSON streaming
```

`/api/chat` translates the request body to OpenAI format, runs it through the same Gemini-key failover, and translates the response back to Ollama format (including `tool_calls`).

### Expose to your LAN

In `.env`:

```env
HOST=0.0.0.0
PORT=8787
PUBLISHED_HOST=0.0.0.0
```

Open `8787/tcp` in the firewall and start the proxy:

```bash
docker compose up -d --build
```

From a client machine:

```bash
curl http://<LAN_IP>:8787/api/version
# {"version":"0.11.0"}
```

### Configure Copilot Chat

1. Open Copilot Chat → model picker → **Manage Models…** (or Command Palette: `Chat: Manage Language Models`).
2. **Add Models…** → **Ollama**.
3. URL: `http://<LAN_IP>:8787` (no `/v1` — the Ollama client targets `/api/*` itself).
4. If an API Key field is shown, set it to your `LOCAL_PROVIDER_KEY`. If there's no field, leave it blank — the shim allows unauthenticated requests; rely on firewall rules.
5. Pick a model from the list, e.g. `gemini-flash-latest:latest` or one of the reasoning variants like `gemini-flash-latest-medium:latest`.

### Known limitations

- Copilot **inline completions** don't use BYOK — those are pinned to Copilot's own models. Only Chat and Edits/Agent modes use the configured provider.
- Vision (`images`) is not advertised in `capabilities`. If Copilot Chat sends images, the proxy forwards them, but Gemini may reject the request unless the selected model supports image input.

## Running without Docker

```bash
npm install
cp .env.example .env
npm start
```

For non-Docker runs, prefer:

```env
HOST=127.0.0.1
```

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

## Security notes

- `.env` is gitignored — keep it that way. Never commit real keys.
- With `PUBLISHED_HOST=0.0.0.0`, the proxy is reachable from your LAN. The Ollama API surface allows unauthenticated requests by design (Copilot Chat doesn't always send a key), so restrict `8787/tcp` at the firewall to known clients, or keep `LOCAL_PROVIDER_KEY` set and make sure your clients send it.
- The OpenAI surface (`/v1/chat/completions` and `/chat/completions`) always requires `Authorization: Bearer $LOCAL_PROVIDER_KEY`.

## License

[MIT](./LICENSE)
