# TODO

## Authentication and Network Exposure

Authentication is intentionally deferred for now, but the current service should be treated as a local/private tool until this work is completed.

### Current risk

- The API is reachable without authentication once the server is bound beyond `127.0.0.1`.
- `CORS` is currently permissive, so browser-based requests from unrelated sites are allowed if the service is reachable from that browser session.
- The OpenAI-compatible routes accept `Authorization` headers in examples, but no bearer token is enforced yet.
- When exposed on Tailscale or `0.0.0.0`, any reachable peer can use the synthesis endpoints unless another layer is added in front.

### Goal

Ship an authentication model that keeps localhost friction low while making non-local exposure safe and explicit.

### Recommended implementation order

1. Keep the default server posture local-only.
2. Add optional API key or bearer-token auth for all non-static API routes.
3. Restrict `CORS` with a configurable allowlist instead of `*`.
4. Add lightweight abuse controls for synthesis endpoints.
5. Sanitize client-facing error messages and keep detailed failures server-side.

### Proposed auth shape

- Environment variables:
  - `KOKORO_API_KEY`
  - `KOKORO_REQUIRE_AUTH`
  - `KOKORO_ALLOWED_ORIGINS`
- Behavior:
  - If `KOKORO_REQUIRE_AUTH=0`, keep current local-dev behavior.
  - If `KOKORO_REQUIRE_AUTH=1`, require `Authorization: Bearer <token>` on:
    - `/api/health` optional, depending on whether runtime visibility should be public
    - `/api/speak`
    - `/api/chunk-plan`
    - `/api/speak-stream`
    - `/ws/speak-stream`
    - `/v1/models`
    - `/v1/models/{model_id}`
    - `/v1/audio/speech`
- OpenAI-compatible routes should validate bearer auth the same way as native routes.
- WebSocket auth can be accepted through the `Authorization` header during handshake, with a documented fallback only if a specific client cannot send headers.

### CORS changes

- Default `allow_origins` should be empty or localhost-only, not `["*"]`.
- Parse `KOKORO_ALLOWED_ORIGINS` as a comma-separated allowlist.
- If auth is enabled and no allowed origins are configured, prefer disabling browser cross-origin access rather than silently allowing all origins.
- If the UI is served from the same FastAPI app, same-origin requests should continue to work without extra configuration.

### Abuse controls

- Add request size and text-length guards before synthesis work begins.
- Add simple per-process concurrency limits for synthesis tasks.
- Add rate limiting if the service is expected to be reachable by multiple peers.
- Consider a maximum chunk count for streaming routes to cap worst-case work.

### Error handling

- Avoid returning raw exception text from ffmpeg, filesystem errors, or internal stack traces to clients.
- Return stable client errors such as:
  - `Authentication failed.`
  - `Pitch shifting is unavailable on this server.`
  - `Audio encoding failed.`
- Log the detailed underlying error server-side.

### Acceptance criteria

- Localhost usage still works with minimal setup.
- Tailscale or LAN exposure requires an explicit opt-in and token.
- Browser cross-origin access is denied by default unless allowlisted.
- OpenAI-compatible clients can authenticate with standard bearer tokens.
- WebSocket streaming follows the same auth policy as HTTP routes.
