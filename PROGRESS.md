# PROGRESS.md — go-llama-adapter

Living log of the project's origin, the incident that motivated it, and the
work done so far. Newest entries first. Append at the top; keep entries short
and factual. This file is the single source of "how did we get here" — do not
let it drift from git history; if a step is not in git, it is not done.

---

## 2026-10-04 — Docker packaging

- Added `Dockerfile` (multi-stage: `golang:1.27-alpine` builder → `scratch`
  runtime with only the static `llama-gateway` binary plus `wget` for the
  `HEALTHCHECK` on `/healthz`), `docker-compose.yml` (a
  `ggml-org/llama.cpp:server` upstream on the compose network plus the
  gateway exposing `:8090`), and `.dockerignore`.
- Verified: image builds; container starts and `/healthz` returns `ok`.

## 2026-07-21 — Gateway added; DSH integration documented

- Added `cmd/llama-gateway/`: a standalone OpenAI-compatible HTTP gateway
  built on the `llama` library (`POST /v1/chat/completions` streaming +
  blocking, `GET /v1/models`, `GET /healthz`). This is the integration point
  for harnesses that only speak the OpenAI chat-completions protocol, because
  DSH's plugin seam is Node/TS-only and a Go binary can only participate over
  the network.
- Reworked the adapter's tool-call slot so the **deduplicated id is known
  before the `toolcall_start` event is emitted** (the id is captured from the
  first delta that carries it). This lets a gateway emit `id`/`name` on the
  first tool_call delta in OpenAI SSE without buffering, and is the correct
  ordering for the OpenAI protocol.
- Fixed a real **data race** the gateway exposed: the adapter emitted `done`
  at `finish_reason`, but OpenAI streams carry the `usage` chunk *after*
  `finish_reason`. A gateway reading `Partial.Usage` at `done` raced the
  producer (and could read zero usage). Fix: collect trailing metadata after
  `finish_reason` and emit `done` only once the stream has drained. Pinned by
  `TestUsageCompleteAtDone`.
- Documented the DSH topology in README: llama.cpp → gateway → DSH provider
  route (`api: openai-completions`, `baseURL` = gateway).
- Added `AGENTS.md` (project scope, technology, core invariants with named
  tests, coding/testing conventions, PROGRESS.md rules) and this
  `PROGRESS.md` itself (full incident-to-project history, including the
  `dsh-llm-pi-ai` / pi-ai root cause and the upstream PR #10397).

## 2026-07-21 — Adapter library written and committed

- Created the Go module, package `llama`:
  - `types.go` — normalized `Message` / `AssistantMessage` / `ContentBlock` /
    `ToolCall` / `Usage` / `StopReason`.
  - `events.go` — the event protocol (a subset of pi-ai's
    `AssistantMessageEvent` union): `start`, `text_*`, `thinking_*`,
    `toolcall_*`, `done`, `error`. `Stream.Wait()` for one-shot consumption.
  - `client.go` — `Client`/`Config`, `Complete`, `ModelInfo`, and request
    wire encoding (verbatim tool-call argument replay, `reasoning_content`,
    image parts, `stream_options.include_usage`).
  - `stream.go` — SSE scanner and message accumulator.
  - `params.go` — per-request sampling params + `Compat` knobs for specific
    llama.cpp builds.
- `stream_test.go` — 14 tests against an in-process `httptest` mock of the
  llama.cpp endpoint, including `TestDuplicateToolCallIDs` (the core
  regression), interleaved text/tool/text, reasoning blocks, missing
  `finish_reason`, server + HTTP errors, and exact request wire format.
- `go build` / `go vet` / `go test -race` all green.
- Published to `github.com/covrom/go-llama-adapter` (private).

---

# Origin of the project

This project exists because of a concrete corruption incident in DeepSeek
Harness (DSH), and a chain of follow-up questions that led to "let's build
the llama.cpp adapter ourselves, in Go." The full arc, oldest first:

## 1. The incident: a corrupt DSH session log

The DSH Web GUI failed to load a session:

```
Failed to load history: stored session "session-91ba02ca-…" is corrupt:
SessionFormatError: assistant/message repeats advertised tool call
call_PzQ4txSlSK6JUsCHvH94UgsE1a5i8NiG|fc_PzQ4txSlSK6JUsCHvH94UgsE1a5i8NiG
```

The session log is a validated, append-only JSONL (`.jsonl.zstd`). Its
validator requires that within a single `assistant/message`, no two
advertised tool-call blocks share the same id. Decompressing the artifact
showed a whole class of violations: 24 assistant messages contained
duplicate tool-call ids, 32 distinct ids collided, and 64 of 122 tool-call
blocks were involved (bash 56, read 8). The subsequent `tool/call` and
`tool/result` records reused those same ids, so the corruption was not a
transient blip — the model genuinely produced two tool-call blocks with one
id, and both were executed.

## 2. Root cause: the LLM adapter, not the session layer

**Which tool created the problem?** DSH's own LLM adapter,
`@deepseek-ai/dsh-llm-pi-ai`, which is a thin adapter **on top of the
`@earendil-works/pi-ai` library**. DSH's LLM seam (`@deepseek-ai/dsh-llm`)
is provider-neutral; it streams through registered provider adapters, each of
which owns its own wire protocol. `dsh-llm-pi-ai` is one such adapter and
delegates actual HTTP/SSE work to pi-ai.

The failing request used the **OpenAI Responses** transport inside pi-ai
(`openai-responses-shared.ts`). When the provider streams parallel
`function_call` items, pi-ai's `createSlot` builds each tool call's id as:

```ts
id: `${item.call_id}|${item.id}`
```

…with **no deduplication**. Some OpenAI-compatible providers re-issue the
last `function_call` in a response with *modified arguments* but the *same*
`(call_id, id)` pair. Both items arrive complete (each receives
`output_item.done`), so the final assistant message legitimately contains two
tool-call blocks sharing one id. That id then flows into DSH's session log,
where the validator rejects the whole log.

**How this differs from pi-ai issue #9974:** #9974 was about *unfinished*
tool calls (llama.cpp, an `output_index` that never completes); it was closed
by commit `1b2aa0ca` (shipped in pi-ai 0.99.0) which makes
`processResponsesStream` *throw* when a stream completes with an unfinished
call. Our case is the opposite: **both calls finish cleanly; only the ids
collide.** The #9974 guard does not cover it.

## 3. Fixing it upstream (pi)

The dedup belongs at id-construction time in `createSlot`: suffix repeated
ids in order of appearance (`#2`, `#3`, …) so the first occurrence stays
byte-identical and normal responses are unaffected.

- Reproduced and fixed in a local clone of `earendil-works/pi`:
  `processResponsesStream` now keeps a per-stream `Map<string, number>` and
  both the `function_call` and `custom_tool_call` branches call
  `uniqueToolCallId(...)`.
- Added a regression test
  (`openai-responses-terminal-event.test.ts`) feeding a stream with two
  `output_item.added` events carrying `function_call` items that share
  `(call_id, id)` but differ in arguments; asserts the ids become
  `call_dup|fc_dup` and `call_dup|fc_dup#2`, args stay intact, and the
  message ends with `stopReason: "toolUse"`. 15/15 pass.
- Committed `eae2af0` on branch `fix/responses-duplicate-tool-call-ids`.
- Submitted upstream via `gh` (logged in as `covrom`):
  - **Issue:** `earendil-works/pi#10396` (labels `bug`, `pkg:ai`).
  - **PR:** `earendil-works/pi#10397` (head `covrom:fix/…` → base `main`).
- Note on process: per the repo's CONTRIBUTING, issues/PRs from new
  contributors are auto-closed by default and need a maintainer `lgtm` /
  `lgtmi` to stay open. That is a process gate, not a rejection of the
  fix.
- A local report was also produced earlier at
  `reports/llm-adapter-tool-call-id-collision-report.md`.

## 4. "Can we just update the adapter?" — why not (yet)

DSH pins pi-ai tightly. `dsh@0.2.0-rc.2` → `dsh-llm-pi-ai@0.2.0-rc.2` →
`@earendil-works/pi-ai ^0.87.1`. The caret on a 0.x version means `<0.88.0`,
and *every* published `dsh-llm-pi-ai` (up to `0.2.1-alpha.1`) pins `^0.87.1`
or lower. So updating DSH alone cannot pull a fixed pi-ai, and the id-dedup
does not exist in any published pi-ai ≤ 1.0.0 (the unfinished-call throw from
0.99.0 does, but that is the wrong fix for this case). The corrupt session
also needs mechanical repair regardless of which library version is used.

## 5. "How hard is an equivalent adapter in Go?" — the answer became this repo

Estimating the effort against the real pi-ai source:

- The whole `packages/ai` is ~26 300 lines: ~30 providers, model catalog,
  OAuth, transform-messages.
- One provider (openai-responses) is ~1 250 lines, plus ~1 100 lines of shared
  types.
- A minimal adapter for a single OpenAI-compatible endpoint is ~1 500–2 500
  lines of Go, and in Go it is *easier* than pi-ai (no `fetch` gymnastics,
  SSE is trivial, channels for the event stream).

A user asked specifically for an adapter **for llama.cpp only, nothing else**,
as a new GitHub project in Go. That is exactly this repository: the `llama`
library bakes the incident's lessons (id dedup, missing `finish_reason`,
interleaved blocks, reasoning replay, verbatim args) directly into one
provider, with the OpenAI gateway added later as the DSH integration point.

## Design decisions carried from the incident

- **Dedupe at id-construction time**, deterministic `#N` suffix in order of
  appearance; first occurrence byte-identical; normal streams unaffected.
- **Do not confuse the two failure modes:** unfinished call (throw / guard)
  vs. finished-but-duplicated id (dedup). They need different handling.
- **Keep the raw wire text** (`ArgsJSON`) even when it is not valid JSON, so
  consumers can see exactly what the server emitted.
- **Collect trailing `usage` before emitting `done`** so any consumer that
  reads usage at the terminal event sees complete counts (this is the
  invariant the gateway's race detector caught and `TestUsageCompleteAtDone`
  pins).

---

## Current state / next

- Library: complete, tested, race-clean.
- Gateway: complete, tested (end-to-end duplicate-id dedup path included).
- DSH: README documents the `llama.cpp → gateway → DSH route` topology and the
  exact provider-route YAML. A live end-to-end check against a real llama.cpp
  server + a real DSH composition is the natural next verification step.
- Upstream: waiting on `earendil-works/pi#10397` (maintainer gate).
