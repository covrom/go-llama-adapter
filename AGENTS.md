# AGENTS.md — go-llama-adapter

Guidance for AI agents (and humans) working in this repository. Read this
first; it complements `PROGRESS.md` (history) and `README.md` (user-facing
docs).

## What this project is

A minimal LLM adapter for the **llama.cpp server** (OpenAI-compatible
`/v1/chat/completions` + SSE), written in Go. Two deliverables:

- `llama/` — the adapter **library**: HTTP client, SSE parser, and a small
  normalized event protocol.
- `cmd/llama-gateway/` — an **OpenAI-compatible gateway** on top of the
  library, so harnesses that only speak the OpenAI chat-completions protocol
  (notably DeepSeek Harness, via a `dsh-llm-pi-ai` route with
  `api: openai-completions`) can reach a llama.cpp model.

**Hard scope constraint: llama.cpp only.** Do not add a provider registry,
OAuth, model catalog, image generation, prompt caching, or any other backend.
If a need appears that implies "more providers", the answer is "out of scope",
not "one more abstraction". This is deliberate: the original motivation is one
provider's quirks handled correctly, not a framework.

## Technology

- **Go 1.27+**, standard library **only**. There is no `require` in `go.mod`
  on purpose. Do not add third-party dependencies; if a task seems to demand
  one, stop and reconsider the design (the project is small enough that stdlib
  always covers it).
- Concurrency model: one `Stream` per request; events flow over a buffered
  channel from a single producer goroutine. The producer is the only writer
  to the accumulated `AssistantMessage`; consumers observe it through
  `Event.Partial` (shared pointer). **Any new shared state between the
  producer goroutine and event consumers must be proven race-free** — run
  `go test -race` before considering work done. This project has already been
  bitten once by this exact pattern (usage chunk after `finish_reason`); see
  `TestUsageCompleteAtDone`.
- Wire protocol knowledge lives in exactly two places:
  - **inbound** (llama.cpp → us): `llama/stream.go` (`wireChunk`, `delta`,
    `chunkToolCallDelta`) and `llama/client.go` (request encoding).
  - **outbound** (us → OpenAI clients): `cmd/llama-gateway/main.go`
    (`sseChunk`, `wireDelta`, `anyMsg`…).
  Keep them separate; do not share types across the gateway boundary. The
  gateway is a protocol translator, not a passthrough.

## Core invariants (do not regress)

These are the invariants the project exists to guarantee. Each has a named
test; if you touch the affected code, the test must keep passing.

1. **Tool-call ids are unique per assistant message.** If the server re-uses
   an id, later occurrences are suffixed `#2`, `#3`, … in order of
   appearance; the first occurrence is byte-identical.
   Test: `TestDuplicateToolCallIDs` (library),
   `TestGatewayStreamDuplicateToolCallIDs` (gateway).
2. **The deduplicated id is known at `toolcall_start`**, not only at
   `toolcall_end` — the gateway relies on this to emit id/name in the first
   tool_call delta of an OpenAI stream.
3. **`done` is emitted only after all writes to `AssistantMessage` are
   complete** — including trailing `usage` that arrives after
   `finish_reason`. A consumer reading `Partial` at `done` sees the final
   state. Test: `TestUsageCompleteAtDone`.
4. **Block start/end events are always paired**, including for interleaved
   text → tool → text streams. Test: `TestInterleavedTextAfterToolCall`.
5. **Tool-call arguments are replayed verbatim** (`ArgsJSON` is never
   re-serialized). Malformed JSON is preserved raw, not rejected.
   Test: `TestRequestWireFormat`.
6. **Missing `finish_reason` is inferred** (`tool_calls` if any tool slot was
   opened, else `stop`) — inference must not read unfinalized blocks.
   Tests: `TestMissingFinishReasonInferred`,
   `TestMissingFinishReasonToolCallsInferred`.
7. **Every stream ends with exactly one terminal event**: `done` or `error`.
   Context cancellation maps to `error` with `Reason: aborted`.

## Coding conventions

- `gofmt`-clean (the formatter is run on every edit; keep it that way).
- `go vet`-clean.
- Errors: wrap with `fmt.Errorf("llama: …: %w", err)` at library boundaries;
  the gateway maps transport errors to HTTP status codes (4xx for client
  mistakes, 502 for upstream failures). HTTP failures surface as
  `*HTTPError` with the status code preserved.
- Comments: explain **why** (protocol quirks, invariants, ordering
  constraints), not what the code obviously does. No AI-slop filler
  paragraphs, no restated function names.
- Naming follows stdlib conventions; the package is named `llama` and
  documents itself as an adapter, not a framework.
- No panics in library code; the one allowed fatal path is in `main()`.

## Testing approach

- All tests run against in-process `httptest` mocks of the llama.cpp
  endpoint. **No test may require a real server, network access, or a model
  file.** If you add behavior that is only reachable against a real server,
  add a mock fixture for the relevant SSE sequence instead.
- SSE fixtures are plain string slices of complete lines (`data: …`), which
  makes wire-level bugs (escaping, chunk framing) visible in diffs.
- Build SSE JSON fixtures carefully: inner JSON strings must be escaped
  (`{\"cmd\":\"ls\"}` inside a raw string literal). A malformed fixture fails
  with a JSON parse error from the *adapter*, not the test — if you see
  "bad stream chunk", suspect the fixture first.
- New behavior requires a test. Bug fixes require a **regression test named
  after the bug** (the duplicate-id tests are the template).
- Coverage is a signal, not a goal: `llama/` sits around 85%; the gateway is
  lower because `main()` wiring is not exercised — that is acceptable, do not
  chase it.

### Required pre-push checks

```sh
gofmt -l .            # must print nothing
go build ./...
go vet ./...
go test -race -count=1 ./...
```

All four must be clean before a commit.

## Working with PROGRESS.md

`PROGRESS.md` is the project's narrative log (incident → decisions → work).
Rules:

- **Append new dated entries at the top of the "log" section** (above the
  oldest entry, below the header). Newest first.
- One short entry per meaningful unit of work: what changed, why, and which
  tests/commits cover it. Factual, no marketing language.
- Do **not** rewrite or delete historical entries; the origin section
  ("Origin of the project") is immutable history.
- An entry describes work that is committed; if you write an entry, commit
  the work in the same or immediately following commit so git and the log
  stay in sync.
- When work changes a core invariant, update both the invariant's test and
  the "Design decisions carried from the incident" list.
- Do not use PROGRESS.md as a todo list; keep it a record of what *is*.

## Repository layout

```
llama/
  types.go       normalized message/event types
  events.go      event protocol + Stream
  client.go      Client, request encoding, ModelInfo
  stream.go      SSE parser + accumulator (the heart of the adapter)
  params.go      per-request params + Compat
  stream_test.go library tests (httptest mock upstream)
cmd/llama-gateway/
  main.go        OpenAI-compatible gateway (env-configured, stdlib only)
  main_test.go   end-to-end gateway tests (mock upstream, real HTTP client)
```

## Integration boundary (DSH)

DeepSeek Harness cannot mount this Go code as a plugin (its LLM seam is a
Node/TS plugin system); integration is over HTTP: llama.cpp →
`llama-gateway` → DSH provider route (`api: openai-completions`). The
gateway must therefore behave like a well-behaved OpenAI server: correct
first-delta role, `finish_reason` before the usage chunk, `[DONE]`
terminated streams, and stable error chunks. See the README section
"Connecting to DeepSeek Harness (DSH)" for the exact route YAML.
