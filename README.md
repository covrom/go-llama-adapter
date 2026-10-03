# go-llama-adapter

A minimal LLM adapter for the [llama.cpp](https://github.com/ggml-org/llama.cpp) server, written in Go.

It speaks the OpenAI-compatible chat completions API that the llama.cpp server exposes (`POST /v1/chat/completions`, SSE streaming) and normalizes the response into a small, stable event protocol modeled after the `openai-completions` adapter of [pi-ai](https://github.com/earendil-works/pi).

**Scope: llama.cpp only.** No provider registry, no OAuth, no multi-API dispatch, no external dependencies (standard library only). If you need other backends, that is a different project.

## Why this exists

LLM harnesses that stream tool calls from OpenAI-compatible servers have to handle provider quirks carefully. This adapter bakes the lessons learned from a real corruption incident in one place:

- **Duplicate tool call ids.** Some servers re-issue the last function call with modified arguments but the *same* id. Naive adapters then emit two tool call blocks with identical ids, which breaks session-log validators that require unique ids. This adapter suffixes repeats in order of appearance: `call_x`, `call_x#2`, `call_x#3`. First occurrence stays byte-identical; normal streams are unaffected.
- **Missing `finish_reason`.** Older servers end the stream without it; the adapter infers `tool_calls`/`stop` from what actually streamed.
- **Interleaved blocks.** Text → tool call → text produces separate text blocks with correctly paired `start`/`end` events.
- **Reasoning replay.** `reasoning_content` is captured into thinking blocks and can be echoed back on assistant turns so reasoning models can continue chains of thought.
- **Verbatim argument replay.** Assistant tool call arguments are replayed exactly as streamed — never re-serialized.

## Requirements

- Go 1.27+
- A running llama.cpp server, e.g.
  `./llama-server -m model.gguf --port 8080` (default OpenAI endpoint at `/v1`)

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/covrom/go-llama-adapter/llama"
)

func main() {
    client, err := llama.New(llama.Config{
        BaseURL: "http://127.0.0.1:8080",
        Model:   "qwen3",
    })
    if err != nil {
        log.Fatal(err)
    }

    stream, err := client.Complete(context.Background(), []llama.Message{
        llama.SystemMessage("You are a helpful assistant."),
        llama.UserMessage("List the files in the current directory."),
    }, llama.Params{
        Tools: []llama.Tool{{
            Name:        "bash",
            Description: "Run a shell command.",
            Parameters:  []byte(`{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}`),
        }},
    })
    if err != nil {
        log.Fatal(err)
    }

    for ev := range stream.Ch {
        switch ev.Type {
        case llama.EventTextDelta:
            fmt.Print(ev.Delta)
        case llama.EventDone:
            msg := ev.Partial
            fmt.Println("\n-- done:", msg.Stop)
            for _, call := range msg.ToolCalls() {
                fmt.Printf("tool %s: %s args=%s\n", call.ID, call.Name, call.ArgsJSON)
            }
        case llama.EventError:
            log.Fatal(ev.Err)
        }
    }
}
```

Or consume the whole stream at once with `msg, err := stream.Wait()`.

## API overview

| Type | Purpose |
| --- | --- |
| `Client`, `Config` | One llama.cpp server + one model. `New` validates config. |
| `Client.Complete(ctx, msgs, params)` | Start a streaming completion. Setup errors return immediately; stream errors arrive as a terminal `error` event. |
| `Client.ModelInfo(ctx)` | Look up the served model via `GET /v1/models`. |
| `Message` (+ `SystemMessage`, `UserMessage`, `UserMessageParts`, `AssistantWithToolCalls`, `ToolMessage`) | Normalized conversation messages, including multimodal user parts and tool results. |
| `AssistantMessage` | Accumulated result: ordered `Blocks` (text / thinking / toolcall), `Usage`, `Stop`. |
| `Stream` / `Event` | Channel of events: `start`, `text_start/delta/end`, `thinking_start/delta/end`, `toolcall_start/delta/end`, `done`, `error`. `Event.Partial` is a live shared message-so-far pointer. |
| `Params` | Sampling params + tools; zero fields are omitted. |
| `Params.Compat` | Knobs for specific server builds: `DisableUsageInStream`, `MaxTokensField`. |

### Event protocol notes

- Events always start with `start` and end with exactly one of `done` / `error`.
- `ContentIndex` refers to `Partial.Blocks` indices.
- `toolcall_delta` carries raw argument fragments; the final call (with `ID`, `Name`, `ArgsJSON`) arrives in `toolcall_end`.
- `done.Reason` is one of `stop`, `length`, `toolUse`, `contentFilter`; `error.Reason` is `error` or `aborted` (context cancellation).

## Known limitations

- One request at a time per `Stream`; the client itself is safe for concurrent `Complete` calls.
- No prompt caching, no image *generation*, no classifier endpoint — deliberately out of scope.
- Malformed JSON in streamed tool call arguments is preserved raw (in `ArgsJSON`) rather than rejected.

## Tests

```sh
go test ./...
```

The test suite runs against an in-process `httptest` mock of the llama.cpp endpoint, covering: plain text, tool call streaming, duplicate-id deduplication, reasoning blocks, interleaved text/tool/text, server-side errors, HTTP errors, missing `finish_reason`, and exact request wire format (verbatim argument replay, `reasoning_content`, image parts, compat fields).
