// llama-gateway exposes an OpenAI-compatible /v1 endpoint backed by
// llama.cpp, translating it through the llama adapter. It exists so that
// harnesses whose LLM seam speaks only the OpenAI chat-completions
// protocol (for example DeepSeek Harness, which routes every request
// through a provider adapter configured with api: openai-completions and
// a baseURL) can consume a llama.cpp model. The adapter's tool-call id
// deduplication is applied transparently: if llama.cpp re-issues a call
// with the same id, the gateway emits #2, #3 suffixed ids instead.
//
// Endpoints:
//
//	POST /v1/chat/completions  (streaming SSE and non-streaming)
//	GET  /v1/models
//	GET  /healthz
//
// Configuration (env):
//
//	LLAMA_GW_UPSTREAM    llama.cpp base URL (required), e.g. http://127.0.0.1:8080
//	LLAMA_GW_MODEL       model id to serve (required); requests must name it or be empty
//	LLAMA_GW_LISTEN      listen address (default :8090)
//	LLAMA_GW_API_KEY     when set, clients must send this bearer token
package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/covrom/go-llama-adapter/llama"
)

func main() {
	upstream := os.Getenv("LLAMA_GW_UPSTREAM")
	model := os.Getenv("LLAMA_GW_MODEL")
	if upstream == "" || model == "" {
		log.Fatal("LLAMA_GW_UPSTREAM and LLAMA_GW_MODEL are required")
	}
	listen := os.Getenv("LLAMA_GW_LISTEN")
	if listen == "" {
		listen = ":8090"
	}
	apiKey := os.Getenv("LLAMA_GW_API_KEY")

	client, err := llama.New(llama.Config{BaseURL: upstream, Model: model})
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if apiKey != "" && !authorized(r, apiKey) {
			writeError(w, http.StatusUnauthorized, "invalid api key")
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, http.StatusBadRequest, "unreadable body")
			return
		}
		var req chatReq
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, "bad request: "+err.Error())
			return
		}
		if req.Model != "" && req.Model != model {
			writeError(w, http.StatusNotFound, fmt.Sprintf("model %q not served (have %q)", req.Model, model))
			return
		}
		msgs, err := decodeMessages(req.Messages)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if len(msgs) == 0 {
			writeError(w, http.StatusBadRequest, "messages is required")
			return
		}
		params, err := decodeParams(req)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if req.Stream {
			handleStream(w, r, client, model, msgs, params)
		} else {
			handleBlocking(w, r, client, model, msgs, params)
		}
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		if apiKey != "" && !authorized(r, apiKey) {
			writeError(w, http.StatusUnauthorized, "invalid api key")
			return
		}
		info, err := client.ModelInfo(r.Context())
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data":   []map[string]string{{"id": info.ID, "object": "model", "owned_by": info.OwnedBy}},
		})
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, "ok")
	})
	log.Printf("llama-gateway listening on %s (upstream %s, model %s)", listen, upstream, model)
	log.Fatal(http.ListenAndServe(listen, mux))
}

func authorized(r *http.Request, apiKey string) bool {
	return r.Header.Get("Authorization") == "Bearer "+apiKey
}

// --- request/response wire shapes (OpenAI chat completions subset) ---

type chatReq struct {
	Model               string    `json:"model"`
	Messages            []anyMsg  `json:"messages"`
	Stream              bool      `json:"stream"`
	Tools               []anyTool `json:"tools"`
	Temperature         *float64  `json:"temperature"`
	TopP                *float64  `json:"top_p"`
	MaxCompletionTokens *int      `json:"max_completion_tokens"`
	MaxTokens           *int      `json:"max_tokens"`
	// Stop is either a string or an array of strings per the OpenAI spec;
	// the concrete type is validated by decodeStop.
	Stop any `json:"stop"`
}

type anyMsg struct {
	Role       string  `json:"role"`
	Content    any     `json:"content"`
	Name       string  `json:"name"`
	ToolCalls  []anyTC `json:"tool_calls"`
	ToolCallID string  `json:"tool_call_id"`
	// ReasoningContent is the DeepSeek/llama.cpp extension used to replay
	// assistant reasoning for reasoning models.
	ReasoningContent string `json:"reasoning_content"`
}

type anyTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type anyTC struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func decodeMessages(in []anyMsg) ([]llama.Message, error) {
	out := make([]llama.Message, 0, len(in))
	for i, m := range in {
		lm := llama.Message{Role: llama.Role(m.Role), Name: m.Name, ToolCallID: m.ToolCallID, Thinking: m.ReasoningContent}
		switch c := m.Content.(type) {
		case nil:
		case string:
			lm.Content = c
		case []any:
			for _, p := range c {
				pm, _ := p.(map[string]any)
				switch pm["type"] {
				case "text":
					t, _ := pm["text"].(string)
					lm.Parts = append(lm.Parts, llama.TextPart(t))
				case "image_url":
					iu, _ := pm["image_url"].(map[string]any)
					url, _ := iu["url"].(string)
					lm.Parts = append(lm.Parts, llama.ImagePart(url))
				default:
					return nil, fmt.Errorf("message %d: unsupported content part %v", i, pm["type"])
				}
			}
		default:
			return nil, fmt.Errorf("message %d: unsupported content type", i)
		}
		for _, tc := range m.ToolCalls {
			lm.ToolCalls = append(lm.ToolCalls, llama.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				ArgsJSON:  tc.Function.Arguments,
				Arguments: json.RawMessage(tc.Function.Arguments),
			})
		}
		out = append(out, lm)
	}
	return out, nil
}

func decodeTools(in []anyTool) []llama.Tool {
	out := make([]llama.Tool, 0, len(in))
	for _, t := range in {
		out = append(out, llama.Tool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}
	return out
}

func decodeParams(req chatReq) (llama.Params, error) {
	stop, err := decodeStop(req.Stop)
	if err != nil {
		return llama.Params{}, err
	}
	p := llama.Params{
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        stop,
		Tools:       decodeTools(req.Tools),
	}
	if req.MaxCompletionTokens != nil {
		p.MaxCompletionTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		p.MaxCompletionTokens = req.MaxTokens
	}
	return p, nil
}

// decodeStop accepts the three shapes OpenAI allows for `stop`: a single
// string, an array of strings, or null/absent.
func decodeStop(v any) ([]string, error) {
	switch s := v.(type) {
	case nil:
		return nil, nil
	case string:
		if s == "" {
			return nil, nil
		}
		return []string{s}, nil
	case []any:
		out := make([]string, 0, len(s))
		for _, e := range s {
			str, ok := e.(string)
			if !ok {
				return nil, errors.New("stop must be a string or an array of strings")
			}
			out = append(out, str)
		}
		if len(out) == 0 {
			return nil, nil
		}
		return out, nil
	default:
		return nil, errors.New("stop must be a string or an array of strings")
	}
}

func chatID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "chatcmpl-gw-0000000000000000"
	}
	return "chatcmpl-gw-" + hex.EncodeToString(b[:])
}

// --- SSE relay ---

// SSE wire shapes
type wireDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []map[string]any `json:"tool_calls,omitempty"`
}

type sseChoice struct {
	Index        int       `json:"index"`
	Delta        wireDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason,omitempty"`
}

type sseChunk struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []sseChoice  `json:"choices"`
	Usage   *llama.Usage `json:"usage,omitempty"`
}

// handleStream relays the adapter event stream as OpenAI SSE.
//
// Mapping rules:
//   - every chunk carries the OpenAI envelope (object "chat.completion.chunk",
//     created, id, model) because SDK streaming helpers filter on the object
//     value and accumulate created;
//   - the assistant role is emitted once, on the first content delta;
//   - text and reasoning deltas map to content / reasoning_content;
//   - toolcall_start carries the (deduplicated) id and name on the
//     first tool_call delta of the slot, argument fragments follow on
//     later deltas of the same slot index;
//   - done emits a finish_reason chunk followed by a usage chunk with an
//     empty choices array and [DONE]; error emits one error chunk and
//     terminates the stream.
func handleStream(w http.ResponseWriter, r *http.Request, c *llama.Client, model string, msgs []llama.Message, p llama.Params) {
	stream, err := c.Complete(r.Context(), msgs, p)
	if err != nil {
		if he, ok := err.(*llama.HTTPError); ok {
			writeError(w, he.Status, he.Body)
			return
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	id := chatID()
	created := time.Now().Unix()
	emit := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if fl != nil {
			fl.Flush()
		}
	}
	chunk := func(choices []sseChoice) sseChunk {
		return sseChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: choices,
		}
	}

	roleSent := false
	roleFor := func() string {
		if roleSent {
			return ""
		}
		roleSent = true
		return "assistant"
	}

	for ev := range stream.Ch {
		switch ev.Type {
		case llama.EventTextDelta:
			emit(chunk([]sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), Content: ev.Delta},
			}}))
		case llama.EventThinkingDelta:
			// Consumers that do not know reasoning_content ignore it;
			// llama.cpp-aware clients use it for thinking replay.
			emit(chunk([]sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), ReasoningContent: ev.Delta},
			}}))
		case llama.EventToolCallStart:
			tc := map[string]any{"index": ev.ContentIndex}
			if ev.ToolCall != nil {
				if ev.ToolCall.ID != "" {
					tc["id"] = ev.ToolCall.ID
				}
				if ev.ToolCall.Name != "" {
					tc["function"] = map[string]any{"name": ev.ToolCall.Name}
				}
				// The OpenAI wire marks the call type on the first delta of
				// the slot; SDKs reject a tool call whose type is missing.
				if ev.ToolCall.ID != "" || ev.ToolCall.Name != "" {
					tc["type"] = "function"
				}
			}
			if len(tc) > 1 {
				emit(chunk([]sseChoice{{
					Index: 0,
					Delta: wireDelta{Role: roleFor(), ToolCalls: []map[string]any{tc}},
				}}))
			}
		case llama.EventToolCallDelta:
			emit(chunk([]sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), ToolCalls: []map[string]any{{
					"index":    ev.ContentIndex,
					"function": map[string]any{"arguments": ev.Delta},
				}}},
			}}))
		case llama.EventDone:
			if !roleSent {
				// The stream carried no content (e.g. empty completion).
				emit(chunk([]sseChoice{{
					Index: 0,
					Delta: wireDelta{Role: "assistant"},
				}}))
			}
			fr := map[llama.StopReason]string{
				llama.StopStop:          "stop",
				llama.StopLength:        "length",
				llama.StopToolUse:       "tool_calls",
				llama.StopContentFilter: "content_filter",
			}[ev.Reason]
			if fr == "" {
				fr = "stop"
			}
			emit(chunk([]sseChoice{{
				Index:        0,
				Delta:        wireDelta{},
				FinishReason: &fr,
			}}))
			if ev.Partial != nil && ev.Partial.Usage.TotalTokens != 0 {
				u := ev.Partial.Usage
				// OpenAI sends the usage-only chunk with an empty (never
				// absent or null) choices array.
				usageChunk := chunk([]sseChoice{})
				usageChunk.Usage = &u
				emit(usageChunk)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			if fl != nil {
				fl.Flush()
			}
			return
		case llama.EventError:
			emit(map[string]any{
				"id":    id,
				"error": map[string]string{"message": errText(ev.Err), "type": "server_error"},
			})
			return
		}
	}
}

func errText(err error) string {
	if err == nil {
		return "stream error"
	}
	return err.Error()
}

// --- blocking (non-streaming) path ---

func handleBlocking(w http.ResponseWriter, r *http.Request, c *llama.Client, model string, msgs []llama.Message, p llama.Params) {
	stream, err := c.Complete(r.Context(), msgs, p)
	if err != nil {
		if he, ok := err.(*llama.HTTPError); ok {
			writeError(w, he.Status, he.Body)
			return
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	msg, err := stream.Wait()
	if err != nil {
		writeError(w, http.StatusBadGateway, errText(err))
		return
	}
	message := map[string]any{"role": "assistant", "content": msg.Text()}
	fr := "stop"
	switch msg.Stop {
	case llama.StopLength:
		fr = "length"
	case llama.StopContentFilter:
		fr = "content_filter"
	}
	// Tool calls are reported whenever the message contains any, not only
	// when the server said finish_reason=tool_calls: a server that streams
	// calls and then stops with "stop" must not have them silently dropped.
	if calls := msg.ToolCalls(); len(calls) > 0 {
		if fr == "stop" {
			fr = "tool_calls"
		}
		tcOut := make([]map[string]any, 0, len(calls))
		for _, tc := range calls {
			tcOut = append(tcOut, map[string]any{
				"id":   tc.ID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Name,
					"arguments": tc.ArgsJSON,
				},
			})
		}
		message["tool_calls"] = tcOut
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":      chatID(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       message,
			"finish_reason": fr,
		}},
		"usage": msg.Usage,
	})
}

func writeError(w http.ResponseWriter, status int, msg string) {
	// OpenAI reports client mistakes as invalid_request_error and upstream
	// failures as server_error; keep the distinction so SDK-side retry and
	// classification logic sees the right kind.
	errType := "invalid_request_error"
	if status >= http.StatusInternalServerError {
		errType = "server_error"
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": strings.TrimSpace(msg),
			"type":    errType,
			"param":   nil,
			"code":    nil,
		},
	})
}
