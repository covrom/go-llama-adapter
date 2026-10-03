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
		if req.Stream {
			handleStream(w, r, client, model, msgs, req)
		} else {
			handleBlocking(w, r, client, model, msgs, req)
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
	MaxCompletionTokens *int      `json:"max_completion_tokens"`
	MaxTokens           *int      `json:"max_tokens"`
}

type anyMsg struct {
	Role       string  `json:"role"`
	Content    any     `json:"content"`
	Name       string  `json:"name"`
	ToolCalls  []anyTC `json:"tool_calls"`
	ToolCallID string  `json:"tool_call_id"`
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
		lm := llama.Message{Role: llama.Role(m.Role), Name: m.Name, ToolCallID: m.ToolCallID}
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

func decodeParams(req chatReq) llama.Params {
	p := llama.Params{
		Temperature: req.Temperature,
		Tools:       decodeTools(req.Tools),
	}
	if req.MaxCompletionTokens != nil {
		p.MaxCompletionTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		p.MaxCompletionTokens = req.MaxTokens
	}
	return p
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
	Model   string       `json:"model"`
	Choices []sseChoice  `json:"choices"`
	Usage   *llama.Usage `json:"usage,omitempty"`
}

// handleStream relays the adapter event stream as OpenAI SSE.
//
// Mapping rules:
//   - the assistant role is emitted once, on the first content delta;
//   - text and reasoning deltas map to content / reasoning_content;
//   - toolcall_start carries the (deduplicated) id and name on the
//     first tool_call delta of the slot, argument fragments follow on
//     later deltas of the same slot index;
//   - done emits a finish_reason chunk followed by a usage chunk and
//     [DONE]; error emits one error chunk and terminates the stream.
func handleStream(w http.ResponseWriter, r *http.Request, c *llama.Client, model string, msgs []llama.Message, req chatReq) {
	stream, err := c.Complete(r.Context(), msgs, decodeParams(req))
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
	emit := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if fl != nil {
			fl.Flush()
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
			emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), Content: ev.Delta},
			}}})
		case llama.EventThinkingDelta:
			// Consumers that do not know reasoning_content ignore it;
			// llama.cpp-aware clients use it for thinking replay.
			emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), ReasoningContent: ev.Delta},
			}}})
		case llama.EventToolCallStart:
			tc := map[string]any{"index": ev.ContentIndex}
			if ev.ToolCall != nil {
				if ev.ToolCall.ID != "" {
					tc["id"] = ev.ToolCall.ID
					tc["type"] = "function"
				}
				if ev.ToolCall.Name != "" {
					tc["function"] = map[string]any{"name": ev.ToolCall.Name}
				}
			}
			if len(tc) > 1 {
				emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
					Index: 0,
					Delta: wireDelta{Role: roleFor(), ToolCalls: []map[string]any{tc}},
				}}})
			}
		case llama.EventToolCallDelta:
			emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
				Index: 0,
				Delta: wireDelta{Role: roleFor(), ToolCalls: []map[string]any{{
					"index":    ev.ContentIndex,
					"function": map[string]any{"arguments": ev.Delta},
				}}},
			}}})
		case llama.EventDone:
			if !roleSent {
				// The stream carried no content (e.g. empty completion).
				emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
					Index: 0,
					Delta: wireDelta{Role: "assistant"},
				}}})
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
			emit(sseChunk{ID: id, Model: model, Choices: []sseChoice{{
				Index:        0,
				Delta:        wireDelta{},
				FinishReason: &fr,
			}}})
			if ev.Partial != nil && ev.Partial.Usage.TotalTokens != 0 {
				u := ev.Partial.Usage
				emit(sseChunk{ID: id, Model: model, Usage: &u})
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

func handleBlocking(w http.ResponseWriter, r *http.Request, c *llama.Client, model string, msgs []llama.Message, req chatReq) {
	stream, err := c.Complete(r.Context(), msgs, decodeParams(req))
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
	case llama.StopToolUse:
		fr = "tool_calls"
		tcOut := []map[string]any{}
		for _, tc := range msg.ToolCalls() {
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
	case llama.StopLength:
		fr = "length"
	case llama.StopContentFilter:
		fr = "content_filter"
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": strings.TrimSpace(msg),
			"type":    "invalid_request_error",
		},
	})
}
