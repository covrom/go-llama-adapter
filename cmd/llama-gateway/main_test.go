package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/covrom/go-llama-adapter/llama"
)

// upstream returns a llama.cpp-style SSE stream. The duplicateID case makes
// the server re-issue the same tool call id twice, to exercise dedup through
// the whole gateway.
func upstream(t *testing.T, duplicateID bool) *httptest.Server {
	t.Helper()
	chunk := func(delta string, finish *string) string {
		fr := "null"
		if finish != nil {
			fr = fmt.Sprintf("%q", *finish)
		}
		return "data: " + fmt.Sprintf(
			`{"id":"up","model":"m","choices":[{"index":0,"delta":%s,"finish_reason":%s}]}`,
			delta, fr,
		)
	}
	var lines []string
	if duplicateID {
		lines = append(lines,
			chunk(`{"role":"assistant"}`, nil),
			`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_dup","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}`,
			`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_dup","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls -R\"}"}}]},"finish_reason":null}]}`,
			chunk(`{}`, strPtr("tool_calls")),
			"data: [DONE]",
		)
	} else {
		lines = append(lines,
			chunk(`{"role":"assistant","content":"Hello"}`, nil),
			chunk(`{"content":" world"}`, nil),
			chunk(`{}`, strPtr("stop")),
			`data: {"id":"up","model":"m","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			"data: [DONE]",
		)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"data":[{"id":"m","object":"model","owned_by":"local"}]}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			for _, l := range lines {
				fmt.Fprintln(w, l)
				if fl != nil {
					fl.Flush()
				}
			}
		default:
			http.NotFound(w, r)
		}
	}))
}

func strPtr(s string) *string { return &s }

// startGateway wires the gateway handlers to the given upstream model name.
func startGateway(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	client, err := llama.New(llama.Config{BaseURL: upstreamURL, Model: "m"})
	if err != nil {
		t.Fatalf("newClient: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req chatReq
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		msgs, err := decodeMessages(req.Messages)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if len(msgs) == 0 {
			writeError(w, http.StatusBadRequest, "messages required")
			return
		}
		params, err := decodeParams(req)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if req.Stream {
			handleStream(w, r, client, "m", msgs, params)
		} else {
			handleBlocking(w, r, client, "m", msgs, params)
		}
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
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
	return httptest.NewServer(mux)
}

func postChat(t *testing.T, gwURL string, body any) (*http.Response, string) {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, gwURL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	return res, string(raw)
}

func TestGatewayStreamText(t *testing.T) {
	up := upstream(t, false)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	res, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", res.StatusCode, raw)
	}
	if !strings.Contains(raw, `"content":"Hello"`) || !strings.Contains(raw, `"content":" world"`) {
		t.Fatalf("text not relayed:\n%s", raw)
	}
	if !strings.Contains(raw, `"finish_reason":"stop"`) {
		t.Fatalf("finish_reason missing:\n%s", raw)
	}
	if !strings.Contains(raw, `"total_tokens":3`) {
		t.Fatalf("usage missing:\n%s", raw)
	}
	if !strings.HasSuffix(strings.TrimSpace(raw), "data: [DONE]") {
		t.Fatalf("must end with [DONE]:\n%s", raw)
	}
}

func TestGatewayStreamDuplicateToolCallIDs(t *testing.T) {
	up := upstream(t, true)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	_, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "list"}},
	})
	if !strings.Contains(raw, `"id":"call_dup"`) {
		t.Fatalf("first id missing:\n%s", raw)
	}
	if !strings.Contains(raw, `"id":"call_dup#2"`) {
		t.Fatalf("deduped id missing:\n%s", raw)
	}
	if !strings.Contains(raw, `"finish_reason":"tool_calls"`) {
		t.Fatalf("finish_reason missing:\n%s", raw)
	}
	// Both argument payloads must be present, each on its own slot. In raw
	// SSE the arguments are a nested JSON string, so inner quotes are escaped.
	if !strings.Contains(raw, `{\"cmd\":\"ls\"}`) || !strings.Contains(raw, `{\"cmd\":\"ls -R\"}`) {
		t.Fatalf("arguments missing:\n%s", raw)
	}
}

func TestGatewayBlocking(t *testing.T) {
	up := upstream(t, false)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	res, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   false,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", res.StatusCode, raw)
	}
	var body struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatalf("decode: %v\n%s", err, raw)
	}
	if len(body.Choices) != 1 || body.Choices[0].Message.Content != "Hello world" {
		t.Fatalf("body = %+v", body)
	}
	if body.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish = %q", body.Choices[0].FinishReason)
	}
	if body.Usage.TotalTokens != 3 {
		t.Fatalf("usage = %+v", body.Usage)
	}
}

func TestGatewayModelsAndAuth(t *testing.T) {
	up := upstream(t, false)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	res, err := http.Get(gw.URL + "/v1/models")
	if err != nil {
		t.Fatalf("get models: %v", err)
	}
	raw, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusOK || !strings.Contains(string(raw), `"id":"m"`) {
		t.Fatalf("models = %d %s", res.StatusCode, raw)
	}
}

// rawUpstream serves arbitrary SSE lines and optionally records the request
// body, so tests can pin wire shapes the shared fixture does not produce.
func rawUpstream(t *testing.T, lines []string, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if captured != nil {
			var m map[string]any
			if err := json.Unmarshal(body, &m); err != nil {
				t.Errorf("bad request json: %v", err)
			}
			*captured = m
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for _, l := range lines {
			fmt.Fprintln(w, l)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
}

// dataChunks decodes every SSE data line except the [DONE] sentinel.
func dataChunks(t *testing.T, raw string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(payload), &m); err != nil {
			t.Fatalf("chunk is not JSON: %v\n%s", err, line)
		}
		out = append(out, m)
	}
	return out
}

func streamChat(t *testing.T, gwURL string) string {
	t.Helper()
	_, raw := postChat(t, gwURL, map[string]any{
		"model":    "m",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	return raw
}

// TestGatewayStreamChunkEnvelope pins the OpenAI chunk envelope. SDK
// streaming helpers filter on it — openai-python's .stream() skips every
// chunk whose object is not "chat.completion.chunk" — so a missing envelope
// silently empties the stream. The usage-only chunk must carry an empty
// choices array (never null/absent), and it must be the last chunk before
// [DONE].
func TestGatewayStreamChunkEnvelope(t *testing.T) {
	up := upstream(t, false)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	raw := streamChat(t, gw.URL)
	chunks := dataChunks(t, raw)
	if len(chunks) == 0 {
		t.Fatalf("no chunks:\n%s", raw)
	}
	usageChunks := 0
	for _, c := range chunks {
		if c["object"] != "chat.completion.chunk" {
			t.Errorf("object = %v (want chat.completion.chunk): %v", c["object"], c)
		}
		if created, ok := c["created"].(float64); !ok || created <= 0 {
			t.Errorf("created = %v (want a positive unix timestamp): %v", c["created"], c)
		}
		if c["model"] != "m" {
			t.Errorf("model = %v", c["model"])
		}
		if id, _ := c["id"].(string); !strings.HasPrefix(id, "chatcmpl-gw-") {
			t.Errorf("id = %v", c["id"])
		}
		choices, ok := c["choices"].([]any)
		if !ok {
			t.Fatalf("choices is %T (%v), want an array", c["choices"], c["choices"])
		}
		if c["usage"] != nil {
			usageChunks++
			if len(choices) != 0 {
				t.Errorf("usage chunk choices = %v, want []", choices)
			}
		}
	}
	if usageChunks != 1 {
		t.Errorf("usage chunks = %d, want 1", usageChunks)
	}
	if last := chunks[len(chunks)-1]; last["usage"] == nil {
		t.Errorf("last chunk before [DONE] = %v, want the usage chunk", last)
	}
}

// TestGatewayStreamToolCallNameAfterID reproduces a server that sends the
// tool-call id in one chunk and the function name with the first argument
// fragment in the next. The name must reach the client on the first delta of
// the slot: OpenAI SDKs reject a finished tool call that has no
// function.name (openai-node throws "missing ...function.name").
func TestGatewayStreamToolCallNameAfterID(t *testing.T) {
	lines := []string{
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function"}]},"finish_reason":null}]}`,
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}`,
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"data: [DONE]",
	}
	up := rawUpstream(t, lines, nil)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	raw := streamChat(t, gw.URL)
	var firstToolCall map[string]any
	for _, c := range dataChunks(t, raw) {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]any)
		delta, _ := choice["delta"].(map[string]any)
		tcs, _ := delta["tool_calls"].([]any)
		if len(tcs) > 0 && firstToolCall == nil {
			firstToolCall, _ = tcs[0].(map[string]any)
		}
	}
	if firstToolCall == nil {
		t.Fatalf("no tool_call delta:\n%s", raw)
	}
	if firstToolCall["id"] != "call_1" || firstToolCall["type"] != "function" {
		t.Fatalf("first tool_call delta = %v", firstToolCall)
	}
	fn, _ := firstToolCall["function"].(map[string]any)
	if fn["name"] != "bash" {
		t.Fatalf("first tool_call delta lost the function name: %v\n%s", firstToolCall, raw)
	}
	if !strings.Contains(raw, `{\"cmd\":\"ls\"}`) {
		t.Fatalf("arguments missing:\n%s", raw)
	}
}

// TestGatewayBlockingToolCallsWithStopFinish covers a server that streams a
// tool call but closes with finish_reason "stop": the tool call must still be
// reported (as finish_reason "tool_calls"), not silently dropped.
func TestGatewayBlockingToolCallsWithStopFinish(t *testing.T) {
	lines := []string{
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]},"finish_reason":"stop"}]}`,
		"data: [DONE]",
	}
	up := rawUpstream(t, lines, nil)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	res, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   false,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", res.StatusCode, raw)
	}
	var body struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatalf("decode: %v\n%s", err, raw)
	}
	if len(body.Choices) != 1 || len(body.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("tool calls dropped: %s", raw)
	}
	if body.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", body.Choices[0].FinishReason)
	}
	if body.Choices[0].Message.ToolCalls[0].Function.Name != "bash" {
		t.Fatalf("tool call = %+v", body.Choices[0].Message.ToolCalls[0])
	}
}

// TestGatewayRequestParams pins pass-through of the request fields the
// gateway accepts, including the DeepSeek reasoning_content replay.
func TestGatewayRequestParams(t *testing.T) {
	lines := []string{
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		"data: [DONE]",
	}
	var capReq map[string]any
	up := rawUpstream(t, lines, &capReq)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	_, raw := postChat(t, gw.URL, map[string]any{
		"model":  "m",
		"stream": true,
		"top_p":  0.9,
		"stop":   []string{"END"},
		"messages": []map[string]any{
			{"role": "assistant", "content": "", "reasoning_content": "chain"},
			{"role": "user", "content": "hi"},
		},
	})
	if capReq["top_p"] != 0.9 {
		t.Errorf("top_p = %v (want 0.9)", capReq["top_p"])
	}
	stops, _ := capReq["stop"].([]any)
	if len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop = %v (want [END])", capReq["stop"])
	}
	msgs, _ := capReq["messages"].([]any)
	if len(msgs) == 0 {
		t.Fatalf("no messages upstream:\n%s", raw)
	}
	m0, _ := msgs[0].(map[string]any)
	if m0["reasoning_content"] != "chain" {
		t.Errorf("reasoning_content = %v (want chain)", m0["reasoning_content"])
	}
}

// TestGatewayStopFormsAndValidation accepts a bare string (OpenAI allows it)
// and rejects the types OpenAI rejects.
func TestGatewayStopFormsAndValidation(t *testing.T) {
	lines := []string{
		`data: {"id":"up","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		"data: [DONE]",
	}
	var capReq map[string]any
	up := rawUpstream(t, lines, &capReq)
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	_, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   true,
		"stop":     "END",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	stops, _ := capReq["stop"].([]any)
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("string stop = %v (want [END]): %s", capReq["stop"], raw)
	}

	res, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   true,
		"stop":     5,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("numeric stop: status = %d, want 400: %s", res.StatusCode, raw)
	}
	if !strings.Contains(raw, "stop must be") {
		t.Fatalf("numeric stop: unexpected error body: %s", raw)
	}
}

// TestGatewayErrorType checks that upstream failures are not reported as
// invalid_request_error, so SDK retry/classification sees a server error.
func TestGatewayErrorType(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"boom"}`, http.StatusInternalServerError)
	}))
	defer up.Close()
	gw := startGateway(t, up.URL)
	defer gw.Close()

	res, raw := postChat(t, gw.URL, map[string]any{
		"model":    "m",
		"stream":   false,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if res.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d: %s", res.StatusCode, raw)
	}
	if !strings.Contains(raw, `"type":"server_error"`) {
		t.Fatalf("error type = %s", raw)
	}
}
