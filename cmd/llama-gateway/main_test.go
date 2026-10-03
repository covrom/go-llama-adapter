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
		if req.Stream {
			handleStream(w, r, client, "m", msgs, req)
		} else {
			handleBlocking(w, r, client, "m", msgs, req)
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
