package llama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sseServer starts a mock llama.cpp server that records the request body
// and replies with the given SSE lines (each line is a complete "data: ..."
// or comment line).
func sseServer(t *testing.T, lines []string, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"data":[{"id":"test-model","object":"model","owned_by":"local"}]}`)
			return
		case "/v1/chat/completions":
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
			for _, line := range lines {
				fmt.Fprintln(w, line)
				if fl != nil {
					fl.Flush()
				}
			}
			return
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
}

// chunk builds a data line: {"id","model","choices":[{"index":0,EXTRA}]SUFFIX}
// where EXTRA is a JSON fragment like `"delta":{...}` and SUFFIX is an
// optional top-level addition like `,"usage":{...}`.
func chunk(extra, suffix string) string {
	body := `{"id":"chatcmpl-1","model":"test-model","choices":[{"index":0,` + extra + `}]` + suffix + `}`
	return "data: " + body
}

func newClient(t *testing.T, url string) *Client {
	t.Helper()
	c, err := New(Config{BaseURL: url, Model: "test-model"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func run(t *testing.T, c *Client, msgs []Message, p Params) (*AssistantMessage, []Event, error) {
	t.Helper()
	s, err := c.Complete(context.Background(), msgs, p)
	if err != nil {
		return nil, nil, err
	}
	var events []Event
	for ev := range s.Ch {
		events = append(events, ev)
	}
	msg, err := s.Wait()
	return msg, events, err
}

func TestSimpleTextStream(t *testing.T) {
	lines := []string{
		": keep-alive comment",
		chunk(`"delta":{"role":"assistant","content":"Hel"}`, ""),
		chunk(`"delta":{"content":"lo "}`, ""),
		chunk(`"delta":{"content":"world"}`, ""),
		chunk(`"delta":{},"finish_reason":"stop"`, ""),
		chunk(`"delta":{}`, `,"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}`),
		"data: [DONE]",
	}
	var capReq map[string]any
	srv := sseServer(t, lines, &capReq)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, events, err := run(t, c, []Message{UserMessage("hi")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if msg.Text() != "Hello world" {
		t.Fatalf("text = %q", msg.Text())
	}
	if msg.Stop != StopStop {
		t.Fatalf("stop = %v", msg.Stop)
	}
	if msg.Usage.TotalTokens != 8 {
		t.Fatalf("usage = %+v", msg.Usage)
	}
	if msg.Model != "test-model" {
		t.Fatalf("model = %q", msg.Model)
	}
	var seq []string
	for _, ev := range events {
		seq = append(seq, ev.Type)
	}
	want := []string{EventStart, EventTextStart, EventTextDelta, EventTextDelta, EventTextDelta, EventTextEnd, EventDone}
	if len(seq) != len(want) {
		t.Fatalf("event seq = %v", seq)
	}
	for i := range want {
		if seq[i] != want[i] {
			t.Fatalf("event seq = %v", seq)
		}
	}
	if capReq["stream"] != true {
		t.Errorf("stream flag not set: %v", capReq)
	}
	so, _ := capReq["stream_options"].(map[string]any)
	if so == nil || so["include_usage"] != true {
		t.Errorf("stream_options = %v", capReq["stream_options"])
	}
}

func TestToolCallsStreaming(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}`, ""),
		chunk(`"delta":{},"finish_reason":"tool_calls"`, ""),
		chunk(`"delta":{}`, `,"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}`),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, _, err := run(t, c, []Message{
		SystemMessage("you are a shell"),
		UserMessage("list files"),
	}, Params{Tools: []Tool{{Name: "bash", Description: "run a command", Parameters: json.RawMessage(`{"type":"object","properties":{"cmd":{"type":"string"}}}`)}}})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	calls := msg.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "bash" {
		t.Fatalf("call = %+v", calls[0])
	}
	args, err := calls[0].ParseArgs()
	if err != nil {
		t.Fatalf("args: %v", err)
	}
	if args["cmd"] != "ls" {
		t.Fatalf("args = %v", args)
	}
	if msg.Stop != StopToolUse {
		t.Fatalf("stop = %v", msg.Stop)
	}
}

// TestDuplicateToolCallIDs reproduces the provider behavior that corrupted
// DSH sessions: the server re-issues the same function call (same id) with
// modified arguments. The adapter must keep ids unique with #2, #3 suffixes
// and keep each call's own arguments.
func TestDuplicateToolCallIDs(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":0,"id":"call_dup","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":1,"id":"call_dup","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls -la\"}"}}]}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":2,"id":"call_dup","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls -R\"}"}}]}`, ""),
		chunk(`"delta":{},"finish_reason":"tool_calls"`, ""),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, _, err := run(t, c, []Message{UserMessage("list files")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	calls := msg.ToolCalls()
	if len(calls) != 3 {
		t.Fatalf("tool calls = %d: %+v", len(calls), calls)
	}
	want := []string{"call_dup", "call_dup#2", "call_dup#3"}
	for i, call := range calls {
		if call.ID != want[i] {
			t.Fatalf("call[%d].ID = %q, want %q", i, call.ID, want[i])
		}
	}
	if calls[0].ArgsJSON != `{"cmd":"ls"}` {
		t.Fatalf("call[0].ArgsJSON = %q", calls[0].ArgsJSON)
	}
	if calls[2].ArgsJSON != `{"cmd":"ls -R"}` {
		t.Fatalf("call[2].ArgsJSON = %q", calls[2].ArgsJSON)
	}
}

func TestThinkingBlocks(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant","reasoning_content":"Hmm, "}`, ""),
		chunk(`"delta":{"reasoning_content":"let me think."}`, ""),
		chunk(`"delta":{"content":"Answer: 42"}`, ""),
		chunk(`"delta":{},"finish_reason":"stop"`, ""),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, events, err := run(t, c, []Message{UserMessage("what is the answer?")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var thinking string
	for _, b := range msg.Blocks {
		if b.Kind == "thinking" {
			thinking += b.Text
		}
	}
	if thinking != "Hmm, let me think." {
		t.Fatalf("thinking = %q", thinking)
	}
	if msg.Text() != "Answer: 42" {
		t.Fatalf("text = %q", msg.Text())
	}
	// thinking must close before text opens
	for i, ev := range events {
		if ev.Type == EventTextStart {
			found := false
			for j := 0; j < i; j++ {
				if events[j].Type == EventThinkingEnd {
					found = true
				}
			}
			if !found {
				t.Fatalf("thinking_end missing before text_start: %+v", events)
			}
			return
		}
	}
	t.Fatalf("no text_start in events")
}

func TestInterleavedTextAfterToolCall(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant","content":"First part."}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]}`, ""),
		chunk(`"delta":{"content":" And more after."}`, ""),
		chunk(`"delta":{},"finish_reason":"tool_calls"`, ""),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, events, err := run(t, c, []Message{UserMessage("read a")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if msg.Text() != "First part. And more after." {
		t.Fatalf("text = %q", msg.Text())
	}
	textBlocks, tcBlocks := 0, 0
	for _, b := range msg.Blocks {
		switch b.Kind {
		case "text":
			textBlocks++
		case "toolcall":
			tcBlocks++
		}
	}
	if textBlocks != 2 || tcBlocks != 1 {
		t.Fatalf("blocks = %d text, %d toolcall: %+v", textBlocks, tcBlocks, msg.Blocks)
	}
	starts, ends := 0, 0
	for _, ev := range events {
		switch ev.Type {
		case EventTextStart:
			starts++
		case EventTextEnd:
			ends++
		}
	}
	if starts != ends {
		t.Fatalf("text start/end mismatch: %d/%d", starts, ends)
	}
}

func TestServerErrorMessage(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		`data: {"id":"chatcmpl-1","error":{"message":"context length exceeded","type":"invalid_request_error"}}`,
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)

	msg, _, err := run(t, c, []Message{UserMessage("hi")}, Params{})
	if err == nil || !strings.Contains(err.Error(), "context length exceeded") {
		t.Fatalf("err = %v", err)
	}
	if msg.Stop != StopError {
		t.Fatalf("stop = %v", msg.Stop)
	}
}

func TestHTTPErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":"nope"}`)
	}))
	defer srv.Close()
	c := newClient(t, srv.URL)
	_, err := c.Complete(context.Background(), []Message{UserMessage("hi")}, Params{})
	if err == nil {
		t.Fatal("expected error")
	}
	var he *HTTPError
	if !errors.As(err, &he) {
		t.Fatalf("err type = %T (%v)", err, err)
	}
	if he.Status != 401 {
		t.Fatalf("status = %d", he.Status)
	}
}

func TestMissingFinishReasonInferred(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant","content":"hi"}`, ""),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)
	msg, _, err := run(t, c, []Message{UserMessage("hi")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if msg.Stop != StopStop {
		t.Fatalf("stop = %v", msg.Stop)
	}
}

func TestMissingFinishReasonToolCallsInferred(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		chunk(`"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]}`, ""),
		"data: [DONE]",
	}
	srv := sseServer(t, lines, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)
	msg, _, err := run(t, c, []Message{UserMessage("hi")}, Params{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if msg.Stop != StopToolUse {
		t.Fatalf("stop = %v", msg.Stop)
	}
}

func TestRequestWireFormat(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		chunk(`"delta":{},"finish_reason":"stop"`, ""),
		"data: [DONE]",
	}
	var capReq map[string]any
	srv := sseServer(t, lines, &capReq)
	defer srv.Close()
	c := newClient(t, srv.URL)

	rawArgs := `{" cmd" : "ls "}` // deliberately ugly; must be replayed verbatim
	_, _, err := run(t, c, []Message{
		SystemMessage("sys"),
		UserMessageParts(TextPart("look"), ImagePart("http://x/img.png")),
		AssistantWithToolCalls("calling", ToolCall{ID: "c9", Name: "bash", ArgsJSON: rawArgs, Arguments: json.RawMessage(rawArgs)}),
		ToolMessage("c9", "bash", "output"),
		Message{Role: RoleAssistant, Content: "done", Thinking: "chain-of-thought"},
		UserMessage("next"),
	}, Params{
		Temperature:         Float64Ptr(0.7),
		MaxCompletionTokens: IntPtr(100),
		Tools:               []Tool{{Name: "bash"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	msgs, _ := capReq["messages"].([]any)
	if len(msgs) != 6 {
		t.Fatalf("messages = %d", len(msgs))
	}
	m1, _ := msgs[1].(map[string]any)
	parts, _ := m1["content"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["type"] != "text" || parts[1].(map[string]any)["type"] != "image_url" {
		t.Fatalf("parts = %v", m1["content"])
	}
	m2, _ := msgs[2].(map[string]any)
	tcs, _ := m2["tool_calls"].([]any)
	fn, _ := tcs[0].(map[string]any)["function"].(map[string]any)
	if fn["arguments"] != rawArgs {
		t.Fatalf("args replayed = %v", fn["arguments"])
	}
	m3, _ := msgs[3].(map[string]any)
	if m3["tool_call_id"] != "c9" || m3["name"] != "bash" {
		t.Fatalf("tool msg = %v", m3)
	}
	m4, _ := msgs[4].(map[string]any)
	if m4["reasoning_content"] != "chain-of-thought" {
		t.Fatalf("reasoning replay = %v", m4)
	}
	if capReq["temperature"] != 0.7 {
		t.Fatalf("temperature = %v", capReq["temperature"])
	}
	if capReq["max_completion_tokens"] != float64(100) {
		t.Fatalf("max_completion_tokens = %v", capReq["max_completion_tokens"])
	}
}

func TestMaxTokensCompatField(t *testing.T) {
	lines := []string{
		chunk(`"delta":{"role":"assistant"}`, ""),
		chunk(`"delta":{},"finish_reason":"stop"`, ""),
		"data: [DONE]",
	}
	var capReq map[string]any
	srv := sseServer(t, lines, &capReq)
	defer srv.Close()
	c := newClient(t, srv.URL)
	_, _, err := run(t, c, []Message{UserMessage("hi")}, Params{
		MaxCompletionTokens: IntPtr(55),
		Compat:              Compat{MaxTokensField: "max_tokens"},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if _, ok := capReq["max_completion_tokens"]; ok {
		t.Fatalf("max_completion_tokens should be absent: %v", capReq)
	}
	if capReq["max_tokens"] != float64(55) {
		t.Fatalf("max_tokens = %v", capReq["max_tokens"])
	}
}

func TestModelInfo(t *testing.T) {
	srv := sseServer(t, nil, nil)
	defer srv.Close()
	c := newClient(t, srv.URL)
	info, err := c.ModelInfo(context.Background())
	if err != nil {
		t.Fatalf("ModelInfo: %v", err)
	}
	if info.ID != "test-model" {
		t.Fatalf("info = %+v", info)
	}
}

func TestConfigValidation(t *testing.T) {
	for _, tc := range []Config{
		{BaseURL: "", Model: "m"},
		{BaseURL: "http://x", Model: ""},
		{BaseURL: "ftp://x", Model: "m"},
	} {
		if _, err := New(tc); err == nil {
			t.Fatalf("New(%+v) should fail", tc)
		}
	}
	if _, err := New(Config{BaseURL: "http://x/", Model: "m"}); err != nil {
		t.Fatalf("New trailing slash: %v", err)
	}
}
