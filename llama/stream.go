package llama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Complete sends a chat completion request and returns a streaming
// response. The returned error is only for setup failures (bad request,
// connection refused, non-2xx); transport and server errors during
// streaming are delivered as terminal error events.
func (c *Client) Complete(ctx context.Context, msgs []Message, p Params) (*Stream, error) {
	if len(msgs) == 0 {
		return nil, errors.New("llama: no messages")
	}
	req, err := c.buildCompleteRequest(msgs, p)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llama: encode request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	res, err := c.do(httpReq)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		defer res.Body.Close()
		return nil, c.httpError(res)
	}
	return c.readStream(ctx, res.Body), nil
}

func (c *Client) buildCompleteRequest(msgs []Message, p Params) (*chatRequest, error) {
	req, err := buildRequest(c.cfg.Model, msgs, p.Tools)
	if err != nil {
		return nil, err
	}
	req.Stream = true
	if !p.Compat.DisableUsageInStream {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	req.Temperature = p.Temperature
	req.TopP = p.TopP
	if p.MaxCompletionTokens != nil {
		if p.Compat.MaxTokensField == "max_tokens" {
			req.MaxTokens = p.MaxCompletionTokens
		} else {
			req.MaxCompletionTokens = p.MaxCompletionTokens
		}
	}
	req.Stop = p.Stop
	return req, nil
}

// wire chunk shapes (chat completion stream)
type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type chunkToolCallDelta struct {
	Index    int `json:"index"`
	ID       string
	Type     string
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// delta is choice.delta of a stream chunk.
type delta struct {
	Role             string               `json:"role"`
	Content          string               `json:"content"`
	ReasoningContent string               `json:"reasoning_content"`
	ToolCalls        []chunkToolCallDelta `json:"tool_calls"`
}

type chunkChoice struct {
	Index        int     `json:"index"`
	Delta        delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
	Usage        *Usage  `json:"usage"`
}

type wireChunk struct {
	ID      string        `json:"id"`
	Model   string        `json:"model"`
	Usage   *Usage        `json:"usage"`
	Error   *apiError     `json:"error"`
	Choices []chunkChoice `json:"choices"`
}

type toolSlot struct {
	blockIdx int
	id       string
	name     string
	args     strings.Builder
	// sawID/sawName let us detect a re-issued call that only carries the
	// id (or only the name) in later deltas.
	sawID   bool
	sawName bool
}

// readStream consumes the SSE body and produces the event stream.
func (c *Client) readStream(ctx context.Context, body io.ReadCloser) *Stream {
	msg := &AssistantMessage{}
	s := &Stream{
		Ch:   make(chan Event, 64),
		done: make(chan struct{}),
		msg:  msg,
	}

	st := &streamState{
		msg:             msg,
		setErr:          func(e error) { s.err = e },
		textBlock:       -1,
		thinkingBlock:   -1,
		usedToolCallIDs: make(map[string]int),
	}
	send := func(ev Event) bool {
		ev.Partial = msg
		select {
		case s.Ch <- ev:
			return true
		case <-ctx.Done():
			return false
		}
	}

	go func() {
		defer body.Close()
		defer close(s.Ch)
		defer close(s.done)

		if !send(Event{Type: EventStart}) {
			return
		}

		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		fin := false
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") { // SSE comment/keep-alive
				continue
			}
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				break
			}
			var ch wireChunk
			if err := json.Unmarshal([]byte(payload), &ch); err != nil {
				st.fail(ctx, fmt.Errorf("llama: bad stream chunk: %w", err), send)
				return
			}
			if ch.Error != nil {
				st.fail(ctx, errors.New("llama: server error: "+ch.Error.Message), send)
				return
			}
			if ch.Model != "" {
				msg.Model = ch.Model
			}
			if ch.Usage != nil {
				msg.Usage = *ch.Usage
			}
			for _, choice := range ch.Choices {
				if choice.Usage != nil {
					msg.Usage = *choice.Usage // some servers put usage on the choice
				}
				st.applyDelta(choice.Delta, send)
				if choice.FinishReason != nil {
					st.finish(*choice.FinishReason, send)
					fin = true
				}
			}
		}
		if err := scanner.Err(); err != nil {
			if ctx.Err() != nil {
				st.fail(ctx, ctx.Err(), send)
				return
			}
			st.fail(ctx, fmt.Errorf("llama: stream read: %w", err), send)
			return
		}
		if ctx.Err() != nil {
			st.fail(ctx, ctx.Err(), send)
			return
		}
		if !fin {
			// Stream ended without finish_reason (older servers): infer.
			st.finish(st.inferFinishReason(), send)
			fin = true
		}
	}()
	return s
}

type streamState struct {
	msg    *AssistantMessage
	setErr func(error)
	// open block indices in msg.Blocks
	textBlock      int // -1 when none open
	thinkingBlock  int
	textClosed     bool
	thinkingClosed bool
	toolSlots      map[int]*toolSlot // wire tool_call index -> slot
	toolOrder      []int             // wire tool_call indices in first-seen order
	// usedToolCallIDs enforces id uniqueness inside the message.
	usedToolCallIDs map[string]int
}

func (st *streamState) emitBlockStart(kind string, evType string, send func(Event) bool) int {
	st.msg.Blocks = append(st.msg.Blocks, ContentBlock{Kind: kind})
	idx := len(st.msg.Blocks) - 1
	send(Event{Type: evType, ContentIndex: idx})
	return idx
}

// closeText ends the currently open text block (if any) once, emitting
// text_end. Called before a tool call opens and at finish.
func (st *streamState) closeText(send func(Event) bool) {
	if st.textBlock >= 0 && !st.textClosed {
		st.textClosed = true
		send(Event{Type: EventTextEnd, ContentIndex: st.textBlock})
	}
}

func (st *streamState) closeThinking(send func(Event) bool) {
	if st.thinkingBlock >= 0 && !st.thinkingClosed {
		st.thinkingClosed = true
		send(Event{Type: EventThinkingEnd, ContentIndex: st.thinkingBlock})
	}
}

func (st *streamState) ensureText(send func(Event) bool) int {
	// A text block that was closed (by an intervening tool call) must not
	// be reused; open a fresh one instead.
	if st.textBlock >= 0 && !st.textClosed && st.msg.Blocks[st.textBlock].Kind == "text" {
		return st.textBlock
	}
	// Opening text supersedes any open thinking block.
	st.closeThinking(send)
	st.closeText(send)
	st.textClosed = false
	st.textBlock = st.emitBlockStart("text", EventTextStart, send)
	return st.textBlock
}

func (st *streamState) ensureThinking(send func(Event) bool) int {
	if st.thinkingBlock >= 0 && !st.thinkingClosed && st.msg.Blocks[st.thinkingBlock].Kind == "thinking" {
		return st.thinkingBlock
	}
	// Opening thinking supersedes any open text block.
	st.closeText(send)
	st.closeThinking(send)
	st.thinkingClosed = false
	st.thinkingBlock = st.emitBlockStart("thinking", EventThinkingStart, send)
	return st.thinkingBlock
}

// uniqueToolCallID suffixes repeated ids in order of appearance so every
// tool call in the message keeps a unique id (llama.cpp can re-issue the
// last function call with modified arguments but the same id).
func (st *streamState) uniqueToolCallID(raw string) string {
	n := st.usedToolCallIDs[raw] + 1
	st.usedToolCallIDs[raw] = n
	if n == 1 {
		return raw
	}
	return fmt.Sprintf("%s#%d", raw, n)
}

func (st *streamState) applyDelta(d delta, send func(Event) bool) {
	if d.ReasoningContent != "" {
		idx := st.ensureThinking(send)
		st.msg.Blocks[idx].Text += d.ReasoningContent
		send(Event{Type: EventThinkingDelta, ContentIndex: idx, Delta: d.ReasoningContent})
	}
	if d.Content != "" {
		idx := st.ensureText(send)
		st.msg.Blocks[idx].Text += d.Content
		send(Event{Type: EventTextDelta, ContentIndex: idx, Delta: d.Content})
	}
	for _, tc := range d.ToolCalls {
		slot, ok := st.toolSlots[tc.Index]
		if !ok {
			st.closeText(send)
			st.closeThinking(send)
			slot = &toolSlot{
				blockIdx: st.emitBlockStart("toolcall", EventToolCallStart, send),
			}
			if st.toolSlots == nil {
				st.toolSlots = make(map[int]*toolSlot)
			}
			st.toolSlots[tc.Index] = slot
			st.toolOrder = append(st.toolOrder, tc.Index)
		}
		if tc.ID != "" && !slot.sawID {
			slot.id = st.uniqueToolCallID(tc.ID)
			slot.sawID = true
		}
		if tc.Function.Name != "" && !slot.sawName {
			slot.name = tc.Function.Name
			slot.sawName = true
		}
		if tc.Function.Arguments != "" {
			slot.args.WriteString(tc.Function.Arguments)
			send(Event{Type: EventToolCallDelta, ContentIndex: slot.blockIdx, Delta: tc.Function.Arguments})
		}
	}
}

func (st *streamState) finalizeToolCalls(send func(Event) bool) {
	for _, wireIdx := range st.toolOrder {
		slot := st.toolSlots[wireIdx]
		args := slot.args.String()
		if args == "" {
			args = "{}"
		}
		arguments := json.RawMessage(args)
		if !json.Valid(arguments) {
			// Keep the raw stream even if a server produced malformed
			// JSON; consumers can still see what was emitted.
		}
		tc := &ToolCall{
			ID:        slot.id,
			Name:      slot.name,
			ArgsJSON:  args,
			Arguments: arguments,
		}
		block := &st.msg.Blocks[slot.blockIdx]
		block.ToolCall = tc
		block.Text = "" // args live in ToolCall, not Text
		ev := Event{Type: EventToolCallEnd, ContentIndex: slot.blockIdx, Block: tcAsBlock(tc)}
		send(ev)
	}
}

func tcAsBlock(tc *ToolCall) *ContentBlock {
	return &ContentBlock{Kind: "toolcall", ToolCall: tc}
}

// inferFinishReason is used when the stream ends without a finish_reason
// (older servers). It must not inspect msg blocks: tool call blocks exist
// at this point but are not finalized yet (ToolCall pointer is nil).
func (st *streamState) inferFinishReason() string {
	if len(st.toolOrder) > 0 {
		return "tool_calls"
	}
	return "stop"
}

func mapFinishReason(wire string) StopReason {
	switch wire {
	case "stop", "":
		return StopStop
	case "length":
		return StopLength
	case "tool_calls", "function_call":
		return StopToolUse
	case "content_filter":
		return StopContentFilter
	default:
		return StopStop
	}
}

func (st *streamState) finish(wireReason string, send func(Event) bool) {
	st.closeText(send)
	st.closeThinking(send)
	st.finalizeToolCalls(send)
	st.msg.Stop = mapFinishReason(wireReason)
	st.msg.FinishReason = wireReason
	send(Event{Type: EventDone, Reason: st.msg.Stop})
}

func (st *streamState) fail(ctx context.Context, err error, send func(Event) bool) {
	var reason StopReason
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		reason = StopAborted
	} else {
		reason = StopError
	}
	st.msg.Stop = reason
	if st.setErr != nil {
		st.setErr(err)
	}
	send(Event{Type: EventError, Reason: reason, Err: err})
}
