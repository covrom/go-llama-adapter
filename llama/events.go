package llama

// Event is one unit of the streaming protocol, mirroring pi-ai's
// AssistantMessageEvent union (only the subset llama.cpp can produce).
type Event struct {
	// Type is one of the Event* constants.
	Type string

	// ContentIndex is the index of the affected block in Partial.Blocks.
	// Valid for text_*/thinking_*/toolcall_* events.
	ContentIndex int

	// Delta is the incremental payload (text, thinking, or tool-call
	// argument fragment). Valid for *_delta events.
	Delta string

	// Block is the finished block. Valid for toolcall_end (and end
	// events generally, when meaningful).
	Block *ContentBlock

	// Reason is set on done and error events.
	Reason StopReason

	// Partial is the live, shared message-so-far. The same pointer is
	// passed to every event and mutated in place as deltas arrive, so a
	// consumer holding the first Partial sees the full message at done.
	Partial *AssistantMessage

	// Err is set on error events.
	Err error
}

const (
	EventStart         = "start"
	EventTextStart     = "text_start"
	EventTextDelta     = "text_delta"
	EventTextEnd       = "text_end"
	EventThinkingStart = "thinking_start"
	EventThinkingDelta = "thinking_delta"
	EventThinkingEnd   = "thinking_end"
	EventToolCallStart = "toolcall_start"
	EventToolCallDelta = "toolcall_delta"
	EventToolCallEnd   = "toolcall_end"
	EventDone          = "done"
	EventError         = "error"
)

// Stream is a live assistant response.
//
// Events are delivered over Ch. The channel is closed exactly once, after
// the terminal done or error event. Wait returns the terminal error (nil on
// success) and the final message; it consumes the rest of the stream, so it
// must not be mixed with manual ranging over Ch.
type Stream struct {
	Ch   chan Event
	done chan struct{}
	err  error
	msg  *AssistantMessage
}

// Wait blocks until the stream finishes.
func (s *Stream) Wait() (*AssistantMessage, error) {
	for range s.Ch {
	}
	<-s.done
	return s.msg, s.err
}
