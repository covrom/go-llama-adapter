// Package llama is a minimal LLM adapter for the llama.cpp server
// (OpenAI-compatible /v1/chat/completions endpoint).
//
// It is scoped to llama.cpp only: no provider registry, no OAuth, no
// multi-API dispatch. The public surface mirrors the normalized message
// and event protocol of pi-ai's openai-completions adapter, so an
// application written against it can be ported to other adapters later.
package llama

import "encoding/json"

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// StopReason is the terminal state of an assistant message.
type StopReason string

const (
	StopStop          StopReason = "stop"
	StopLength        StopReason = "length"
	StopToolUse       StopReason = "toolUse"
	StopContentFilter StopReason = "contentFilter"
	StopError         StopReason = "error"
	StopAborted       StopReason = "aborted"
)

// Usage reports token counts. llama.cpp fills this in the final streaming
// chunk when stream_options.include_usage is set.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ToolCall is one function invocation requested by the model.
//
// ID is unique within a single assistant message: if the server reuses the
// same id for more than one tool call (observed on llama.cpp when it
// re-issues the last call with modified arguments), the adapter suffixes
// the later occurrences in order of appearance: "call_x", "call_x#2",
// "call_x#3". The first occurrence keeps the server id byte-identical.
type ToolCall struct {
	ID        string
	Name      string
	ArgsJSON  string // raw arguments exactly as streamed (never modified)
	Arguments json.RawMessage
}

// ParseArgs decodes Arguments into a map.
func (tc ToolCall) ParseArgs() (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(tc.Arguments, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// Tool declares a callable function for the request.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage // JSON Schema; may be nil
}

// ContentBlock is one block of an assistant message, in wire order.
type ContentBlock struct {
	Kind     string // "text" | "thinking" | "toolcall"
	Text     string // for "text" and "thinking" blocks
	ToolCall *ToolCall
}

// AssistantMessage is the accumulated result of one completion.
type AssistantMessage struct {
	Blocks       []ContentBlock
	Usage        Usage
	Stop         StopReason
	Model        string
	FinishReason string // raw wire value, e.g. "tool_calls"
}

// Text returns the concatenation of all text blocks.
func (m *AssistantMessage) Text() string {
	var out string
	for _, b := range m.Blocks {
		if b.Kind == "text" {
			out += b.Text
		}
	}
	return out
}

// ToolCalls returns all tool call blocks in order.
func (m *AssistantMessage) ToolCalls() []ToolCall {
	var out []ToolCall
	for _, b := range m.Blocks {
		if b.Kind == "toolcall" {
			out = append(out, *b.ToolCall)
		}
	}
	return out
}

// ContentPart is a piece of user or system message content.
type ContentPart struct {
	Type     string `json:"type"` // "text" | "image_url"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

func TextPart(s string) ContentPart {
	return ContentPart{Type: "text", Text: s}
}

// ImagePart references an image by URL (llama.cpp multimodal models).
func ImagePart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: url}
}

// Message is one input message, normalized.
//
// Exactly one of Content / Parts (user, system) or ToolCalls (assistant)
// or ToolCallID (tool result) should be set.
type Message struct {
	Role       Role
	Content    string
	Parts      []ContentPart
	ToolCalls  []ToolCall
	ToolCallID string
	Name       string
	// Thinking replays assistant reasoning content; sent back to the
	// server as reasoning_content so llama.cpp can continue chains of
	// thought for reasoning models.
	Thinking string
}

func SystemMessage(s string) Message {
	return Message{Role: RoleSystem, Content: s}
}

func UserMessage(s string) Message {
	return Message{Role: RoleUser, Content: s}
}

func UserMessageParts(parts ...ContentPart) Message {
	return Message{Role: RoleUser, Parts: parts}
}

// AssistantWithToolCalls builds an assistant message that only invoked tools.
func AssistantWithToolCalls(text string, calls ...ToolCall) Message {
	return Message{Role: RoleAssistant, Content: text, ToolCalls: calls}
}

// ToolMessage reports the result of one tool call.
func ToolMessage(callID, name, result string) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Name: name, Content: result}
}

// ModelInfo describes one model served by llama.cpp.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}
