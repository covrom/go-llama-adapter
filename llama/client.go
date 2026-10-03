package llama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Config configures a Client.
type Config struct {
	// BaseURL is the llama.cpp server root, e.g. "http://127.0.0.1:8080".
	// A trailing "/v1" is tolerated and stripped.
	BaseURL string
	// Model is the served model ID (matches /v1/models).
	Model string
	// APIKey is optional; set for servers behind an auth proxy.
	APIKey string
	// HTTPClient is optional; the default client is used when nil.
	HTTPClient *http.Client
}

// Client talks to one llama.cpp server.
type Client struct {
	cfg Config
}

// New validates the config and returns a Client.
func New(cfg Config) (*Client, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("llama: BaseURL is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("llama: Model is required")
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/v1")
	if !strings.HasPrefix(cfg.BaseURL, "http://") && !strings.HasPrefix(cfg.BaseURL, "https://") {
		return nil, fmt.Errorf("llama: BaseURL must start with http:// or https://, got %q", cfg.BaseURL)
	}
	return &Client{cfg: cfg}, nil
}

// Model returns the configured model ID.
func (c *Client) Model() string { return c.cfg.Model }

// ModelInfo fetches the served model metadata.
func (c *Client) ModelInfo(ctx context.Context) (*ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.BaseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	res, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, c.httpError(res)
	}
	var body struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("llama: decode /v1/models: %w", err)
	}
	for i := range body.Data {
		if body.Data[i].ID == c.cfg.Model {
			return &body.Data[i], nil
		}
	}
	return nil, fmt.Errorf("llama: model %q not served (have %d models)", c.cfg.Model, len(body.Data))
}

func (c *Client) do(req *http.Request) (*http.Response, error) {
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	res, err := c.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func (c *Client) httpError(res *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
	return &HTTPError{
		Status:     res.StatusCode,
		StatusCode: res.StatusCode,
		Body:       strings.TrimSpace(string(body)),
	}
}

// HTTPError is a non-2xx HTTP response from the server.
type HTTPError struct {
	Status     int
	StatusCode int // alias kept for clarity
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("llama: server returned HTTP %d: %s", e.Status, firstLine(e.Body))
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// chatRequest is the wire format of POST /v1/chat/completions.
type chatRequest struct {
	Model               string         `json:"model"`
	Messages            []wireMessage  `json:"messages"`
	Tools               []wireTool     `json:"tools,omitempty"`
	Stream              bool           `json:"stream"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type wireTool struct {
	Type     string   `json:"type"`
	Function wireFunc `json:"function"`
}

type wireFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type wireMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	Name             string          `json:"name,omitempty"`
	ToolCalls        []wireToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
}

type wireToolCall struct {
	ID       string     `json:"id"`
	Type     string     `json:"type"`
	Function wireFuncIO `json:"function"`
}

type wireFuncIO struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// encodeContent renders message content as a string or parts array.
func encodeContent(m Message) (json.RawMessage, error) {
	if len(m.Parts) > 0 {
		parts := make([]map[string]any, 0, len(m.Parts))
		for _, p := range m.Parts {
			switch p.Type {
			case "text":
				parts = append(parts, map[string]any{"type": "text", "text": p.Text})
			case "image_url":
				parts = append(parts, map[string]any{
					"type":      "image_url",
					"image_url": map[string]string{"url": p.ImageURL},
				})
			default:
				return nil, fmt.Errorf("llama: unsupported content part type %q", p.Type)
			}
		}
		b, err := json.Marshal(parts)
		if err != nil {
			return nil, err
		}
		return b, nil
	}
	b, err := json.Marshal(m.Content)
	if err != nil {
		return nil, err
	}
	return b, nil
}

// buildRequest converts normalized messages and tools into the wire format.
//
// llama.cpp quirks handled here:
//   - assistant tool calls are replayed verbatim (ArgsJSON is never
//     re-serialized, so server-generated argument formatting is preserved);
//   - reasoning_content is echoed back on assistant messages when set, so
//     reasoning models can continue a chain of thought;
//   - a trailing assistant message that contains only tool calls is kept,
//     as llama.cpp expects the full turn before tool results.
func buildRequest(model string, msgs []Message, tools []Tool) (*chatRequest, error) {
	req := &chatRequest{Model: model}
	for _, m := range msgs {
		wm := wireMessage{Role: string(m.Role)}
		content, err := encodeContent(m)
		if err != nil {
			return nil, err
		}
		wm.Content = content
		switch m.Role {
		case RoleAssistant:
			for _, tc := range m.ToolCalls {
				// Replay raw args verbatim; never re-serialize, so the
				// server's original argument formatting is preserved.
				args := tc.ArgsJSON
				if args == "" {
					args = string(tc.Arguments)
				}
				if args == "" {
					args = "{}"
				}
				wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
					ID:       tc.ID,
					Type:     "function",
					Function: wireFuncIO{Name: tc.Name, Arguments: args},
				})
			}
			if m.Thinking != "" {
				// llama.cpp accepts reasoning_content on assistant turns.
				wm.ReasoningContent = m.Thinking
			}
		case RoleTool:
			wm.ToolCallID = m.ToolCallID
			wm.Name = m.Name
		}
		req.Messages = append(req.Messages, wm)
	}
	for _, t := range tools {
		req.Tools = append(req.Tools, wireTool{
			Type:     "function",
			Function: wireFunc{Name: t.Name, Description: t.Description, Parameters: t.Parameters},
		})
	}
	return req, nil
}
