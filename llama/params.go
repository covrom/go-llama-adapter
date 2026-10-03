package llama

// Params are per-request generation parameters. Nil/zero fields are omitted
// from the request, letting the server apply its own defaults.
type Params struct {
	// Temperature, 0..2 in llama.cpp terms.
	Temperature *float64
	// TopP is nucleus sampling.
	TopP *float64
	// MaxCompletionTokens caps output length. llama.cpp uses the standard
	// max_completion_tokens field; some very old builds reject it, in which
	// case set Compat.MaxTokensField to "max_tokens".
	MaxCompletionTokens *int
	// Stop sequences.
	Stop []string
	// Tools declared for this request.
	Tools []Tool
	// Compat adjusts wire behavior for specific llama.cpp builds.
	Compat Compat
}

// Compat overrides wire-level behavior for specific llama.cpp builds.
// Everything defaults to the current server behavior.
type Compat struct {
	// DisableUsageInStream drops stream_options.include_usage for builds
	// that reject the field; Usage then stays zero.
	DisableUsageInStream bool
	// MaxTokensField selects the wire field name for MaxCompletionTokens.
	// "max_completion_tokens" (default) or "max_tokens" for legacy builds.
	MaxTokensField string
}

func Float64Ptr(v float64) *float64 { return &v }
func IntPtr(v int) *int             { return &v }
