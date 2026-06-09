package gemma

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
)

// Client talks to Ollama (official Gemma 4 QAT support).
type Client struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewClient(model string) *Client {
	if model == "" {
		model = "gemma4:4b-q4"
	}
	return &Client{
		baseURL: "http://localhost:11434",
		model:   model,
		client:  &http.Client{},
	}
}

type ollamaReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

func (c *Client) Generate(ctx context.Context, prompt string) (string, error) {
	reqBody := ollamaReq{Model: c.model, Prompt: prompt, Stream: false}
	data, _ := json.Marshal(reqBody)

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(data))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Response, nil
}
