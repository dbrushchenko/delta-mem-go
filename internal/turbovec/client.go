package turbovec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Client talks to the turbovec Python FastAPI sidecar (real implementation).
type Client struct {
	baseURL string
	client  *http.Client
}

func NewClient(baseURL string) *Client {
	if baseURL == "" {
		baseURL = "http://localhost:8001"
	}
	return &Client{baseURL: baseURL, client: &http.Client{}}
}

type addReq struct {
	Owner  string    `json:"owner"`
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

func (c *Client) Add(ctx context.Context, owner, id string, vector []float32) error {
	req := addReq{Owner: owner, ID: id, Vector: vector}
	data, _ := json.Marshal(req)
	_, err := c.post(ctx, "/add", data)
	return err
}

type searchReq struct {
	Owner string    `json:"owner"`
	Query []float32 `json:"query"`
	K     int       `json:"k"`
}

func (c *Client) Search(ctx context.Context, owner string, query []float32, k int) ([]string, []float32, error) {
	req := searchReq{Owner: owner, Query: query, K: k}
	data, _ := json.Marshal(req)
	resp, err := c.post(ctx, "/search", data)
	if err != nil {
		return nil, nil, err
	}
	var result struct {
		IDs    []string  `json:"ids"`
		Scores []float32 `json:"scores"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, nil, err
	}
	return result.IDs, result.Scores, nil
}

func (c *Client) post(ctx context.Context, path string, body []byte) ([]byte, error) {
	req, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("turbovec sidecar returned %d", resp.StatusCode)
	}
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return respBody, nil
}
