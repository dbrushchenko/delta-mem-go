package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Handler implements MCP streamable-http protocol (JSON-RPC 2.0 over POST /mcp).
type Handler struct {
	tools   map[string]Tool
	version string
}

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
	Fn          func(map[string]any) (string, error)
}

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type jsonRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id"`
	Result  any    `json:"result,omitempty"`
	Error   any    `json:"error,omitempty"`
}

func New() *Handler {
	return &Handler{tools: make(map[string]Tool), version: "2025-03-26"}
}

func (h *Handler) AddTool(t Tool) {
	h.tools[t.Name] = t
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req jsonRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, nil, -32700, "parse error")
		return
	}

	switch req.Method {
	case "initialize":
		h.handleInitialize(w, req)
	case "notifications/initialized":
		writeResult(w, req.ID, map[string]any{})
	case "tools/list":
		h.handleToolsList(w, req)
	case "tools/call":
		h.handleToolsCall(w, req)
	default:
		writeError(w, req.ID, -32601, fmt.Sprintf("unknown method: %s", req.Method))
	}
}

func (h *Handler) handleInitialize(w http.ResponseWriter, req jsonRPCRequest) {
	writeResult(w, req.ID, map[string]any{
		"protocolVersion": h.version,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"serverInfo":      map[string]any{"name": "dmem", "version": "1.0.0"},
	})
}

func (h *Handler) handleToolsList(w http.ResponseWriter, req jsonRPCRequest) {
	var tools []map[string]any
	for _, t := range h.tools {
		tools = append(tools, map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"inputSchema": t.InputSchema,
		})
	}
	writeResult(w, req.ID, map[string]any{"tools": tools})
}

func (h *Handler) handleToolsCall(w http.ResponseWriter, req jsonRPCRequest) {
	var params struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeError(w, req.ID, -32602, "invalid params")
		return
	}

	tool, ok := h.tools[params.Name]
	if !ok {
		writeError(w, req.ID, -32602, fmt.Sprintf("unknown tool: %s", params.Name))
		return
	}

	result, err := tool.Fn(params.Arguments)
	if err != nil {
		writeResult(w, req.ID, map[string]any{
			"content": []map[string]any{{"type": "text", "text": fmt.Sprintf("error: %v", err)}},
			"isError": true,
		})
		return
	}

	writeResult(w, req.ID, map[string]any{
		"content": []map[string]any{{"type": "text", "text": result}},
	})
}

func writeResult(w http.ResponseWriter, id any, result any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
}

func writeError(w http.ResponseWriter, id any, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Error: map[string]any{"code": code, "message": msg}})
}

// Helper to build inputSchema
func Schema(props map[string]any, required ...string) map[string]any {
	s := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 { s["required"] = required }
	return s
}

func StringProp(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}

func IntProp(desc string, def int) map[string]any {
	return map[string]any{"type": "integer", "description": desc, "default": def}
}

func BoolProp(desc string, def bool) map[string]any {
	return map[string]any{"type": "boolean", "description": desc, "default": def}
}
