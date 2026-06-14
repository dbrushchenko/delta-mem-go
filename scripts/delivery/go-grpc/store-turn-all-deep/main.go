package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// Kiro CLI hook (deep, bidirectional) — stores from BOTH prompts and responses.
// Uses StoreDeep RPC = single call, all 12 layers.

const maxFacts = 3

type hookInput struct {
	HookEventName     string `json:"hook_event_name"`
	Prompt            string `json:"prompt"`
	AssistantResponse string `json:"assistant_response"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}

	var text string
	switch input.HookEventName {
	case "userPromptSubmit":
		text = input.Prompt
	case "stop":
		text = input.AssistantResponse
	default:
		os.Exit(0)
	}
	if len(text) < 40 {
		os.Exit(0)
	}

	facts := extractFacts(text)
	if len(facts) == 0 {
		os.Exit(0)
	}

	conn, err := dial("localhost:19090")
	if err != nil {
		os.Exit(0)
	}
	defer conn.Close()
	client := pb.NewDeltaMemClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	owner := os.Getenv("USERNAME")
	if owner == "" {
		owner = "default"
	}

	for _, f := range facts {
		client.StoreDeep(ctx, &pb.StoreRequest{Owner: owner, Key: f.key, Content: f.content})
	}
}

type fact struct{ key, content string }

func extractFacts(text string) []fact {
	var facts []fact
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if len(line) < 40 || strings.HasPrefix(line, "```") || strings.HasPrefix(line, "---") || strings.HasPrefix(line, "|") {
			continue
		}
		lower := strings.ToLower(line)
		if containsAny(lower,
			"created", "fixed", "deployed", "configured", "installed", "built",
			"updated", "deleted", "moved", "copied", "cloned", "wired",
			"decided", "chose", "switched", "set", "changed", "renamed",
			"running", "listening", "connected", "verified", "confirmed",
			"because", "means", "requires", "uses", "stores", "loads",
			"the issue", "the problem", "the fix", "works by",
			"want", "need", "should", "let's", "can you", "please",
		) {
			key := line
			if len(key) > 60 { key = key[:60] }
			facts = append(facts, fact{key, line})
			if len(facts) >= maxFacts { break }
		}
	}
	return facts
}

func containsAny(s string, words ...string) bool {
	for _, w := range words { if strings.Contains(s, w) { return true } }
	return false
}

func dial(addr string) (*grpc.ClientConn, error) {
	token := loadToken()
	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	if token != "" {
		opts = append(opts, grpc.WithUnaryInterceptor(func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, callOpts ...grpc.CallOption) error {
			ctx = metadata.AppendToOutgoingContext(ctx, "x-api-key", token)
			return invoker(ctx, method, req, reply, cc, callOpts...)
		}))
	}
	return grpc.NewClient(addr, opts...)
}

func loadToken() string {
	home, _ := os.UserHomeDir()
	data, err := os.ReadFile(filepath.Join(home, ".memgo", "token"))
	if err != nil { return "" }
	return strings.TrimSpace(string(data))
}
