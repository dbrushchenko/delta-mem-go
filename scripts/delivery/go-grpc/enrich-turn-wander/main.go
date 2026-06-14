package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// Kiro CLI userPromptSubmit hook (wander) — harvests spontaneous thoughts.
// Calls HarvestWander to collect any insights generated between turns.
// Lightweight, additive — stack alongside other enrich hooks.

type hookInput struct {
	HookEventName string `json:"hook_event_name"`
	Prompt        string `json:"prompt"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "userPromptSubmit" || len(input.Prompt) < 5 {
		os.Exit(0)
	}

	conn, err := dial("localhost:19090")
	if err != nil {
		os.Exit(0)
	}
	defer conn.Close()
	client := pb.NewDeltaMemClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	owner := os.Getenv("USERNAME")
	if owner == "" {
		owner = "default"
	}

	resp, err := client.HarvestWander(ctx, &pb.OwnerRequest{Owner: owner})
	if err != nil || resp == nil || len(resp.Thoughts) == 0 {
		os.Exit(0)
	}

	// Dedup and output
	provided := loadProvided()
	var output []string
	for _, t := range resp.Thoughts {
		if t.Idea == "" || !t.Valid {
			continue
		}
		key := t.Idea
		if len(key) > 60 { key = key[:60] }
		if provided[key] {
			continue
		}
		provided[key] = true
		output = append(output, fmt.Sprintf("δ~[conf=%.2f|nov=%.2f] %s", t.Confidence, t.Novelty, t.Idea))
		if len(output) >= 2 {
			break
		}
	}

	if len(output) == 0 {
		os.Exit(0)
	}

	saveProvided(provided)
	fmt.Print(strings.Join(output, "\n"))
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

func sessionFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kiro", "session", "dmem-provided-wander.json")
}

func loadProvided() map[string]bool {
	m := make(map[string]bool)
	data, err := os.ReadFile(sessionFile())
	if err != nil { return m }
	var keys []string
	json.Unmarshal(data, &keys)
	for _, k := range keys { m[k] = true }
	return m
}

func saveProvided(m map[string]bool) {
	keys := make([]string, 0, len(m))
	for k := range m { keys = append(keys, k) }
	if len(keys) > 100 { keys = keys[len(keys)-100:] }
	data, _ := json.Marshal(keys)
	os.MkdirAll(filepath.Dir(sessionFile()), 0755)
	os.WriteFile(sessionFile(), data, 0644)
}
