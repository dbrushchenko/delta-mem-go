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

// Kiro CLI userPromptSubmit hook (deep) — uses Think RPC for full-pipeline enrichment.
// Think uses: δ-mem → turbogo → IBNN crystallization → Gemma/substrate → NLI validation.
// Produces a synthesized insight from stored knowledge, not just retrieved IDs.
// Heavier (~1-5s) but produces novel connections. Use for important queries.

const maxSeeds = 3

type hookInput struct {
	HookEventName string `json:"hook_event_name"`
	Prompt        string `json:"prompt"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "userPromptSubmit" || len(input.Prompt) < 20 {
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

	// Confidence gate — only invoke Think if δ-mem has relevant state
	recallResp, err := client.Recall(ctx, &pb.RecallRequest{Owner: owner, Query: input.Prompt})
	if err != nil || recallResp.Confidence < 0.10 {
		os.Exit(0) // not enough stored knowledge to think about
	}

	// Extract seeds from the prompt (split into meaningful phrases)
	seeds := extractSeeds(input.Prompt)
	if len(seeds) == 0 {
		os.Exit(0)
	}

	// Think — full pipeline synthesis
	thought, err := client.Think(ctx, &pb.ThinkRequest{Owner: owner, Seeds: seeds})
	if err != nil || thought.Idea == "" {
		os.Exit(0)
	}

	// Only output if confidence is meaningful
	if thought.Confidence < 0.05 || !thought.Valid {
		os.Exit(0)
	}

	// Dedup
	provided := loadProvided()
	ideaKey := thought.Idea
	if len(ideaKey) > 60 {
		ideaKey = ideaKey[:60]
	}
	if provided[ideaKey] {
		os.Exit(0)
	}
	provided[ideaKey] = true
	saveProvided(provided)

	// Output: synthesized insight + neighbors
	out := fmt.Sprintf("δ-think[conf=%.2f|nov=%.2f] %s", thought.Confidence, thought.Novelty, thought.Idea)
	if len(thought.Neighbors) > 0 {
		out += fmt.Sprintf(" (via: %s)", strings.Join(thought.Neighbors, ", "))
	}
	fmt.Print(out)
}

func extractSeeds(prompt string) []string {
	// Split on sentence boundaries and take up to maxSeeds meaningful chunks
	var seeds []string
	for _, s := range strings.FieldsFunc(prompt, func(r rune) bool {
		return r == '.' || r == '?' || r == '!' || r == ','
	}) {
		s = strings.TrimSpace(s)
		if len(s) > 10 {
			seeds = append(seeds, s)
			if len(seeds) >= maxSeeds {
				break
			}
		}
	}
	if len(seeds) == 0 && len(prompt) > 10 {
		seeds = []string{prompt}
	}
	return seeds
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
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func sessionFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kiro", "session", "dmem-think-provided.json")
}

func loadProvided() map[string]bool {
	m := make(map[string]bool)
	data, err := os.ReadFile(sessionFile())
	if err != nil {
		return m
	}
	var keys []string
	json.Unmarshal(data, &keys)
	for _, k := range keys {
		m[k] = true
	}
	return m
}

func saveProvided(m map[string]bool) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	if len(keys) > 100 {
		keys = keys[len(keys)-100:]
	}
	data, _ := json.Marshal(keys)
	os.MkdirAll(filepath.Dir(sessionFile()), 0755)
	os.WriteFile(sessionFile(), data, 0644)
}
