package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// Kiro CLI stop hook — stores facts from assistant responses.
// Pair with dmem-store-turn-all-prompt.exe for full bidirectional capture.

const maxFacts = 3

type hookInput struct {
	HookEventName     string `json:"hook_event_name"`
	AssistantResponse string `json:"assistant_response"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "stop" || len(input.AssistantResponse) < 80 {
		os.Exit(0)
	}

	facts := extractFacts(input.AssistantResponse)
	if len(facts) == 0 {
		os.Exit(0)
	}

	conn, err := grpc.NewClient("localhost:19090", grpc.WithTransportCredentials(insecure.NewCredentials()))
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

	for _, f := range facts {
		client.Store(ctx, &pb.StoreRequest{Owner: owner, Key: f.key, Content: f.content})
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
