package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// Kiro CLI hook — reads JSON from stdin, stores facts to δ-mem-go via gRPC.
// Install: copy binary to .kiro/hooks/ and configure in hook settings.

const minResponseLen = 100

type hookInput struct {
	HookEventName     string `json:"hook_event_name"`
	AssistantResponse string `json:"assistant_response"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0) // silent fail — hooks must not block
	}
	if input.HookEventName != "stop" || len(input.AssistantResponse) < minResponseLen {
		os.Exit(0)
	}

	// Extract key facts (simple heuristic: lines with decisions/actions)
	facts := extractFacts(input.AssistantResponse)
	if len(facts) == 0 {
		os.Exit(0)
	}

	// Connect to δ-mem-go
	conn, err := grpc.NewClient("localhost:19090", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { os.Exit(0) }
	defer conn.Close()
	client := pb.NewDeltaMemClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	owner := os.Getenv("USERNAME")
	if owner == "" { owner = "default" }

	for _, f := range facts {
		client.Store(ctx, &pb.StoreRequest{Owner: owner, Key: f.key, Content: f.content})
	}
}

type fact struct{ key, content string }

func extractFacts(response string) []fact {
	var facts []fact
	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) < 50 { continue }
		// Heuristic: lines with action words are storable
		lower := strings.ToLower(line)
		if containsAny(lower, "decided", "created", "fixed", "deployed", "configured", "installed", "built", "wired", "committed") {
			key := fmt.Sprintf("action-%x", hashQuick(line))
			facts = append(facts, fact{key, line})
			if len(facts) >= 2 { break }
		}
	}
	return facts
}

func containsAny(s string, words ...string) bool {
	for _, w := range words { if strings.Contains(s, w) { return true } }
	return false
}

func hashQuick(s string) uint32 {
	h := uint32(2166136261)
	for _, c := range s { h ^= uint32(c); h *= 16777619 }
	return h
}
