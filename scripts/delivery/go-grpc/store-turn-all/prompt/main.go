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

// Kiro CLI userPromptSubmit hook — stores facts from user prompts.
// Pair with dmem-store-turn-all-stop.exe for full bidirectional capture.

const maxFacts = 3

type hookInput struct {
	HookEventName string `json:"hook_event_name"`
	Prompt        string `json:"prompt"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "userPromptSubmit" || len(input.Prompt) < 30 {
		os.Exit(0)
	}

	facts := extractFacts(input.Prompt)
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
		if len(line) < 30 {
			continue
		}
		key := line
		if len(key) > 60 { key = key[:60] }
		facts = append(facts, fact{key, line})
		if len(facts) >= maxFacts { break }
	}
	return facts
}
