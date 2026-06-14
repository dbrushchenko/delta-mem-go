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

// Kiro CLI userPromptSubmit hook (standard) — full retrieval matching main.go wiring.
// 1. Recall → δ-mem confidence + correction vector
// 2. IBNNForward → embed through IBNN (sharpened)
// 3. IBNNForwardHidden → crystallize the δ-mem correction
// 4. TurboSearch with raw embed → turbovec neighbors
// 5. TurboSearch with crystallized → turbogo-informed neighbors
// Merges both result sets, deduplicates, outputs top matches.

const (
	topK          = 5
	maxOutput     = 2
	confThreshold = 0.05
)

type hookInput struct {
	HookEventName string `json:"hook_event_name"`
	Prompt        string `json:"prompt"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "userPromptSubmit" || len(input.Prompt) < 10 {
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

	// Layer 1: δ-mem Recall — confidence gate + correction vector
	recallResp, err := client.Recall(ctx, &pb.RecallRequest{Owner: owner, Query: input.Prompt})
	if err != nil || recallResp.Confidence < confThreshold {
		os.Exit(0)
	}

	// Layer 2: IBNNForward — embed prompt through IBNN (sharpened representation)
	embResp, err := client.IBNNForward(ctx, &pb.IBNNForwardRequest{Owner: owner, Text: input.Prompt})
	if err != nil || len(embResp.Output) == 0 {
		os.Exit(0)
	}

	// Layer 3: IBNNForwardHidden — crystallize the δ-mem correction vector
	var crystallized []float32
	if len(recallResp.Correction) == int(embResp.Dim) {
		crystResp, err := client.IBNNForwardHidden(ctx, &pb.IBNNForwardHiddenRequest{Owner: owner, HiddenState: recallResp.Correction})
		if err == nil && len(crystResp.Output) > 0 {
			crystallized = crystResp.Output
		}
	}

	// Layer 4: TurboSearch with IBNN-embedded prompt → primary neighbors
	searchResp, err := client.TurboSearch(ctx, &pb.TurboSearchRequest{Owner: owner, Query: embResp.Output, K: int32(topK)})
	if err != nil {
		os.Exit(0)
	}

	// Layer 5: TurboSearch with crystallized δ-mem correction → secondary neighbors
	var secondaryIDs []string
	var secondaryScores []float32
	if crystallized != nil {
		s2, err := client.TurboSearch(ctx, &pb.TurboSearchRequest{Owner: owner, Query: crystallized, K: 3})
		if err == nil {
			secondaryIDs = s2.Ids
			secondaryScores = s2.Scores
		}
	}

	// Merge and deduplicate
	provided := loadProvided()
	seen := make(map[string]bool)
	var output []string

	// Primary results first (higher relevance)
	for i, id := range searchResp.Ids {
		if provided[id] || seen[id] || searchResp.Scores[i] < 0.25 {
			continue
		}
		seen[id] = true
		output = append(output, fmt.Sprintf("δ[%s|%.2f]", id, searchResp.Scores[i]))
		provided[id] = true
		if len(output) >= maxOutput {
			break
		}
	}

	// Secondary results (crystallized — deeper associations)
	if len(output) < maxOutput {
		for i, id := range secondaryIDs {
			if provided[id] || seen[id] || secondaryScores[i] < 0.30 {
				continue
			}
			seen[id] = true
			output = append(output, fmt.Sprintf("δ°[%s|%.2f]", id, secondaryScores[i]))
			provided[id] = true
			if len(output) >= maxOutput {
				break
			}
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
	return filepath.Join(home, ".kiro", "session", "dmem-provided-standard.json")
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
	if len(keys) > 200 { keys = keys[len(keys)-200:] }
	data, _ := json.Marshal(keys)
	os.MkdirAll(filepath.Dir(sessionFile()), 0755)
	os.WriteFile(sessionFile(), data, 0644)
}
