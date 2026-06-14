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

// Kiro CLI userPromptSubmit hook (deep, individual layers).
// Client-side orchestration — calls each layer separately for visibility.
// Output is tagged by source layer for quality evaluation.
//
// Pipeline:
// 1. AmIConfident     → gate
// 2. Recall           → δ-mem confidence + correction
// 3. IBNNForward      → embed prompt
// 4. IBNNForwardHidden→ crystallize correction
// 5. TurbogoSearch    → deep neighbors (crystallized)
// 6. TurboSearch      → service neighbors (raw embed)
// 7. Validate         → truth-check candidates
// 8. QueryTemporal    → recent events
// 9. HarvestWander    → spontaneous thoughts

type hookInput struct {
	HookEventName string `json:"hook_event_name"`
	Prompt        string `json:"prompt"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "userPromptSubmit" || len(input.Prompt) < 15 {
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

	// 1. AmIConfident — gate
	confResp, err := client.AmIConfident(ctx, &pb.ConfidenceRequest{Owner: owner, Text: input.Prompt})
	if err != nil || confResp.Level == 0 {
		os.Exit(0)
	}

	// 2. Recall — δ-mem confidence + correction vector
	recallResp, err := client.Recall(ctx, &pb.RecallRequest{Owner: owner, Query: input.Prompt})
	if err != nil || recallResp.Confidence < 0.05 {
		os.Exit(0)
	}

	// 3. IBNNForward — embed prompt (sharpened)
	embResp, err := client.IBNNForward(ctx, &pb.IBNNForwardRequest{Owner: owner, Text: input.Prompt})
	if err != nil || len(embResp.Output) == 0 {
		os.Exit(0)
	}

	provided := loadProvided()
	var output []string

	// 4. IBNNForwardHidden — crystallize δ-mem correction
	var crystallized []float32
	if len(recallResp.Correction) == int(embResp.Dim) {
		crystResp, err := client.IBNNForwardHidden(ctx, &pb.IBNNForwardHiddenRequest{Owner: owner, HiddenState: recallResp.Correction})
		if err == nil && len(crystResp.Output) > 0 {
			crystallized = crystResp.Output
		}
	}

	// 5. TurbogoSearch — deep neighbors from crystallized vector
	if crystallized != nil {
		resp, err := client.TurbogoSearch(ctx, &pb.TurboSearchRequest{Owner: owner, Query: crystallized, K: 3})
		if err == nil {
			for i, id := range resp.Ids {
				if provided[id] || resp.Scores[i] < 0.25 { continue }
				// 7. Validate each turbogo result
				valid := true
				if vr, err := client.Validate(ctx, &pb.ValidateRequest{Owner: owner, Statement: id}); err == nil {
					valid = vr.Valid
				}
				marker := "δ⁺"
				if !valid { marker = "δ⁻" }
				output = append(output, fmt.Sprintf("%s[%s|%.2f]", marker, id, resp.Scores[i]))
				provided[id] = true
				if len(output) >= 2 { break }
			}
		}
	}

	// 6. TurboSearch — service-layer neighbors from raw embed
	if len(output) < 3 {
		resp, err := client.TurboSearch(ctx, &pb.TurboSearchRequest{Owner: owner, Query: embResp.Output, K: 3})
		if err == nil {
			for i, id := range resp.Ids {
				if provided[id] || resp.Scores[i] < 0.25 { continue }
				output = append(output, fmt.Sprintf("δ[%s|%.2f]", id, resp.Scores[i]))
				provided[id] = true
				if len(output) >= 3 { break }
			}
		}
	}

	// 8. QueryTemporal — recent events
	if len(output) < 4 {
		resp, err := client.QueryTemporal(ctx, &pb.TemporalRequest{Owner: owner, Limit: 3})
		if err == nil {
			for _, e := range resp.Events {
				if provided[e.Id] || e.Content == "" { continue }
				content := e.Content
				if len(content) > 60 { content = content[:60] }
				output = append(output, fmt.Sprintf("δ↺[%s|%s]", content, e.When[:10]))
				provided[e.Id] = true
				if len(output) >= 4 { break }
			}
		}
	}

	// 9. HarvestWander — spontaneous thoughts
	if len(output) < 5 {
		resp, err := client.HarvestWander(ctx, &pb.OwnerRequest{Owner: owner})
		if err == nil && resp != nil {
			for _, t := range resp.Thoughts {
				if t.Idea == "" || !t.Valid { continue }
				key := t.Idea
				if len(key) > 60 { key = key[:60] }
				if provided[key] { continue }
				output = append(output, fmt.Sprintf("δ~[%s|conf=%.2f]", key, t.Confidence))
				provided[key] = true
				if len(output) >= 5 { break }
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
	return filepath.Join(home, ".kiro", "session", "dmem-provided-deep.json")
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
