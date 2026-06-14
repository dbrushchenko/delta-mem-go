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

// Kiro CLI stop hook (wander) — starts background wander after each response.
// Wander uses δ-mem residuals to find spontaneous associations in turbogo.
// Pair with enrich-turn-wander to harvest insights on next prompt.

type hookInput struct {
	HookEventName     string `json:"hook_event_name"`
	AssistantResponse string `json:"assistant_response"`
}

func main() {
	var input hookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		os.Exit(0)
	}
	if input.HookEventName != "stop" || len(input.AssistantResponse) < 40 {
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

	client.StartWander(ctx, &pb.OwnerRequest{Owner: owner})
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
