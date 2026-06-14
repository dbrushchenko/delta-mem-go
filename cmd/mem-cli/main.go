package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

func main() {
	addr := flag.String("addr", "http://localhost:18080", "server address")
	grpcAddr := flag.String("grpc-addr", "localhost:19090", "gRPC address")
	owner := flag.String("owner", os.Getenv("USERNAME"), "owner name")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}

	// Auto-enroll: get or create token transparently
	token := getOrEnrollToken(*addr, *owner)

	conn, err := grpc.NewClient(*grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(tokenInterceptor(token)),
	)
	if err != nil { fatal("connect: %v", err) }
	defer conn.Close()
	client := pb.NewDeltaMemClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	switch args[0] {
	case "store":
		key, content := parseKV(args[1:])
		resp, err := client.Store(ctx, &pb.StoreRequest{Owner: *owner, Key: key, Content: content})
		if err != nil { fatal("store: %v", err) }
		fmt.Printf("stored: norm=%.4f\n", resp.StateNorm)

	case "recall":
		query := strings.Join(args[1:], " ")
		resp, err := client.Recall(ctx, &pb.RecallRequest{Owner: *owner, Query: query})
		if err != nil { fatal("recall: %v", err) }
		fmt.Printf("confidence=%.4f\n", resp.Confidence)

	case "think":
		seeds := args[1:]
		if len(seeds) == 0 { fatal("think requires seeds") }
		resp, err := client.Think(ctx, &pb.ThinkRequest{Owner: *owner, Seeds: seeds})
		if err != nil { fatal("think: %v", err) }
		fmt.Printf("idea: %s\ndepth=%d conf=%.4f novelty=%.4f valid=%v\n",
			resp.Idea, resp.Depth, resp.Confidence, resp.Novelty, resp.Valid)

	case "adapt":
		if len(args) < 3 { fatal("adapt <wrong> <right>") }
		resp, err := client.Adapt(ctx, &pb.AdaptRequest{Owner: *owner, Wrong: args[1], Right: args[2]})
		if err != nil { fatal("adapt: %v", err) }
		fmt.Printf("adapted: impact=%.4f\n", resp.Impact)

	case "learn":
		fact := strings.Join(args[1:], " ")
		_, err := client.Learn(ctx, &pb.LearnRequest{Owner: *owner, Fact: fact})
		if err != nil { fatal("learn: %v", err) }
		fmt.Println("learned")

	case "forget":
		what := strings.Join(args[1:], " ")
		_, err := client.Forget(ctx, &pb.ForgetRequest{Owner: *owner, What: what})
		if err != nil { fatal("forget: %v", err) }
		fmt.Println("forgotten")

	case "undo":
		if len(args) < 3 { fatal("undo <original-wrong> <original-right>") }
		// Undo calls Adapt in reverse
		resp, err := client.Adapt(ctx, &pb.AdaptRequest{Owner: *owner, Wrong: args[2], Right: args[1]})
		if err != nil { fatal("undo: %v", err) }
		fmt.Printf("undone: impact=%.4f\n", resp.Impact)

	case "axiom":
		statement := strings.Join(args[1:], " ")
		_, err := client.AddAxiom(ctx, &pb.AxiomRequest{Statement: statement, Domain: ""})
		if err != nil { fatal("axiom: %v", err) }
		fmt.Printf("axiom set: %s\n", statement)

	case "wander":
		if len(args) < 2 { fatal("wander start|stop|harvest") }
		switch args[1] {
		case "start":
			_, err := client.StartWander(ctx, &pb.OwnerRequest{Owner: *owner})
			if err != nil { fatal("wander start: %v", err) }
			fmt.Println("wander started")
		case "stop":
			_, err := client.StopWander(ctx, &pb.OwnerRequest{Owner: *owner})
			if err != nil { fatal("wander stop: %v", err) }
			fmt.Println("wander stopped")
		case "harvest":
			resp, err := client.HarvestWander(ctx, &pb.OwnerRequest{Owner: *owner})
			if err != nil { fatal("wander harvest: %v", err) }
			if resp == nil || len(resp.Thoughts) == 0 {
				fmt.Println("no spontaneous thoughts yet")
			} else {
				for i, t := range resp.Thoughts {
					fmt.Printf("  [%d] %s\n", i+1, t.Idea)
				}
			}
		default:
			fatal("wander start|stop|harvest")
		}

	case "initiate":
		// HTTP path — sends training file to /initiate
		filePath := ""
		for i, a := range args[1:] {
			if a == "--file" && i+1 < len(args[1:])-1 { filePath = args[i+2] }
		}
		if filePath == "" && len(args) > 1 { filePath = args[1] }
		if filePath == "" { fatal("initiate --file <path-to-training-data>") }
		data, err := os.ReadFile(filePath)
		if err != nil { fatal("read file: %v", err) }
		body := fmt.Sprintf(`{"owner":"%s","text":%s,"epochs":5}`, *owner, jsonEscape(string(data)))
		resp, err := http.Post(*addr+"/initiate", "application/json", strings.NewReader(body))
		if err != nil { fatal("initiate: %v", err) }
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		fmt.Println(string(out))

	case "health":
		resp, err := client.Health(ctx, &pb.Empty{})
		if err != nil { fatal("health: %v", err) }
		fmt.Printf("owners=%d stores=%d recalls=%d uptime=%s\n",
			resp.OwnersActive, resp.TotalStores, resp.TotalRecalls, resp.Uptime)

	case "create-token":
		if len(args) < 2 { fatal("create-token <owner-name>") }
		tokenOwner := args[1]
		body := fmt.Sprintf(`{"owner":"%s"}`, tokenOwner)
		resp, err := http.Post(*addr+"/enroll", "application/json", strings.NewReader(body))
		if err != nil { fatal("enroll: %v", err) }
		defer resp.Body.Close()
		var result struct{ Token string `json:"token"`; Owner string `json:"owner"` }
		json.NewDecoder(resp.Body).Decode(&result)
		fmt.Printf("owner: %s\ntoken: %s\n\nUsage:\n  mem-cli --grpc-addr %s store --key K --content C\n  # Token auto-saved. Or set MEMGO_TOKEN=%s\n", result.Owner, result.Token, *grpcAddr, result.Token)

	default:
		usage()
		os.Exit(1)
	}
}

func parseKV(args []string) (key, content string) {
	for i, a := range args {
		if a == "--key" && i+1 < len(args) { key = args[i+1] }
		if a == "--content" && i+1 < len(args) { content = strings.Join(args[i+1:], " "); break }
	}
	if key == "" && len(args) > 0 { key = args[0] }
	if content == "" && len(args) > 1 { content = strings.Join(args[1:], " ") }
	return
}

func usage() {
	fmt.Println(`mem-cli — client for δ-mem-go

Data Operations (gRPC):
  store --key <key> --content <text>   Store a fact
  recall <query>                        Recall related knowledge
  think <seed1> <seed2> ...            Synthesize a thought
  adapt <wrong> <right>                Correct a misconception
  learn <fact>                         Absorb a new fact
  forget <text>                        Fade a memory
  undo <original-wrong> <original-right>  Reverse a correction
  axiom <statement>                    Add immutable truth constraint
  wander start|stop|harvest            Spontaneous thought control
  health                               Check server status

Setup Operations (HTTP):
  initiate --file <path>               Train on domain data (first-time)
  create-token <owner>                 Generate service token for agent

Flags:
  --addr       HTTP address (default: http://localhost:18080)
  --grpc-addr  gRPC address (default: localhost:19090)
  --owner      Owner name for auto-enrollment (default: $USERNAME)`)
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}

// --- Token management ---

func tokenPath() string {
	dir := filepath.Join(os.Getenv("USERPROFILE"), ".memgo")
	if dir == filepath.Join("", ".memgo") {
		dir = filepath.Join(os.Getenv("HOME"), ".memgo")
	}
	os.MkdirAll(dir, 0700)
	return filepath.Join(dir, "token")
}

// getOrEnrollToken reads saved token or auto-enrolls transparently.
func getOrEnrollToken(serverAddr, owner string) string {
	path := tokenPath()
	// Try reading existing token
	if data, err := os.ReadFile(path); err == nil {
		tok := strings.TrimSpace(string(data))
		if tok != "" { return tok }
	}
	// Auto-enroll via HTTP
	body := fmt.Sprintf(`{"owner":"%s"}`, owner)
	resp, err := http.Post(serverAddr+"/enroll", "application/json", strings.NewReader(body))
	if err != nil { return "" } // server not available, proceed without token
	defer resp.Body.Close()
	var result struct{ Token string `json:"token"` }
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Token != "" {
		os.WriteFile(path, []byte(result.Token), 0600)
	}
	return result.Token
}

// tokenInterceptor adds the bearer token to every gRPC call.
func tokenInterceptor(token string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if token != "" {
			ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+token)
		}
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}
