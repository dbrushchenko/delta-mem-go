package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

func main() {
	addr := flag.String("addr", "localhost:19090", "gRPC server address")
	owner := flag.String("owner", os.Getenv("USERNAME"), "owner name")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}

	conn, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
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

	case "health":
		resp, err := client.Health(ctx, &pb.Empty{})
		if err != nil { fatal("health: %v", err) }
		fmt.Printf("owners=%d stores=%d recalls=%d uptime=%s\n",
			resp.OwnersActive, resp.TotalStores, resp.TotalRecalls, resp.Uptime)

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
	fmt.Println(`mem-cli — gRPC client for δ-mem-go

Commands:
  store --key <key> --content <text>   Store a fact
  recall <query>                        Recall related knowledge
  think <seed1> <seed2> ...            Synthesize a thought
  adapt <wrong> <right>                Correct a misconception
  learn <fact>                         Absorb a new fact
  health                               Check server status

Flags:
  --addr    gRPC address (default: localhost:19090)
  --owner   Owner name (default: $USERNAME)`)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
