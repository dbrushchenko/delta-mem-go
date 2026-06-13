package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

func main() {
	fmt.Println("=== mem-go Full Lifecycle Test ===\n")

	// Setup
	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil { fmt.Fprintf(os.Stderr, "FAIL embeddings: %v\n", err); os.Exit(1) }
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/lifecycle"
	os.RemoveAll(dataDir)
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	owner := "lifecycle"

	// Phase 1: Initiation
	fmt.Println("--- Phase 1: INITIATION ---")
	data, _ := os.ReadFile(`C:\Users\dabrush\mem-go\data\training_data\training_3.txt`)
	start := time.Now()
	result, err := engine.Initiate(string(data), owner, thoughts.InitConfig{Epochs: 3, ChunkSize: 200})
	if err != nil { fmt.Fprintf(os.Stderr, "FAIL initiate: %v\n", err); os.Exit(1) }
	fmt.Printf("  ✓ %d chunks, %d epochs in %v\n", result.Chunks, result.Epochs, time.Since(start))
	fmt.Printf("    state_norm=%.4f  avg_conf=%.4f\n\n", result.FinalNorm, result.AvgConf)

	// Phase 2: Add axioms
	fmt.Println("--- Phase 2: TRUTH AXIOMS ---")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("LoggerNet runs on Windows EC2 instances", "infrastructure")
	fmt.Println("  ✓ 2 axioms set\n")

	// Phase 3: Think (substrate-only, no Gemma)
	fmt.Println("--- Phase 3: THINK (substrate synthesis) ---")
	tests := [][]string{
		{"LoggerNet data collection", "AWS cloud migration"},
		{"Loki log monitoring", "Kubernetes mesh"},
		{"LoRaWAN IoT sensors", "real-time data"},
	}
	for _, seeds := range tests {
		start = time.Now()
		thought, err := engine.Think(ctx, owner, seeds)
		if err != nil { fmt.Printf("  FAIL: %v\n", err); continue }
		fmt.Printf("  Seeds: %v\n", seeds)
		fmt.Printf("  Time=%v Depth=%d Conf=%.4f Novelty=%.4f Valid=%v\n", time.Since(start), thought.Depth, thought.Confidence, thought.Novelty, thought.Valid)
		fmt.Printf("  Idea: %s\n\n", truncate(thought.Idea, 120))
	}

	// Phase 4: Adapt (correction)
	fmt.Println("--- Phase 4: ADAPT (correction) ---")
	start = time.Now()
	c, err := engine.Adapt(ctx, owner, "LoggerNet runs on Kubernetes pods", "LoggerNet runs on Windows EC2 instances managed by CloudFormation")
	if err != nil { fmt.Fprintf(os.Stderr, "FAIL adapt: %v\n", err); os.Exit(1) }
	fmt.Printf("  ✓ Corrected in %v  impact=%.4f\n\n", time.Since(start), c.Impact)

	// Phase 5: Learn new facts
	fmt.Println("--- Phase 5: LEARN (new facts) ---")
	facts := []string{
		"The mesh uses Flagger for canary deployments",
		"CertOps manages SSL certificate lifecycle across all sites",
	}
	for _, f := range facts {
		engine.Learn(ctx, owner, f)
		fmt.Printf("  ✓ %s\n", f)
	}
	fmt.Println()

	// Phase 6: Think again (should reflect learned knowledge)
	fmt.Println("--- Phase 6: THINK AGAIN (post-learning) ---")
	start = time.Now()
	thought, err := engine.Think(ctx, owner, []string{"certificate management", "deployment automation"})
	if err != nil { fmt.Printf("  FAIL: %v\n", err) } else {
		fmt.Printf("  Time=%v Depth=%d Conf=%.4f Novelty=%.4f\n", time.Since(start), thought.Depth, thought.Confidence, thought.Novelty)
		fmt.Printf("  Idea: %s\n\n", truncate(thought.Idea, 120))
	}

	// Phase 7: Truth validation
	fmt.Println("--- Phase 7: TRUTH VALIDATION ---")
	checks := []string{
		"LoggerNet runs on Windows instances in AWS",
		"LoggerNet runs on Kubernetes pods",
		"The earth orbits the sun providing seasons",
		"The sun orbits the earth",
	}
	for _, c := range checks {
		v := engine.Truth().Validate(ctx, c)
		status := "✓ VALID"
		if !v.Valid { status = "✗ REJECTED" }
		fmt.Printf("  %s  %q", status, c)
		if !v.Valid { fmt.Printf(" (reason: %s)", v.Reason) }
		fmt.Println()
	}

	fmt.Println("\n=== LIFECYCLE TEST COMPLETE ===")
}

func truncate(s string, n int) string { if len(s) <= n { return s }; return s[:n] + "..." }
