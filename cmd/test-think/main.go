package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/gemma"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

func main() {
	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil { fmt.Fprintf(os.Stderr, "embeddings: %v\n", err); os.Exit(1) }
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/think_test"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	gemmaClient := gemma.NewClient("gemma2:2b")

	engine := thoughts.New(deltaOM, ibnnOM, turboOM, gemmaClient)
	owner := "thinker"
	ctx := context.Background()

	// === Seed with knowledge ===
	fmt.Println("=== Seeding knowledge base ===")
	knowledge := []string{
		"USGS monitors water resources across the United States using sensor networks",
		"LoRaWAN is a low-power wide-area network protocol for IoT devices",
		"Kubernetes orchestrates containers across distributed clusters",
		"Delta-rule memory accumulates information through interference patterns",
		"Neural networks learn through gradient descent and backpropagation",
		"Real-time monitoring requires low latency data pipelines",
		"Edge computing processes data near the source to reduce bandwidth",
		"Semantic embeddings encode meaning as geometric relationships in vector space",
	}
	for _, k := range knowledge {
		vec := emb.EmbedText(k)
		deltaOM.Store(owner, vec)
		turboOM.AddVector(owner, k[:50], vec)
	}
	fmt.Printf("  Stored %d facts\n", len(knowledge))

	// === Add truth axioms ===
	fmt.Println("\n=== Setting truth axioms ===")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("water freezes at zero degrees celsius", "physics")
	engine.Truth().AddAxiom("information cannot travel faster than light", "physics")
	fmt.Println("  3 axioms set")

	// === Test 1: Think() with related seeds ===
	fmt.Println("\n=== Think Test 1: Related seeds ===")
	start := time.Now()
	thought, err := engine.Think(ctx, owner, []string{
		"IoT sensors for water monitoring",
		"edge computing at remote locations",
	})
	elapsed := time.Since(start)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
	} else {
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Depth: %d iterations\n", thought.Depth)
		fmt.Printf("  Confidence: %.4f\n", thought.Confidence)
		fmt.Printf("  Novelty: %.4f\n", thought.Novelty)
		fmt.Printf("  Grounding: %.4f\n", thought.Grounding)
		fmt.Printf("  Valid: %v\n", thought.Valid)
		fmt.Printf("  Neighbors: %v\n", thought.Neighbors)
		fmt.Printf("  Idea:\n    %s\n", strings.ReplaceAll(thought.Idea, "\n", "\n    "))
	}

	// === Test 2: Think() with divergent seeds (force creativity) ===
	fmt.Println("\n=== Think Test 2: Divergent seeds (force novelty) ===")
	start = time.Now()
	thought2, err := engine.Think(ctx, owner, []string{
		"delta-rule memory interference patterns",
		"water resource monitoring networks",
		"container orchestration",
	})
	elapsed = time.Since(start)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
	} else {
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Depth: %d iterations\n", thought2.Depth)
		fmt.Printf("  Confidence: %.4f\n", thought2.Confidence)
		fmt.Printf("  Novelty: %.4f\n", thought2.Novelty)
		fmt.Printf("  Valid: %v\n", thought2.Valid)
		fmt.Printf("  Idea:\n    %s\n", strings.ReplaceAll(thought2.Idea, "\n", "\n    "))
	}

	// === Test 3: Think() that should trigger truth rejection ===
	fmt.Println("\n=== Think Test 3: Seeds that might produce truth violation ===")
	start = time.Now()
	thought3, err := engine.Think(ctx, owner, []string{
		"instant communication across galaxies",
		"faster than light data transfer",
	})
	elapsed = time.Since(start)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
	} else {
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Depth: %d\n", thought3.Depth)
		fmt.Printf("  Valid: %v\n", thought3.Valid)
		fmt.Printf("  Grounding: %.4f\n", thought3.Grounding)
		fmt.Printf("  Idea:\n    %s\n", strings.ReplaceAll(thought3.Idea, "\n", "\n    "))
	}

	// === Test 4: Iterative re-entry (does depth increase on surprise?) ===
	fmt.Println("\n=== Think Test 4: Novel topic (should go deeper) ===")
	start = time.Now()
	thought4, err := engine.Think(ctx, owner, []string{
		"consciousness emerging from recursive self-reference",
	})
	elapsed = time.Since(start)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
	} else {
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Depth: %d (more = thought harder)\n", thought4.Depth)
		fmt.Printf("  Confidence: %.4f\n", thought4.Confidence)
		fmt.Printf("  Novelty: %.4f\n", thought4.Novelty)
		fmt.Printf("  Idea:\n    %s\n", strings.ReplaceAll(thought4.Idea, "\n", "\n    "))
	}

	fmt.Println("\n=== All Think tests complete ===")
	_ = bufio.NewReader(os.Stdin)
}
