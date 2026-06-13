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
	emb, _ := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/substrate_test"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)

	// NO GEMMA — substrate only
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	owner := "substrate"
	ctx := context.Background()

	// Seed knowledge
	knowledge := []string{
		"USGS monitors water resources using sensor networks across the US",
		"LoRaWAN enables low-power wide-area IoT communication",
		"Kubernetes orchestrates containers across distributed clusters",
		"Delta-rule memory accumulates associations through interference",
		"Edge computing reduces latency by processing data near the source",
		"Semantic embeddings encode meaning as vector geometry",
		"Real-time monitoring requires low-latency data pipelines",
		"Certificate management ensures secure encrypted connections",
		"Ansible automates infrastructure configuration via SSH",
		"LoggerNet collects data from Campbell Scientific dataloggers",
	}
	fmt.Printf("Seeding %d facts...\n", len(knowledge))
	for _, k := range knowledge {
		v := emb.EmbedText(k)
		deltaOM.Store(owner, v)
		turboOM.AddVector(owner, k, v)
	}

	// Set axioms
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("information cannot travel faster than light", "physics")

	// Think tests — no Gemma, pure substrate
	tests := []struct{ name string; seeds []string }{
		{"Related", []string{"IoT water monitoring", "edge computing sensors"}},
		{"Divergent", []string{"container orchestration", "water resource data", "neural memory"}},
		{"Novel", []string{"consciousness emerging from recursive self-reference"}},
		{"Familiar", []string{"Ansible automation SSH configuration"}},
	}

	for _, t := range tests {
		fmt.Printf("\n=== %s ===\n  Seeds: %v\n", t.name, t.seeds)
		start := time.Now()
		thought, err := engine.Think(ctx, owner, t.seeds)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			continue
		}
		fmt.Printf("  Time: %v\n", elapsed)
		fmt.Printf("  Depth: %d\n", thought.Depth)
		fmt.Printf("  Confidence: %.4f\n", thought.Confidence)
		fmt.Printf("  Novelty: %.4f\n", thought.Novelty)
		fmt.Printf("  Valid: %v\n", thought.Valid)
		fmt.Printf("  Idea: %s\n", thought.Idea)
		if len(thought.Neighbors) > 0 {
			fmt.Printf("  Neighbors: %v\n", thought.Neighbors[:min(len(thought.Neighbors), 3)])
		}
	}
}

func min(a, b int) int { if a < b { return a }; return b }
