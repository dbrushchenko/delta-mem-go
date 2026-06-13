package main

import (
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

	dataDir := "./data/initiated"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)

	owner := "dabrush"

	// Load training data
	data, _ := os.ReadFile(`C:\Users\dabrush\mem-go\training_3.txt`)
	fmt.Printf("Training data: %d bytes\n", len(data))

	// Run initiation
	fmt.Println("\n=== Initiation (3 epochs) ===")
	start := time.Now()
	result, err := engine.Initiate(string(data), owner, thoughts.InitConfig{
		Epochs:       3,
		LearningRate: 0.01,
		ChunkSize:    200,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "initiation failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Chunks: %d\n", result.Chunks)
	fmt.Printf("  Epochs: %d\n", result.Epochs)
	fmt.Printf("  Duration: %v\n", result.Duration)
	fmt.Printf("  State norm: %.4f\n", result.FinalNorm)
	fmt.Printf("  Avg confidence: %.4f\n", result.AvgConf)
	fmt.Printf("  Total time (incl embedding): %v\n", time.Since(start))

	// Now test recall quality post-initiation
	fmt.Println("\n=== Post-initiation recall ===")
	queries := []string{
		"LoggerNet migration to AWS EC2",
		"Ansible SSH automation with Semaphore",
		"Loki log collection and dashboards",
		"Kubernetes Istio service mesh",
		"certificate renewal and SSL",
		"cooking pasta recipes",
	}
	for _, q := range queries {
		vec := emb.EmbedText(q)
		_, _, conf, _ := deltaOM.Recall(owner, vec)
		ids, scores, _ := turboOM.SearchVector(owner, vec, 2)
		top := ""
		if len(ids) > 0 {
			top = ids[0]
			if len(top) > 60 { top = top[:60] }
		}
		topScore := float32(0)
		if len(scores) > 0 { topScore = scores[0] }
		fmt.Printf("  [conf=%.4f] %q → [%.3f] %s\n", conf, q, topScore, top)
	}
}
