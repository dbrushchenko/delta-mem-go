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
	fmt.Println("=== mem-go STRESS TEST ===\n")

	// Setup embeddings
	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL embeddings: %v\n", err)
		os.Exit(1)
	}
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	// Create engine components
	dataDir := "./data/stress_test"
	os.RemoveAll(dataDir)
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	owner := "stress"
	totalStart := time.Now()

	// Phase 1: Initiation
	fmt.Println("--- INITIATION (3 epochs) ---")
	data, err := os.ReadFile(`C:\Users\dabrush\mem-go\training_3.txt`)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL read training: %v\n", err)
		os.Exit(1)
	}
	initStart := time.Now()
	result, err := engine.Initiate(string(data), owner, thoughts.InitConfig{Epochs: 3, ChunkSize: 200})
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL initiate: %v\n", err)
		os.Exit(1)
	}
	initTime := time.Since(initStart)
	fmt.Printf("  ✓ %d chunks, %d epochs in %v\n", result.Chunks, result.Epochs, initTime)
	fmt.Printf("    norm=%.4f avg_conf=%.4f\n\n", result.FinalNorm, result.AvgConf)
	indexAfterInit := turboOM.Count(owner)

	// Phase 2: 20 Think() calls
	fmt.Println("--- 20 THINK CALLS ---")
	seeds := [][]string{
		{"LoggerNet AWS", "Ansible SSH"},
		{"Kubernetes pods", "monitoring"},
		{"certificate SSL", "renewal"},
		{"LoRaWAN sensors", "edge computing"},
		{"delta-rule memory", "neural networks"},
		{"Nagios alerting", "service health"},
		{"VMware snapshots", "disaster recovery"},
		{"Grafana dashboards", "Prometheus metrics"},
		{"CloudFormation stacks", "infrastructure as code"},
		{"Flagger canary", "mesh deployment"},
		{"Veeam backup", "restore point"},
		{"GitLab CI/CD", "pipeline automation"},
		{"Istio service mesh", "traffic routing"},
		{"Node-RED flows", "data transformation"},
		{"ChirpStack gateway", "packet forwarder"},
		{"Docker containers", "image registry"},
		{"Semaphore playbooks", "inventory management"},
		{"SSL chain validation", "trust anchors"},
		{"PostgreSQL replication", "high availability"},
		{"OneNote collaboration", "knowledge sharing"},
	}
	var thinkTotal time.Duration
	for i, s := range seeds {
		start := time.Now()
		thought, err := engine.Think(ctx, owner, s)
		elapsed := time.Since(start)
		thinkTotal += elapsed
		if err != nil {
			fmt.Printf("  [%02d] FAIL: %v\n", i+1, err)
			continue
		}
		fmt.Printf("  [%02d] %v  conf=%.3f nov=%.3f depth=%d  %s\n", i+1, elapsed, thought.Confidence, thought.Novelty, thought.Depth, truncate(thought.Idea, 80))
	}
	indexAfterThink := turboOM.Count(owner)
	fmt.Printf("\n  Index: %d after init → %d after 20 thinks\n", indexAfterInit, indexAfterThink)
	fmt.Printf("  Dedup check: %d entries (should be much less than %d = 20*neighbors)\n\n", indexAfterThink, 20*8)

	// Phase 3: 5 Adapt() calls
	fmt.Println("--- 5 ADAPT CALLS ---")
	adaptions := [][2]string{
		{"LoggerNet runs on Kubernetes", "LoggerNet runs on Windows EC2 instances"},
		{"Nagios uses Prometheus", "Nagios uses its own check engine with NRPE/NCPA"},
		{"Veeam stores to local disk", "Veeam stores to scale-out repositories with cloud tier"},
		{"SSL certs are self-signed", "SSL certs are issued by DOI CA with proper chain"},
		{"LoRa uses WiFi", "LoRa uses sub-GHz radio for long-range low-power communication"},
	}
	var adaptTotal time.Duration
	for i, a := range adaptions {
		start := time.Now()
		c, err := engine.Adapt(ctx, owner, a[0], a[1])
		elapsed := time.Since(start)
		adaptTotal += elapsed
		if err != nil {
			fmt.Printf("  [%d] FAIL: %v\n", i+1, err)
			continue
		}
		fmt.Printf("  [%d] %v  impact=%.4f\n", i+1, elapsed, c.Impact)
	}

	// Summary
	fmt.Println("\n=== RESULTS ===")
	fmt.Printf("  Total time:       %v\n", time.Since(totalStart))
	fmt.Printf("  Initiation:       %v\n", initTime)
	fmt.Printf("  Avg think time:   %v\n", thinkTotal/20)
	fmt.Printf("  Avg adapt time:   %v\n", adaptTotal/5)
	fmt.Printf("  Index size:       %d entries\n", turboOM.Count(owner))
	fmt.Printf("  Dedup effective:  %v (bounded vs %d theoretical max)\n", indexAfterThink < 20*8, 20*8)
	fmt.Println("\n=== STRESS TEST COMPLETE ===")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
