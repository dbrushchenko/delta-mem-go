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
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

func main() {
	modelPath := `C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`
	ortLib := `C:\Users\dabrush\mem-go\models\onnxruntime.dll`
	trainFile := `C:\Users\dabrush\mem-go\training_3.txt`

	emb, err := embeddings.Get(modelPath, ortLib)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embeddings: %v\n", err)
		os.Exit(1)
	}
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/train_test"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)

	owner := "training"
	ctx := context.Background()

	// === Load and chunk training data ===
	fmt.Printf("Loading %s...\n", trainFile)
	data, _ := os.ReadFile(trainFile)
	chunks := chunkText(string(data), 200) // ~200 char chunks
	fmt.Printf("Chunks: %d\n\n", len(chunks))

	// === Ingest ===
	fmt.Println("=== Ingesting into δ-mem + turbogo ===")
	start := time.Now()
	for i, chunk := range chunks {
		vec := emb.EmbedText(chunk)
		deltaOM.Store(owner, vec)
		turboOM.AddVector(owner, fmt.Sprintf("chunk_%d", i), vec)
	}
	elapsed := time.Since(start)
	fmt.Printf("Ingested %d chunks in %v (%.1f ms/chunk, %.0f chunks/sec)\n",
		len(chunks), elapsed, float64(elapsed.Milliseconds())/float64(len(chunks)),
		float64(len(chunks))/elapsed.Seconds())

	m, _ := deltaOM.Get(owner)
	fmt.Printf("δ-mem state norm: %.4f\n\n", m.StateNorm())

	// === Recall tests ===
	fmt.Println("=== Recall Quality (querying with related concepts) ===")
	queries := []struct{ q, expect string }{
		{"LoggerNet migration to AWS", "should recall AWS/EC2 infra work"},
		{"Ansible automation with Semaphore", "should recall Semaphore/SSH config"},
		{"LoRaWAN sensor deployment", "should recall IoT/sensor work"},
		{"Loki log collection dashboard", "should recall cawsc-loki work"},
		{"Kubernetes pod scheduling", "should recall k8s/mesh work"},
		{"certificate renewal process", "should recall SSL/cert work"},
		{"completely unrelated topic about cooking pasta", "should have low confidence"},
	}

	for _, qt := range queries {
		vec := emb.EmbedText(qt.q)
		_, _, conf, _ := deltaOM.Recall(owner, vec)

		// Also search turbogo for nearest neighbors
		ids, scores, _ := turboOM.SearchVector(owner, vec, 3)
		var topChunks []string
		for i, id := range ids {
			idx := 0
			fmt.Sscanf(id, "chunk_%d", &idx)
			preview := chunks[idx]
			if len(preview) > 80 {
				preview = preview[:80]
			}
			topChunks = append(topChunks, fmt.Sprintf("    [%.3f] %s", scores[i], preview))
		}

		fmt.Printf("  Q: %q\n  δ-mem confidence: %.4f  (%s)\n  Top matches:\n%s\n\n",
			qt.q, conf, qt.expect, strings.Join(topChunks, "\n"))
	}

	// === Adapt test ===
	fmt.Println("=== Adapt: Correct a misconception ===")
	c, _ := engine.Adapt(ctx, owner,
		"LoggerNet runs on Kubernetes",
		"LoggerNet runs on EC2 Windows instances managed by CloudFormation")
	fmt.Printf("  impact=%.4f\n\n", c.Impact)

	// === Learn test ===
	fmt.Println("=== Learn: Absorb new facts ===")
	newFacts := []string{
		"The new Istio service mesh uses mTLS for inter-pod encryption",
		"USGS water monitoring stations use Campbell Scientific dataloggers",
		"The delta-mem-go system provides per-owner persistent memory",
	}
	for _, f := range newFacts {
		engine.Learn(ctx, owner, f)
		fmt.Printf("  learned: %s\n", f)
	}

	// === Post-learn recall ===
	fmt.Println("\n=== Post-learn recall ===")
	vec := emb.EmbedText("persistent memory for AI agents")
	ids, scores, _ := turboOM.SearchVector(owner, vec, 3)
	for i, id := range ids {
		idx := 0
		fmt.Sscanf(id, "chunk_%d", &idx)
		if idx < len(chunks) {
			preview := chunks[idx]
			if len(preview) > 80 {
				preview = preview[:80]
			}
			fmt.Printf("  [%.3f] %s\n", scores[i], preview)
		} else {
			fmt.Printf("  [%.3f] %s\n", scores[i], id)
		}
	}

	fmt.Printf("\n=== Complete. Total vectors in index: %d ===\n", len(chunks)+len(newFacts)+1)
}

func chunkText(text string, targetLen int) []string {
	scanner := bufio.NewScanner(strings.NewReader(text))
	var chunks []string
	var current strings.Builder

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			if current.Len() > 0 {
				chunks = append(chunks, current.String())
				current.Reset()
			}
			continue
		}
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(line)
		if current.Len() >= targetLen {
			chunks = append(chunks, current.String())
			current.Reset()
		}
	}
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}
	return chunks
}
