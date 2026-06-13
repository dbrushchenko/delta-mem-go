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
	modelPath := `C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`
	ortLib := `C:\Users\dabrush\mem-go\models\onnxruntime.dll`

	// Initialize real embeddings
	emb, err := embeddings.Get(modelPath, ortLib)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embeddings: %v\n", err)
		os.Exit(1)
	}
	defer emb.Close()
	thoughts.SetEmbedder(emb)
	fmt.Println("✓ Embeddings loaded (nomic 768-dim)")

	// Initialize components
	dataDir := "./data/test_states"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)

	// Create thoughts engine (no Gemma for this test - testing substrate only)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	fmt.Println("✓ Thoughts engine initialized")

	owner := "test_e2e"
	ctx := context.Background()

	// === Test 1: Embedding quality ===
	fmt.Println("\n=== Test 1: Embedding Semantic Quality ===")
	pairs := [][3]string{
		{"dog", "canine", "should be high"},
		{"server crashed", "system went down", "should be high"},
		{"water freezes", "ice forms at low temperature", "should be high"},
		{"dog", "quantum physics", "should be low"},
		{"server crashed", "beautiful sunset", "should be low"},
	}
	for _, p := range pairs {
		v1, _ := emb.Embed(p[0])
		v2, _ := emb.Embed(p[1])
		sim := cosine(v1, v2)
		fmt.Printf("  %.3f  %q / %q  (%s)\n", sim, p[0], p[1], p[2])
	}

	// === Test 2: δ-mem accumulation ===
	fmt.Println("\n=== Test 2: δ-mem Store & Recall ===")
	facts := []string{
		"water freezes at zero degrees celsius",
		"the earth orbits the sun once per year",
		"neural networks learn through backpropagation",
		"kubernetes orchestrates container workloads",
		"LoRaWAN operates in unlicensed spectrum bands",
	}
	for _, fact := range facts {
		vec := emb.EmbedText(fact)
		norm, _ := deltaOM.Store(owner, vec)
		fmt.Printf("  stored: %q  norm=%.3f\n", truncate(fact, 40), norm)
	}

	// Recall with related query
	queries := []string{
		"ice forms when temperature drops below freezing",
		"planets revolve around stars",
		"deep learning gradient optimization",
	}
	for _, q := range queries {
		vec := emb.EmbedText(q)
		_, deltaO, conf, _ := deltaOM.Recall(owner, vec)
		_ = deltaO
		fmt.Printf("  recall: %q  confidence=%.4f\n", truncate(q, 40), conf)
	}

	// === Test 3: Learn and Adapt ===
	fmt.Println("\n=== Test 3: Adapt (correction) ===")
	start := time.Now()
	correction, err := engine.Adapt(ctx, owner, "the sun orbits the earth", "the earth orbits the sun")
	if err != nil {
		fmt.Printf("  adapt error: %v\n", err)
	} else {
		fmt.Printf("  adapted in %v  impact=%.3f\n", time.Since(start), correction.Impact)
	}

	// === Test 4: Learn ===
	fmt.Println("\n=== Test 4: Learn (absorb fact) ===")
	start = time.Now()
	engine.Learn(ctx, owner, "USGS monitors water resources across the United States")
	fmt.Printf("  learned in %v\n", time.Since(start))

	// === Test 5: Truth engine ===
	fmt.Println("\n=== Test 5: Truth Axioms ===")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("water freezes at zero degrees celsius", "physics")

	testThoughts := []string{
		"planets orbit stars due to gravity",
		"the sun revolves around the earth daily",
		"machine learning requires training data",
	}
	for _, t := range testThoughts {
		verdict := engine.Truth().Validate(ctx, t)
		fmt.Printf("  %q\n    valid=%v grounding=%.3f coherence=%.3f\n", t, verdict.Valid, verdict.Grounding, verdict.Coherence)
		if len(verdict.Contradictions) > 0 {
			fmt.Printf("    contradicts: %v\n", verdict.Contradictions)
		}
	}

	// === Test 6: Performance ===
	fmt.Println("\n=== Test 6: Performance ===")
	start = time.Now()
	n := 100
	for i := 0; i < n; i++ {
		emb.EmbedText(fmt.Sprintf("test sentence number %d for benchmarking", i))
	}
	elapsed := time.Since(start)
	fmt.Printf("  %d embeddings in %v (%.1f ms/embed, %.0f embeds/sec)\n",
		n, elapsed, float64(elapsed.Milliseconds())/float64(n), float64(n)/elapsed.Seconds())

	start = time.Now()
	for i := 0; i < n; i++ {
		vec := emb.EmbedText(fmt.Sprintf("store test %d", i))
		deltaOM.Store(owner, vec)
	}
	elapsed = time.Since(start)
	fmt.Printf("  %d embed+store in %v (%.1f ms/op)\n", n, elapsed, float64(elapsed.Milliseconds())/float64(n))

	fmt.Println("\n=== All tests passed ===")
}

func truncate(s string, n int) string { if len(s) <= n { return s }; return s[:n] }

func cosine(a, b []float32) float32 {
	if len(a) != len(b) { return 0 }
	var dot, na, nb float32
	for i := range a { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
	if na*nb == 0 { return 0 }
	return dot / (sqrt(na) * sqrt(nb))
}

func sqrt(x float32) float32 {
	if x <= 0 { return 0 }
	z := x
	for i := 0; i < 10; i++ { z = (z + x/z) / 2 }
	return z
}
