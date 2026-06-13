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

	dataDir := "./data/ibnn_test"
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	owner := "ibnn_test"

	// === Test IBNN directly ===
	fmt.Println("=== IBNN Forward Test ===")
	input := emb.EmbedText("neural networks learn through gradient descent")
	start := time.Now()
	out, err := ibnnOM.ForwardBatch(owner, [][]float32{input})
	fmt.Printf("  time: %v  err: %v\n", time.Since(start), err)
	if err == nil && len(out) > 0 {
		nonzero := 0
		for _, v := range out[0] {
			if v > 0 { nonzero++ }
		}
		fmt.Printf("  input dim: %d, output dim: %d\n", len(input), len(out[0]))
		fmt.Printf("  non-zero activations: %d/%d (%.1f%% sparse)\n",
			nonzero, len(out[0]), 100.0*float64(len(out[0])-nonzero)/float64(len(out[0])))
	}

	// IBNN sharpening effect
	fmt.Println("\n=== IBNN Sharpening Effect ===")
	texts := []string{"dog", "canine", "cat", "quantum physics"}
	vecs := make([][]float32, len(texts))
	ibnnVecs := make([][]float32, len(texts))
	for i, t := range texts {
		vecs[i] = emb.EmbedText(t)
		o, _ := ibnnOM.ForwardBatch(owner, [][]float32{vecs[i]})
		ibnnVecs[i] = o[0]
	}
	fmt.Printf("  Before IBNN: dog/canine=%.3f  dog/cat=%.3f  dog/physics=%.3f\n",
		cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2]), cosine(vecs[0], vecs[3]))
	fmt.Printf("  After IBNN:  dog/canine=%.3f  dog/cat=%.3f  dog/physics=%.3f\n",
		cosine(ibnnVecs[0], ibnnVecs[1]), cosine(ibnnVecs[0], ibnnVecs[2]), cosine(ibnnVecs[0], ibnnVecs[3]))

	// === Test Think() without Gemma ===
	fmt.Println("\n=== Think() without Gemma ===")
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	_, err = engine.Think(ctx, owner, []string{"LoggerNet", "AWS migration"})
	fmt.Printf("  result: %v\n", err)

	// === Manual Think substrate ===
	fmt.Println("\n=== Manual Think Substrate ===")
	seeds := []string{
		"infrastructure automation",
		"container orchestration",
		"monitoring and observability",
		"water resource data collection",
		"IoT sensor networks",
	}
	for _, s := range seeds {
		v := emb.EmbedText(s)
		deltaOM.Store(owner, v)
		turboOM.AddVector(owner, s, v)
	}

	// Combined probe — concept that spans multiple seeds
	probe := emb.EmbedText("automated water monitoring using IoT containers")
	_, deltaO, conf, _ := deltaOM.Recall(owner, probe)
	fmt.Printf("  δ-mem confidence: %.4f\n", conf)
	fmt.Printf("  δ-mem output norm: %.4f\n", norm(deltaO))

	// IBNN crystallize the δ-mem output
	crystal, _ := ibnnOM.ForwardBatch(owner, [][]float32{deltaO})
	if len(crystal) > 0 && len(crystal[0]) > 0 {
		nonzero := 0
		var maxVal float32
		for _, v := range crystal[0] {
			if v > 0 { nonzero++ }
			if v > maxVal { maxVal = v }
		}
		fmt.Printf("  IBNN crystallized: %d/%d non-zero (%.1f%% sparse), max=%.4f\n",
			nonzero, len(crystal[0]), 100.0*float64(len(crystal[0])-nonzero)/float64(len(crystal[0])), maxVal)

		// Search turbogo with crystallized vector
		ids, scores, _ := turboOM.SearchVector(owner, crystal[0], 5)
		fmt.Printf("  Turbogo search with CRYSTALLIZED vector:\n")
		for i, id := range ids {
			fmt.Printf("    [%.4f] %s\n", scores[i], id)
		}

		// Compare: search with RAW probe (no IBNN)
		ids2, scores2, _ := turboOM.SearchVector(owner, probe, 5)
		fmt.Printf("  Turbogo search with RAW probe (no IBNN):\n")
		for i, id := range ids2 {
			fmt.Printf("    [%.4f] %s\n", scores2[i], id)
		}
	}
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
	if na*nb == 0 { return 0 }
	return dot / (sqrt(na) * sqrt(nb))
}
func norm(v []float32) float32 { var s float32; for _, x := range v { s += x * x }; return sqrt(s) }
func sqrt(x float32) float32 { if x <= 0 { return 0 }; z := x; for i := 0; i < 10; i++ { z = (z + x/z) / 2 }; return z }
