package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/nli"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

func main() {
	emb, _ := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	defer emb.Close()
	thoughts.SetEmbedder(emb)
	os.MkdirAll("./data/nli_test", 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, "./data/nli_test")
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, "./data/nli_test")
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)

	checker, err := nli.New(`C:\Users\dabrush\mem-go\models\nli-deberta.onnx`, `C:\Users\dabrush\mem-go\models\nli-tokenizer.json`)
	if err != nil { fmt.Println("NLI load error:", err); return }
	engine.Truth().SetNLI(checker)
	fmt.Println("NLI loaded")

	// Direct NLI test
	fmt.Println("\n=== Direct NLI Check ===")
	pairs := [][2]string{
		{"the earth orbits the sun", "the sun revolves around the earth"},
		{"the earth orbits the sun", "the earth orbits the sun providing seasons"},
		{"water freezes at zero degrees", "water boils at zero degrees"},
	}
	for _, p := range pairs {
		label, conf := checker.Check(p[0], p[1])
		fmt.Printf("  %s (%.3f)  %q vs %q\n", label, conf, p[0], p[1])
	}

	// Truth engine test with axioms
	fmt.Println("\n=== Truth Validation with NLI ===")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	ctx := context.Background()
	tests := []string{
		"the sun revolves around the earth",
		"the earth orbits the sun providing seasons",
		"planets move around stars due to gravity",
		"the sun is the center of our solar system",
	}
	for _, t := range tests {
		v := engine.Truth().Validate(ctx, t)
		status := "✓ VALID"
		if !v.Valid { status = "✗ REJECTED" }
		fmt.Printf("  %s  %q", status, t)
		if !v.Valid { fmt.Printf("  (%s)", v.Reason) }
		fmt.Println()
	}
}
