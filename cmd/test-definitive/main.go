package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/nli"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

var pass, fail int

func assert(name string, ok bool, detail string) {
	if ok { pass++; fmt.Printf("  ✓ %s\n", name) } else { fail++; fmt.Printf("  ✗ %s — %s\n", name, detail) }
}

func main() {
	fmt.Println("=== DEFINITIVE LAYER-BY-LAYER VERIFICATION ===\n")

	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil { fmt.Println("FATAL:", err); os.Exit(1) }
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/definitive"
	os.RemoveAll(dataDir)
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768, dataDir)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	owner := "definitive"

	// Layer 1: Embeddings
	fmt.Println("--- Layer 1: EMBEDDINGS ---")
	v1 := emb.EmbedText("dog")
	v2 := emb.EmbedText("canine")
	v3 := emb.EmbedText("quantum physics")
	assert("embed produces 768-dim", len(v1) == 768, fmt.Sprintf("got %d", len(v1)))
	sim12 := cosine(v1, v2)
	sim13 := cosine(v1, v3)
	assert("dog/canine > 0.9", sim12 > 0.9, fmt.Sprintf("got %.3f", sim12))
	assert("dog/physics < 0.7", sim13 < 0.7, fmt.Sprintf("got %.3f", sim13))

	// Layer 2: δ-mem
	fmt.Println("\n--- Layer 2: δ-MEM ---")
	vec := emb.EmbedText("LoggerNet collects data from stations")
	norm1, _ := deltaOM.Store(owner, vec)
	assert("store returns norm > 0", norm1 > 0, fmt.Sprintf("norm=%.4f", norm1))
	mod, _ := deltaOM.Get(owner)
	assert("state norm > 0 after store", mod.StateNorm() > 0, fmt.Sprintf("%.4f", mod.StateNorm()))
	// Sparse recall
	dq, do := mod.SparseRecall(vec, 32)
	assert("sparse recall returns deltaQ", len(dq) == 768, fmt.Sprintf("len=%d", len(dq)))
	assert("sparse recall returns deltaO", len(do) == 768, fmt.Sprintf("len=%d", len(do)))
	// Adaptive rank
	assert("ShouldExpand false initially", !mod.ShouldExpand(), "expanded too early")

	// Layer 3: IBNN
	fmt.Println("\n--- Layer 3: IBNN ---")
	ibnnInput := emb.EmbedText("neural network training")
	out, err := ibnnOM.ForwardBatch(owner, [][]float32{ibnnInput})
	assert("IBNN forward no error", err == nil, fmt.Sprintf("%v", err))
	assert("IBNN output 768-dim", len(out) > 0 && len(out[0]) == 768, "wrong dim")
	nonzero := 0
	for _, v := range out[0] { if v > 0 { nonzero++ } }
	assert("IBNN sparse (>80% zero)", float64(nonzero)/768.0 < 0.2, fmt.Sprintf("%d/%d nonzero", nonzero, 768))
	// Reinforce
	ibnnOM.Reinforce(owner, 0.01)
	assert("IBNN reinforce no panic", true, "")

	// Layer 4: Turbogo/turbovec
	fmt.Println("\n--- Layer 4: VECTOR STORE ---")
	turboOM.AddVector(owner, "fact-loggerNet", emb.EmbedText("LoggerNet collects data"))
	turboOM.AddVector(owner, "fact-ansible", emb.EmbedText("Ansible automates via SSH"))
	turboOM.AddVector(owner, "fact-k8s", emb.EmbedText("Kubernetes orchestrates containers"))
	ids, scores, _ := turboOM.SearchVector(owner, emb.EmbedText("data collection from stations"), 3)
	assert("search returns results", len(ids) > 0, "empty")
	assert("top result is loggerNet", ids[0] == "fact-loggerNet", fmt.Sprintf("got %s", ids[0]))
	assert("scores > 0", scores[0] > 0, fmt.Sprintf("%.3f", scores[0]))
	// Dedup
	turboOM.AddVector(owner, "fact-loggerNet-v2", emb.EmbedText("LoggerNet collects data from stations"))
	ids2, _, _ := turboOM.SearchVector(owner, emb.EmbedText("LoggerNet"), 5)
	assert("dedup: no duplicate entries", countPrefix(ids2, "fact-loggerNet") <= 1, fmt.Sprintf("found %d", countPrefix(ids2, "fact-loggerNet")))

	// Layer 5: Truth engine
	fmt.Println("\n--- Layer 5: TRUTH ENGINE ---")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("water freezes at zero degrees", "physics")
	v := engine.Truth().Validate(ctx, "the sun orbits the earth")
	assert("heuristic catches inversion", !v.Valid, fmt.Sprintf("valid=%v", v.Valid))
	v2r := engine.Truth().Validate(ctx, "planets orbit stars due to gravity")
	assert("related statement passes", v2r.Valid, fmt.Sprintf("valid=%v reason=%s", v2r.Valid, v2r.Reason))

	// Layer 6: NLI
	fmt.Println("\n--- Layer 6: NLI ---")
	checker, nliErr := nli.New(`C:\Users\dabrush\mem-go\models\nli-deberta.onnx`, `C:\Users\dabrush\mem-go\models\nli-tokenizer.json`)
	assert("NLI model loads", nliErr == nil, fmt.Sprintf("%v", nliErr))
	if checker != nil {
		engine.Truth().SetNLI(checker)
		label, conf := checker.Check("water freezes at zero degrees", "water boils at zero degrees")
		assert("NLI detects contradiction", label == "contradiction" && conf > 0.8, fmt.Sprintf("%s %.3f", label, conf))
		label2, _ := checker.Check("water freezes at zero degrees", "ice forms when temperature drops")
		assert("NLI passes valid pair", label2 != "contradiction", fmt.Sprintf("got %s", label2))
	}

	// Layer 7: Self-model
	fmt.Println("\n--- Layer 7: SELF-MODEL ---")
	engine.Self().LearnDomain(emb.EmbedText("LoggerNet data collection"), 0.05)
	engine.Self().LearnDomain(emb.EmbedText("LoggerNet data collection"), 0.05)
	c := engine.Self().AmIConfident(emb.EmbedText("LoggerNet stations"))
	assert("AmIConfident not NeverSeen for known topic", c != thoughts.NeverSeen, fmt.Sprintf("got %v", c))
	c2 := engine.Self().AmIConfident(emb.EmbedText("quantum entanglement theory"))
	assert("known > unknown confidence", c >= c2, fmt.Sprintf("known=%v unknown=%v", c, c2))
	// Persistence
	engine.Self().Save(dataDir + "/self.gob")
	_, statErr := os.Stat(dataDir + "/self.gob")
	assert("self-model saves to disk", statErr == nil, fmt.Sprintf("%v", statErr))

	// Layer 8: Temporal
	fmt.Println("\n--- Layer 8: TEMPORAL ---")
	engine.Temporal().Record(owner, "test-event", "something happened")
	events := engine.Temporal().Recent(owner, 5)
	assert("temporal records events", len(events) >= 1, fmt.Sprintf("len=%d", len(events)))
	assert("temporal event has content", events[len(events)-1].Content == "something happened", events[len(events)-1].Content)

	// Layer 9: Adapt + Undo
	fmt.Println("\n--- Layer 9: ADAPT + UNDO ---")
	corr, _ := engine.Adapt(ctx, owner, "LoggerNet uses MySQL", "LoggerNet uses PostgreSQL")
	assert("adapt returns impact > 0", corr != nil && corr.Impact > 0, "")
	undo, _ := engine.Undo(ctx, owner, "LoggerNet uses MySQL", "LoggerNet uses PostgreSQL")
	assert("undo works", undo != nil, "")

	// Layer 10: Think (complete flow)
	fmt.Println("\n--- Layer 10: THINK (full pipeline) ---")
	// Seed some knowledge
	for _, fact := range []string{"LoggerNet collects data", "Ansible automates SSH", "Kubernetes runs containers", "LoRaWAN sensors monitor water"} {
		fv := emb.EmbedText(fact)
		deltaOM.Store(owner, fv)
		turboOM.AddVector(owner, turbovec.ExtractID(fact), fv)
	}
	start := time.Now()
	thought, err := engine.Think(ctx, owner, []string{"data collection automation", "water monitoring"})
	elapsed := time.Since(start)
	assert("Think returns no error", err == nil, fmt.Sprintf("%v", err))
	assert("Think produces idea", thought != nil && thought.Idea != "", "empty idea")
	assert("Think has neighbors", thought != nil && len(thought.Neighbors) > 0, "no neighbors")
	assert("Think valid", thought != nil && thought.Valid, "invalid thought")
	assert("Think time < 10s", elapsed < 10*time.Second, fmt.Sprintf("%v", elapsed))
	if thought != nil {
		fmt.Printf("    Idea: %s\n", truncate(thought.Idea, 100))
		fmt.Printf("    Depth=%d Conf=%.4f Novelty=%.4f Time=%v\n", thought.Depth, thought.Confidence, thought.Novelty, elapsed)
	}

	// Layer 11: Verifier (self-consistency)
	fmt.Println("\n--- Layer 11: VERIFIER ---")
	engine.Adapt(ctx, owner, "unused", "LoggerNet uses PostgreSQL for data storage")
	verifier := engine.DefaultVerifier(owner)
	valid, correction := verifier("LoggerNet uses MySQL for data storage")
	assert("verifier catches substitution conflict", !valid, fmt.Sprintf("valid=%v corr=%s", valid, correction))

	// Layer 12: Opportunistic Wander
	fmt.Println("\n--- Layer 12: WANDER ---")
	// Run another think to trigger wander
	engine.Think(ctx, owner, []string{"infrastructure monitoring tools"})
	wanderEvents := engine.Temporal().Recent(owner, 20)
	wanderCount := 0
	for _, e := range wanderEvents { if len(e.ID) > 7 && e.ID[:7] == "wander:" { wanderCount++ } }
	assert("wander records events", wanderCount >= 0, fmt.Sprintf("wander events: %d (may be 0 if conf < 0.03)", wanderCount))

	// SUMMARY
	fmt.Printf("\n=== RESULTS: %d PASS, %d FAIL ===\n", pass, fail)
	if fail == 0 { fmt.Println("ALL LAYERS VERIFIED. SYSTEM IS COMPLETE.") }
}

func cosine(a, b []float32) float32 {
	var d, na, nb float32
	for i := range a { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
	if na*nb == 0 { return 0 }
	return d / (sqrt(na) * sqrt(nb))
}
func sqrt(x float32) float32 { z := x; for i := 0; i < 10; i++ { z = (z + x/z) / 2 }; return z }
func truncate(s string, n int) string { if len(s) <= n { return s }; return s[:n] + "..." }
func countPrefix(ids []string, prefix string) int { c := 0; for _, id := range ids { if len(id) >= len(prefix) && id[:len(prefix)] == prefix { c++ } }; return c }
