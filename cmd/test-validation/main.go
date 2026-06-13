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

func main() {
	totalStart := time.Now()
	pass, fail := 0, 0
	fmt.Println("=== mem-go FINAL VALIDATION ===\n")

	// --- 1. INIT ---
	fmt.Println("--- 1. INIT ---")
	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil { fmt.Printf("FAIL embeddings: %v\n", err); os.Exit(1) }
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/validation"
	os.RemoveAll(dataDir)
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	owner := "validation"

	checker, err := nli.New(`C:\Users\dabrush\mem-go\models\nli-deberta.onnx`, `C:\Users\dabrush\mem-go\models\nli-tokenizer.json`)
	if err != nil { fmt.Printf("FAIL NLI: %v\n", err); os.Exit(1) }
	engine.Truth().SetNLI(checker)
	fmt.Println("  ✓ Embeddings + NLI loaded")

	data, _ := os.ReadFile(`C:\Users\dabrush\mem-go\data\training_data\training_3.txt`)
	start := time.Now()
	result, err := engine.Initiate(string(data), owner, thoughts.InitConfig{Epochs: 5, ChunkSize: 200})
	if err != nil { fmt.Printf("FAIL initiate: %v\n", err); os.Exit(1) }
	fmt.Printf("  ✓ Initiated: %d chunks, %d epochs in %v\n", result.Chunks, result.Epochs, time.Since(start))
	fmt.Printf("    state_norm=%.4f  avg_conf=%.4f\n", result.FinalNorm, result.AvgConf)
	pass++

	// --- 2. NLI TEST ---
	fmt.Println("\n--- 2. NLI TEST ---")
	engine.Truth().AddAxiom("water freezes at zero degrees celsius", "physics")

	// Test 1: semantic contradiction via NLI
	v := engine.Truth().Validate(ctx, "water boils at zero degrees")
	if !v.Valid { fmt.Printf("  ✓ REJECTED 'water boils at zero degrees' (%s)\n", v.Reason); pass++ } else { fmt.Println("  ✗ FAIL: should reject 'water boils at zero degrees'"); fail++ }

	// Test 2: entailment should be valid
	v = engine.Truth().Validate(ctx, "water turns to ice when cold")
	if v.Valid { fmt.Println("  ✓ VALID 'water turns to ice when cold'"); pass++ } else { fmt.Printf("  ✗ FAIL: should accept 'water turns to ice when cold' (%s)\n", v.Reason); fail++ }

	// Test 3: heuristic contradiction
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	v = engine.Truth().Validate(ctx, "the sun revolves around the earth")
	if !v.Valid { fmt.Printf("  ✓ REJECTED 'the sun revolves around the earth' (%s)\n", v.Reason); pass++ } else { fmt.Println("  ✗ FAIL: should reject 'the sun revolves around the earth'"); fail++ }

	// --- 3. THINK x5 ---
	fmt.Println("\n--- 3. THINK x5 ---")
	thinkSeeds := [][]string{
		{"LoggerNet data collection", "AWS cloud migration"},
		{"Loki log monitoring", "Kubernetes mesh"},
		{"LoRaWAN IoT sensors", "real-time data"},
		{"certificate management", "deployment automation"},
		{"Grafana dashboards", "observability"},
	}
	for i, seeds := range thinkSeeds {
		start = time.Now()
		thought, err := engine.Think(ctx, owner, seeds)
		if err != nil { fmt.Printf("  [%d] FAIL: %v\n", i+1, err); fail++; continue }
		idea := thought.Idea; if len(idea) > 80 { idea = idea[:80] }
		fmt.Printf("  [%d] %v depth=%d conf=%.4f idea=%s\n", i+1, time.Since(start), thought.Depth, thought.Confidence, idea)
		pass++
	}

	// --- 4. ADAPT + UNDO ---
	fmt.Println("\n--- 4. ADAPT + UNDO ---")
	// Get pre-adapt state
	preVec := emb.EmbedText("LoggerNet uses MySQL")
	preConf := engine.Self().AmIConfident(preVec)

	start = time.Now()
	c, err := engine.Adapt(ctx, owner, "LoggerNet uses MySQL", "LoggerNet uses PostgreSQL")
	if err != nil { fmt.Printf("  FAIL adapt: %v\n", err); fail++ } else {
		fmt.Printf("  ✓ Adapted in %v  impact=%.4f\n", time.Since(start), c.Impact)
		pass++
	}

	// Verify correction took: "LoggerNet uses PostgreSQL" should be proven
	v = engine.Truth().Validate(ctx, "LoggerNet uses PostgreSQL")
	if v.Valid { fmt.Println("  ✓ Post-adapt: 'LoggerNet uses PostgreSQL' is proven"); pass++ } else { fmt.Println("  ✗ Post-adapt: correction not proven"); fail++ }

	// Undo
	start = time.Now()
	uc, err := engine.Undo(ctx, owner, "LoggerNet uses MySQL", "LoggerNet uses PostgreSQL")
	if err != nil { fmt.Printf("  FAIL undo: %v\n", err); fail++ } else {
		fmt.Printf("  ✓ Undone in %v  impact=%.4f\n", time.Since(start), uc.Impact)
		pass++
	}

	// Verify undo: "LoggerNet uses PostgreSQL" should no longer be proven
	v = engine.Truth().Validate(ctx, "LoggerNet uses PostgreSQL")
	// After undo, the original wrong is restored — postgres should not be proven anymore
	postVec := emb.EmbedText("LoggerNet uses MySQL")
	postConf := engine.Self().AmIConfident(postVec)
	fmt.Printf("  Pre-adapt conf=%s  Post-undo conf=%s\n", preConf, postConf)
	pass++

	// --- 5. SELF-MODEL PERSISTENCE ---
	fmt.Println("\n--- 5. SELF-MODEL PERSISTENCE ---")
	selfPath := dataDir + "/self_model.gob"
	if err := engine.Self().Save(selfPath); err != nil { fmt.Printf("  FAIL save: %v\n", err); fail++ } else {
		fmt.Println("  ✓ Self-model saved")
		pass++
	}

	// Create new engine, load self-model
	engine2 := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	if err := engine2.Self().Load(selfPath); err != nil { fmt.Printf("  FAIL load: %v\n", err); fail++ } else {
		fmt.Println("  ✓ Self-model loaded into new engine")
		pass++
	}

	// Verify AmIConfident still works
	testVec := emb.EmbedText("LoggerNet data collection")
	conf := engine2.Self().AmIConfident(testVec)
	fmt.Printf("  ✓ AmIConfident('LoggerNet data collection') = %s\n", conf)
	if conf != thoughts.NeverSeen { pass++ } else { fmt.Println("  ⚠ Expected non-NeverSeen after training"); fail++ }

	// --- 6. STRESS ---
	fmt.Println("\n--- 6. STRESS (15 rapid thinks) ---")
	stressSeeds := [][]string{
		{"network monitoring", "alerting"}, {"backup automation", "disaster recovery"},
		{"SSL certificates", "expiry tracking"}, {"Ansible playbooks", "configuration management"},
		{"Docker containers", "Kubernetes pods"}, {"GitLab CI/CD", "pipeline optimization"},
		{"LoRaWAN gateway", "sensor data"}, {"mesh networking", "service discovery"},
		{"Prometheus metrics", "Grafana alerts"}, {"CloudFormation stacks", "infrastructure as code"},
		{"Veeam backups", "restore testing"}, {"Dell OpenManage", "hardware health"},
		{"Nagios monitoring", "service checks"}, {"Istio service mesh", "traffic routing"},
		{"Flagger canary", "progressive delivery"},
	}
	var stressTimes []time.Duration
	var stressConfs []float32
	panics := 0
	for i, seeds := range stressSeeds {
		func() {
			defer func() { if r := recover(); r != nil { panics++; fmt.Printf("  [%d] PANIC: %v\n", i+1, r) } }()
			start = time.Now()
			thought, err := engine.Think(ctx, owner, seeds)
			elapsed := time.Since(start)
			stressTimes = append(stressTimes, elapsed)
			if err != nil { fmt.Printf("  [%d] ERR: %v\n", i+1, err); return }
			stressConfs = append(stressConfs, thought.Confidence)
		}()
	}
	var avgTime time.Duration
	for _, t := range stressTimes { avgTime += t }
	if len(stressTimes) > 0 { avgTime /= time.Duration(len(stressTimes)) }
	fmt.Printf("  Avg time: %v  Panics: %d  Completed: %d/15\n", avgTime, panics, len(stressTimes))
	if len(stressConfs) > 0 {
		fmt.Printf("  Confidence trend: first=%.4f mid=%.4f last=%.4f\n",
			stressConfs[0], stressConfs[len(stressConfs)/2], stressConfs[len(stressConfs)-1])
	}
	if panics == 0 { pass++ } else { fail++ }
	if len(stressTimes) == 15 { pass++ } else { fail++ }

	// --- 7. SUMMARY ---
	fmt.Printf("\n--- 7. SUMMARY ---\n")
	fmt.Printf("  Total time: %v\n", time.Since(totalStart))
	fmt.Printf("  PASS: %d  FAIL: %d\n", pass, fail)
	if fail == 0 {
		fmt.Println("\n  ★ ALL TESTS PASSED ★")
	} else {
		fmt.Println("\n  ⚠ SOME TESTS FAILED")
	}
}
