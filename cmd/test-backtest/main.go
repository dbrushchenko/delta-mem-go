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
	fmt.Println("╔══════════════════════════════════════════════╗")
	fmt.Println("║   mem-go COMPLETE BACKTEST — ALL LAYERS      ║")
	fmt.Println("╚══════════════════════════════════════════════╝")
	totalStart := time.Now()

	// Setup
	emb, err := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	if err != nil { fatal("embeddings", err) }
	defer emb.Close()
	thoughts.SetEmbedder(emb)

	dataDir := "./data/backtest"
	os.RemoveAll(dataDir)
	os.MkdirAll(dataDir, 0755)
	deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, dataDir)
	ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, dataDir)
	turboOM := turbovec.NewOwnerManager(768)
	engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)
	ctx := context.Background()
	owner := "backtest"

	// ═══════════════════════════════════════════════
	// 1. INITIATION
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 1. INITIATION ━━━")
	data, err := os.ReadFile(`C:\Users\dabrush\mem-go\training_3.txt`)
	if err != nil { fatal("read training", err) }
	initStart := time.Now()
	result, err := engine.Initiate(string(data), owner, thoughts.InitConfig{Epochs: 5, ChunkSize: 200})
	if err != nil { fatal("initiate", err) }
	initTime := time.Since(initStart)
	fmt.Printf("  ✓ Chunks: %d\n", result.Chunks)
	fmt.Printf("  ✓ Epochs: %d\n", result.Epochs)
	fmt.Printf("  ✓ Time: %v\n", initTime)
	fmt.Printf("  ✓ State Norm: %.6f\n", result.FinalNorm)
	fmt.Printf("  ✓ Avg Confidence: %.6f\n", result.AvgConf)
	pass("INITIATION", true)

	// ═══════════════════════════════════════════════
	// 2. SELF-MODEL
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 2. SELF-MODEL ━━━")
	loggernetVec := emb.EmbedText("LoggerNet")
	quantumVec := emb.EmbedText("quantum physics")
	loggernetConf := engine.Self().AmIConfident(loggernetVec)
	quantumConf := engine.Self().AmIConfident(quantumVec)
	fmt.Printf("  AmIConfident('LoggerNet'):      %v (%s)\n", loggernetConf, loggernetConf.String())
	fmt.Printf("  AmIConfident('quantum physics'): %v (%s)\n", quantumConf, quantumConf.String())
	selfPass := loggernetConf == thoughts.Confident && quantumConf == thoughts.NeverSeen
	if !selfPass {
		fmt.Printf("  ⚠ Expected LoggerNet=Confident(%d), quantum=NeverSeen(%d)\n", thoughts.Confident, thoughts.NeverSeen)
	}
	pass("SELF-MODEL", selfPass)

	// ═══════════════════════════════════════════════
	// 3. THINK x10
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 3. THINK x10 ━━━")
	seeds := [][]string{
		{"LoggerNet data collection", "AWS cloud infrastructure"},
		{"LoRaWAN IoT sensors", "real-time monitoring"},
		{"Kubernetes container orchestration", "service mesh"},
		{"certificate management", "SSL deployment"},
		{"Ansible automation", "infrastructure as code"},
		{"Loki log aggregation", "Grafana dashboards"},
		{"edge computing", "low-latency pipelines"},
		{"delta-rule memory", "neural interference patterns"},
		{"canary deployments", "Flagger progressive delivery"},
		{"water resource monitoring", "USGS sensor networks"},
	}
	var thinkTimes []time.Duration
	var thinkResults []*thoughts.Thought
	for i, s := range seeds {
		start := time.Now()
		thought, err := engine.Think(ctx, owner, s)
		elapsed := time.Since(start)
		thinkTimes = append(thinkTimes, elapsed)
		if err != nil {
			fmt.Printf("  [%02d] FAIL: %v\n", i+1, err)
			thinkResults = append(thinkResults, nil)
			continue
		}
		thinkResults = append(thinkResults, thought)
		idea := thought.Idea
		if len(idea) > 80 { idea = idea[:80] }
		fmt.Printf("  [%02d] %v  depth=%d  conf=%.4f  novelty=%.4f  valid=%v\n       %s\n",
			i+1, elapsed, thought.Depth, thought.Confidence, thought.Novelty, thought.Valid, idea)
	}
	var avgThink time.Duration
	for _, t := range thinkTimes { avgThink += t }
	avgThink /= time.Duration(len(thinkTimes))
	fmt.Printf("  Avg Think Time: %v\n", avgThink)
	pass("THINK x10", len(thinkResults) == 10 && thinkResults[0] != nil)

	// ═══════════════════════════════════════════════
	// 4. SELF-TRAINING PROOF
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 4. SELF-TRAINING PROOF ━━━")
	// Compare first think confidence vs last think confidence
	var firstConf, lastConf float32
	if thinkResults[0] != nil { firstConf = thinkResults[0].Confidence }
	if thinkResults[9] != nil { lastConf = thinkResults[9].Confidence }
	fmt.Printf("  First Think conf: %.6f\n", firstConf)
	fmt.Printf("  Last Think conf:  %.6f\n", lastConf)
	improved := lastConf > firstConf
	fmt.Printf("  Confidence increased: %v (delta=%.6f)\n", improved, lastConf-firstConf)
	if !improved {
		fmt.Println("  ⚠ Confidence did not increase — projections may not be training inline")
	}
	pass("SELF-TRAINING PROOF", improved)

	// ═══════════════════════════════════════════════
	// 5. TRUTH VALIDATION
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 5. TRUTH VALIDATION ━━━")
	engine.Truth().AddAxiom("the earth orbits the sun", "astronomy")
	engine.Truth().AddAxiom("water freezes at zero degrees celsius", "physics")
	engine.Truth().AddAxiom("LoggerNet runs on Windows EC2 instances", "infrastructure")

	truthTests := []struct{ stmt string; expectValid bool }{
		{"the sun orbits the earth", false},
		{"the earth orbits the sun providing seasons", true},
		{"water never freezes", false},
		{"LoggerNet runs on Windows instances in AWS", true},
	}
	truthAllPass := true
	for _, tt := range truthTests {
		v := engine.Truth().Validate(ctx, tt.stmt)
		status := "✓ VALID"
		if !v.Valid { status = "✗ REJECTED" }
		matched := v.Valid == tt.expectValid
		if !matched { truthAllPass = false }
		matchStr := ""
		if !matched { matchStr = " ← UNEXPECTED" }
		fmt.Printf("  %s  %q%s\n", status, tt.stmt, matchStr)
		if !v.Valid { fmt.Printf("       reason: %s\n", v.Reason) }
	}
	pass("TRUTH VALIDATION", truthAllPass)

	// ═══════════════════════════════════════════════
	// 6. ADAPT x5
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 6. ADAPT x5 ━━━")
	corrections := []struct{ wrong, right string }{
		{"LoggerNet uses MySQL database", "LoggerNet uses PostgreSQL database"},
		{"Kubernetes runs on bare metal only", "Kubernetes runs on VMs and bare metal"},
		{"LoRaWAN has a 1km range", "LoRaWAN has a 10-15km range in rural areas"},
		{"Grafana only supports Prometheus", "Grafana supports Prometheus, Loki, and many datasources"},
		{"SSL certificates never expire", "SSL certificates expire and must be renewed"},
	}
	var adaptTimes []time.Duration
	adaptAllPass := true
	for i, c := range corrections {
		start := time.Now()
		corr, err := engine.Adapt(ctx, owner, c.wrong, c.right)
		elapsed := time.Since(start)
		adaptTimes = append(adaptTimes, elapsed)
		if err != nil {
			fmt.Printf("  [%d] FAIL: %v\n", i+1, err)
			adaptAllPass = false
			continue
		}
		ok := corr.Impact > 0
		if !ok { adaptAllPass = false }
		fmt.Printf("  [%d] %v  impact=%.4f  %s\n", i+1, elapsed, corr.Impact, boolIcon(ok))
	}
	pass("ADAPT x5", adaptAllPass)

	// ═══════════════════════════════════════════════
	// 7. SELF-CONSISTENCY VERIFIER
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 7. SELF-CONSISTENCY VERIFIER ━━━")
	// After adapting 'LoggerNet uses MySQL' → 'LoggerNet uses PostgreSQL',
	// thinking with seeds that mention MySQL should trigger contradiction detection.
	thought7, err := engine.Think(ctx, owner, []string{"LoggerNet database MySQL"})
	var verifierCaught bool
	if err != nil {
		fmt.Printf("  Think error: %v\n", err)
	} else {
		fmt.Printf("  Think result: valid=%v  idea=%s\n", thought7.Valid, trunc(thought7.Idea, 80))
		// The verifier should have caught contradiction OR the truth engine should reject
		// Check if the verifier function catches it directly
		if engine.Verifier != nil {
			valid, correction := engine.Verifier("LoggerNet database uses MySQL")
			verifierCaught = !valid
			fmt.Printf("  Verifier('LoggerNet database uses MySQL'): valid=%v\n", valid)
			if correction != "" { fmt.Printf("  Correction: %s\n", trunc(correction, 80)) }
		}
	}
	if !verifierCaught {
		fmt.Println("  ⚠ Verifier did not catch MySQL contradiction (may need higher similarity)")
	}
	pass("SELF-CONSISTENCY VERIFIER", verifierCaught)

	// ═══════════════════════════════════════════════
	// 8. TEMPORAL
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 8. TEMPORAL ━━━")
	events := engine.Temporal().Recent(owner, 5)
	fmt.Printf("  Recent events (last 5): %d returned\n", len(events))
	for i, e := range events {
		content := e.Content
		if len(content) > 60 { content = content[:60] }
		fmt.Printf("    [%d] id=%s  when=%v  content=%s\n", i+1, trunc(e.ID, 30), e.When.Format("15:04:05.000"), content)
	}
	temporalPass := len(events) > 0
	pass("TEMPORAL", temporalPass)

	// ═══════════════════════════════════════════════
	// 9. DEDUP
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 9. DEDUP ━━━")
	countBefore := turboOM.Count(owner)
	// Run 10 more thinks with SAME seeds as test 3 — should dedup heavily
	for _, s := range seeds {
		engine.Think(ctx, owner, s)
	}
	countAfter := turboOM.Count(owner)
	growth := countAfter - countBefore
	fmt.Printf("  Index count BEFORE: %d\n", countBefore)
	fmt.Printf("  Index count AFTER:  %d\n", countAfter)
	fmt.Printf("  Growth: %d entries (from 10 repeated thinks)\n", growth)
	// With dedup threshold 0.92, repeated similar thoughts should supersede, not grow linearly
	dedupWorking := growth < 10*5 // each think adds ~5 entries (seeds+thought), dedup should reduce
	fmt.Printf("  Dedup effective: %v (growth < 50 expected)\n", dedupWorking)
	pass("DEDUP", dedupWorking)

	// ═══════════════════════════════════════════════
	// 10. PERFORMANCE SUMMARY
	// ═══════════════════════════════════════════════
	fmt.Println("\n━━━ 10. PERFORMANCE SUMMARY ━━━")
	totalTime := time.Since(totalStart)
	var avgAdapt time.Duration
	for _, t := range adaptTimes { avgAdapt += t }
	avgAdapt /= time.Duration(len(adaptTimes))
	fmt.Printf("  Total Time:      %v\n", totalTime)
	fmt.Printf("  Initiation Time: %v\n", initTime)
	fmt.Printf("  Avg Think Time:  %v\n", avgThink)
	fmt.Printf("  Avg Adapt Time:  %v\n", avgAdapt)
	fmt.Printf("  Final Index Size: %d entries\n", turboOM.Count(owner))
	fmt.Printf("  Self-Model Snapshot: %v\n", engine.Self().Snapshot())

	fmt.Println("\n╔══════════════════════════════════════════════╗")
	fmt.Println("║           BACKTEST COMPLETE                   ║")
	fmt.Printf("║  Total: %v                        ║\n", totalTime.Truncate(time.Millisecond))
	fmt.Println("╚══════════════════════════════════════════════╝")
}

func fatal(label string, err error) {
	fmt.Fprintf(os.Stderr, "FATAL %s: %v\n", label, err)
	os.Exit(1)
}

func pass(name string, ok bool) {
	if ok {
		fmt.Printf("  ══ %s: ✓ PASS ══\n", name)
	} else {
		fmt.Printf("  ══ %s: ✗ FAIL ══\n", name)
	}
}

func boolIcon(b bool) string { if b { return "✓" }; return "✗" }

func trunc(s string, n int) string { if len(s) <= n { return s }; return s[:n] + "..." }
