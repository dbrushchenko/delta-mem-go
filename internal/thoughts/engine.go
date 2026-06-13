package thoughts

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/gemma"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

// Thought is the output of a synthesis cycle.
type Thought struct {
	Idea       string
	Seeds      []string
	Neighbors  []string
	Confidence float32
	Novelty    float32
	Grounding  float32 // truth engine grounding score
	Depth      int     // how many re-entry iterations it took
	Valid      bool    // did it pass truth constraints?
}

// Engine orchestrates ideation with iterative re-entry and truth grounding.
type Engine struct {
	delta    *deltamem.OwnerManager
	ibnn     *ibnn.OwnerManager
	turbo    *turbovec.OwnerManager
	gemma    *gemma.Client
	truth    *TruthEngine
	wanderer map[string]*Wanderer // per-owner wanderers

	MaxDepth         int     // max re-entry iterations (default 5)
	SurpriseThreshold float32 // below this confidence = surprise → think deeper
	ConvergenceThreshold float32 // cosine similarity between iterations to detect fixed point
}

func New(delta *deltamem.OwnerManager, ibnn *ibnn.OwnerManager, turbo *turbovec.OwnerManager, gemma *gemma.Client) *Engine {
	return &Engine{
		delta:    delta,
		ibnn:     ibnn,
		turbo:    turbo,
		gemma:    gemma,
		truth:    NewTruthEngine(),
		wanderer: make(map[string]*Wanderer),
		MaxDepth:              5,
		SurpriseThreshold:     0.4,
		ConvergenceThreshold:  0.95,
	}
}

// Truth returns the truth engine for external axiom registration.
func (e *Engine) Truth() *TruthEngine { return e.truth }

// StartWander begins spontaneous thought for an owner.
func (e *Engine) StartWander(owner string) {
	if _, ok := e.wanderer[owner]; ok {
		return
	}
	w := NewWanderer(e, owner)
	e.wanderer[owner] = w
	w.Start()
}

// StopWander halts spontaneous thought for an owner.
func (e *Engine) StopWander(owner string) {
	if w, ok := e.wanderer[owner]; ok {
		w.Stop()
		delete(e.wanderer, owner)
	}
}

// HarvestWander returns spontaneous thoughts that emerged in the background.
func (e *Engine) HarvestWander(owner string) []*Thought {
	if w, ok := e.wanderer[owner]; ok {
		return w.Harvest()
	}
	return nil
}

// Think runs iterative synthesis with surprise-gated depth and truth validation.
//
// The loop:
//   iteration 0: seeds → δ-mem interference → crystallize → articulate
//   iteration N: previous thought becomes new seed → re-enter
//   stop when: converged (same thought) OR max depth OR truth violation
//
// Surprise gate: if δ-mem confidence < threshold, something unexpected was found.
// This triggers deeper processing (more retrieval, more iterations).
func (e *Engine) Think(ctx context.Context, owner string, seeds []string) (*Thought, error) {
	if e.gemma == nil {
		return nil, fmt.Errorf("thoughts engine requires gemma")
	}
	if len(seeds) == 0 {
		return nil, fmt.Errorf("need at least one seed")
	}

	var prevEmbedding []float32
	var lastThought *Thought
	currentSeeds := seeds

	for depth := 0; depth < e.MaxDepth; depth++ {
		// Core synthesis pass
		thought, err := e.singlePass(ctx, owner, currentSeeds, depth)
		if err != nil {
			return nil, err
		}

		// Truth validation — reject thoughts that violate axioms
		verdict := e.truth.Validate(ctx, thought.Idea)
		thought.Valid = verdict.Valid
		thought.Grounding = verdict.Grounding
		thought.Depth = depth + 1

		if !verdict.Valid {
			// Truth violation. Don't store. Don't propagate.
			// Instead, perturb and retry with the contradiction as context.
			if depth < e.MaxDepth-1 {
				currentSeeds = append(seeds, "Avoid: "+verdict.Reason)
				continue
			}
			// Final iteration still invalid — return with Valid=false
			return thought, nil
		}

		// Convergence check — did we reach a fixed point?
		ideaEmbed := embed(thought.Idea)
		if prevEmbedding != nil && cosine(ideaEmbed, prevEmbedding) > e.ConvergenceThreshold {
			// Converged. This thought is stable. Prove it.
			e.truth.Prove(thought.Idea, thought.Confidence, "convergence")
			return thought, nil
		}

		// Surprise gate — low confidence means unexpected pattern.
		// Think DEEPER when surprised.
		if thought.Confidence >= e.SurpriseThreshold && depth > 0 {
			// High confidence, not first pass = familiar territory. Done.
			return thought, nil
		}

		// Re-entry: the thought becomes the seed for the next cycle
		prevEmbedding = ideaEmbed
		lastThought = thought
		currentSeeds = []string{thought.Idea}
	}

	return lastThought, nil
}

// singlePass is one iteration of the synthesis loop.
func (e *Engine) singlePass(ctx context.Context, owner string, seeds []string, depth int) (*Thought, error) {
	// Step 1: Embed seeds and store into δ-mem, recall interference pattern
	var combinedRaw []float32 // raw seed embedding (for retrieval)
	var combinedDelta []float32
	var totalConf float32
	for _, seed := range seeds {
		hidden := embed(seed)
		combinedRaw = vecAdd(combinedRaw, hidden)
		e.delta.Store(owner, hidden)
		_, deltaO, conf, err := e.delta.Recall(owner, hidden)
		if err != nil {
			return nil, err
		}
		combinedDelta = vecAdd(combinedDelta, deltaO)
		totalConf += conf
	}
	avgConf := totalConf / float32(len(seeds))
	normalize(combinedRaw)

	// Step 2: Retrieval — search turbogo with RAW embeddings (not crystallized)
	k := 3
	if avgConf < e.SurpriseThreshold {
		k = 8
	}
	var neighbors []string
	if e.turbo != nil {
		ids, _, _ := e.turbo.SearchVector(owner, combinedRaw, k)
		neighbors = ids
	}

	// Step 3: IBNN crystallization — normalize δ-mem output first, then sharpen
	// IBNN shapes what goes into the generation prompt (dominant patterns), NOT retrieval
	normalize(combinedDelta) // scale to unit norm so IBNN has real signal
	var crystallized []float32
	if e.ibnn != nil {
		out, err := e.ibnn.ForwardBatch(owner, [][]float32{combinedDelta})
		if err == nil && len(out) > 0 {
			crystallized = out[0]
		}
	}
	if crystallized == nil {
		crystallized = combinedDelta
	}

	// Step 4: Articulation — IBNN activations inform the prompt
	prompt := buildThoughtPrompt(seeds, neighbors, topActivations(crystallized, 5))
	if depth > 0 {
		prompt += fmt.Sprintf("\n(Iteration %d — go deeper, refine, challenge your previous thought)", depth+1)
	}
	idea, err := e.gemma.Generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// Step 5: Novelty
	novelty := computeNovelty(idea, seeds)

	// Store back
	thoughtHidden := embed(idea)
	e.delta.Store(owner, thoughtHidden)
	if e.turbo != nil {
		e.turbo.AddVector(owner, fmt.Sprintf("thought_%d", hashStr(idea)), thoughtHidden)
	}

	return &Thought{
		Idea:       strings.TrimSpace(idea),
		Seeds:      seeds,
		Neighbors:  neighbors,
		Confidence: avgConf,
		Novelty:    novelty,
	}, nil
}

func buildThoughtPrompt(seeds, neighbors []string, activations []int) string {
	var b strings.Builder
	b.WriteString("You are a creative thinking engine. Given these seed concepts and related memories, synthesize a novel insight or theory that emerges from their intersection. Do not simply summarize — produce something new.\n\n")
	b.WriteString("Seeds:\n")
	for _, s := range seeds {
		b.WriteString("- ")
		b.WriteString(s)
		b.WriteString("\n")
	}
	if len(neighbors) > 0 {
		b.WriteString("\nRelated memories:\n")
		for _, n := range neighbors {
			b.WriteString("- ")
			b.WriteString(n)
			b.WriteString("\n")
		}
	}
	b.WriteString(fmt.Sprintf("\nNeural activation pattern (strongest dimensions): %v\n", activations))
	b.WriteString("\nNovel thought:")
	return b.String()
}

func computeNovelty(idea string, seeds []string) float32 {
	ideaVec := embed(idea)
	maxSim := float32(0)
	for _, s := range seeds {
		if sim := cosine(ideaVec, embed(s)); sim > maxSim {
			maxSim = sim
		}
	}
	return 1.0 - maxSim
}

func topActivations(vec []float32, n int) []int {
	type iv struct{ idx int; val float32 }
	top := make([]iv, 0, n)
	for i, v := range vec {
		abs := v; if abs < 0 { abs = -abs }
		if len(top) < n {
			top = append(top, iv{i, abs})
		} else {
			minIdx := 0
			for j := 1; j < len(top); j++ { if top[j].val < top[minIdx].val { minIdx = j } }
			if abs > top[minIdx].val { top[minIdx] = iv{i, abs} }
		}
	}
	result := make([]int, len(top))
	for i, t := range top { result[i] = t.idx }
	return result
}

func textToHidden(text string) []float32 {
	hidden := make([]float32, 768)
	seed := uint64(0)
	for _, c := range text { seed = seed*31 + uint64(c) }
	for i := range hidden { seed = seed*6364136223846793005 + 1; hidden[i] = float32(math.Sin(float64(seed)/1e15)) * 0.1 }
	normalize(hidden)
	return hidden
}

func normalize(v []float32) {
	var norm float32
	for _, x := range v { norm += x * x }
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 { for i := range v { v[i] /= norm } }
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 { return 0 }
	var dot, na, nb float32
	for i := range a { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
	denom := float32(math.Sqrt(float64(na * nb)))
	if denom == 0 { return 0 }
	return dot / denom
}

func vecAdd(a, b []float32) []float32 {
	if a == nil { out := make([]float32, len(b)); copy(out, b); return out }
	for i := range a { a[i] += b[i] }
	return a
}

func hashStr(s string) uint64 {
	h := uint64(14695981039346656037)
	for _, c := range s { h ^= uint64(c); h *= 1099511628211 }
	return h
}
