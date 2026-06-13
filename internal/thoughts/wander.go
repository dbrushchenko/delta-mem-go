package thoughts

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Wanderer runs spontaneous thought generation in the background.
// Like the default mode network: probes δ-mem with internal noise,
// looking for salient patterns that cross threshold without external stimulus.
type Wanderer struct {
	engine    *Engine
	owner     string
	interval  time.Duration
	salience  float32 // threshold for a spontaneous thought to "matter"
	thoughts  []*Thought
	mu        sync.Mutex
	cancel    context.CancelFunc
	running   bool
}

func NewWanderer(engine *Engine, owner string) *Wanderer {
	return &Wanderer{
		engine:   engine,
		owner:    owner,
		interval: 5 * time.Second,
		salience: 0.3, // confidence must exceed this for a thought to surface
	}
}

// Start begins the background wandering loop.
func (w *Wanderer) Start() {
	w.mu.Lock()
	if w.running {
		w.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	w.cancel = cancel
	w.running = true
	w.mu.Unlock()

	go w.loop(ctx)
}

// Stop halts the wandering loop.
func (w *Wanderer) Stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cancel != nil {
		w.cancel()
	}
	w.running = false
}

// Harvest returns and clears accumulated spontaneous thoughts.
func (w *Wanderer) Harvest() []*Thought {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := w.thoughts
	w.thoughts = nil
	return out
}

func (w *Wanderer) loop(ctx context.Context) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(w.interval):
			w.wander(ctx, rng)
		}
	}
}

// wander generates a random perturbation of the δ-mem state and probes it.
// If the recall confidence crosses the salience threshold, something
// meaningful emerged spontaneously — store it.
func (w *Wanderer) wander(ctx context.Context, rng *rand.Rand) {
	if w.engine.delta == nil {
		return
	}

	// Generate noise vector — a random probe into the state space
	noise := make([]float32, 768)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64()) * 0.1
	}
	normalize(noise)

	// Probe δ-mem with noise — what does the accumulated state "think" about randomness?
	_, deltaO, confidence, err := w.engine.delta.Recall(w.owner, noise)
	if err != nil || confidence < w.salience {
		return // nothing salient surfaced
	}

	// Something resonated. Crystallize through IBNN.
	var crystallized []float32
	if w.engine.ibnn != nil {
		out, err := w.engine.ibnn.ForwardBatch(w.owner, [][]float32{deltaO})
		if err == nil && len(out) > 0 {
			crystallized = out[0]
		}
	}
	if crystallized == nil {
		crystallized = deltaO
	}

	// Find what this spontaneous activation is "about" via turbogo
	var neighbors []string
	if w.engine.turbo != nil {
		ids, _, _ := w.engine.turbo.SearchVector(w.owner, crystallized, 3)
		neighbors = ids
	}

	// Score novelty against what we already have stored
	nov := float32(1.0) - confidence // high confidence = familiar, low = novel

	thought := &Thought{
		Idea:       "", // no articulation yet — just the raw activation
		Seeds:      []string{"[spontaneous]"},
		Neighbors:  neighbors,
		Confidence: confidence,
		Novelty:    nov,
	}

	// If Gemma is available, articulate
	if w.engine.gemma != nil {
		prompt := "A spontaneous thought emerged from memory. Related concepts: " +
			joinStrings(neighbors) +
			"\nArticulate this as a single novel insight:"
		if idea, err := w.engine.gemma.Generate(ctx, prompt); err == nil {
			thought.Idea = idea
		}
	}

	// Store back into δ-mem — spontaneous thoughts feed future thinking
	w.engine.delta.Store(w.owner, crystallized)

	w.mu.Lock()
	w.thoughts = append(w.thoughts, thought)
	// Cap buffer
	if len(w.thoughts) > 100 {
		w.thoughts = w.thoughts[len(w.thoughts)-50:]
	}
	w.mu.Unlock()
}

func joinStrings(ss []string) string {
	if len(ss) == 0 {
		return "(none)"
	}
	out := ss[0]
	for _, s := range ss[1:] {
		out += ", " + s
	}
	return out
}

func init() { _ = math.Abs }
