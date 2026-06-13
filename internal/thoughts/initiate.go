package thoughts

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
)

// InitConfig controls the initiation process.
type InitConfig struct {
	Epochs       int     // number of passes over the data (default 3)
	LearningRate float32 // projection update step size (default 0.01)
	ChunkSize    int     // characters per chunk (default 200)
}

// InitResult reports what happened during initiation.
type InitResult struct {
	Chunks     int
	Epochs     int
	Duration   time.Duration
	FinalNorm  float32
	AvgConf    float32 // average recall confidence after training
}

// Initiate trains the δ-mem projections on a text corpus.
// This is the "initiation" — run once per owner with their domain data.
// After this, the substrate immediately works well for that domain.
//
// How it works:
// 1. Chunk the text into segments
// 2. Embed each chunk
// 3. For each epoch: store chunk, recall with next chunk, compute error,
//    update projections to maximize recall confidence on related pairs
func (e *Engine) Initiate(text string, owner string, cfg InitConfig) (*InitResult, error) {
	if cfg.Epochs == 0 { cfg.Epochs = 5 }
	if cfg.LearningRate == 0 { cfg.LearningRate = 0.01 }
	if cfg.ChunkSize == 0 { cfg.ChunkSize = 200 }

	chunks := chunkForTraining(text, cfg.ChunkSize)
	if len(chunks) < 2 {
		return nil, fmt.Errorf("need at least 2 chunks for initiation")
	}

	// Embed all chunks upfront
	embeddings := make([][]float32, len(chunks))
	for i, c := range chunks {
		embeddings[i] = embed(c)
	}

	// Get the δ-mem module for this owner
	mod, err := e.delta.Get(owner)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		// Shuffle order each epoch
		order := rng.Perm(len(chunks))

		for i := 0; i < len(order)-1; i++ {
			idx := order[i]
			nextIdx := order[i+1]

			// Store this chunk
			mod.Forward(embeddings[idx])

			// Recall with the next chunk (should be somewhat related since it's nearby in text)
			_, deltaO, _, _ := e.delta.Recall(owner, embeddings[nextIdx])

			// Compute update signal: the recall output should align with the target embedding.
			// Error = target - actual. Update projections toward reducing this error.
			target := embeddings[nextIdx]
			updateProjections(mod, embeddings[idx], target, deltaO, cfg.LearningRate)
		}
	}

	// Measure final quality
	var totalConf float32
	for i := 0; i < min(50, len(chunks)-1); i++ {
		_, _, conf, _ := e.delta.Recall(owner, embeddings[i+1])
		totalConf += conf
	}
	avgConf := totalConf / float32(min(50, len(chunks)-1))

	// Save state
	e.delta.Save(owner)

	// Also populate turbogo with all chunks for retrieval
	for i, c := range chunks {
		if e.turbo != nil {
			label := c
			if len(label) > 60 { label = label[:60] }
			e.turbo.AddVector(owner, label, embeddings[i])
		}
	}

	// Attach self-consistency verifier now that knowledge base is populated
	if e.Verifier == nil {
		e.Verifier = e.DefaultVerifier(owner)
	}

	return &InitResult{
		Chunks:   len(chunks),
		Epochs:   cfg.Epochs,
		Duration: time.Since(start),
		FinalNorm: mod.StateNorm(),
		AvgConf:  avgConf,
	}, nil
}

// updateProjections applies a small gradient step to Wq and Wk based on recall error.
// This is online Hebbian-like learning: strengthen connections that reduce recall error.
func updateProjections(mod *deltamem.Module, input, target, actual []float32, lr float32) {
	r := len(mod.S)
	dim := len(input)
	if dim == 0 || r == 0 { return }

	// Error signal
	err := make([]float32, dim)
	var errNorm float32
	for d := 0; d < dim && d < len(target) && d < len(actual); d++ {
		err[d] = target[d] - actual[d]
		errNorm += err[d] * err[d]
	}
	errNorm = float32(math.Sqrt(float64(errNorm)))
	if errNorm < 1e-8 { return }

	// Normalize error
	for d := range err { err[d] /= errNorm }

	// Update Wq: adjust query projection to better encode the input
	// Δ_Wq[i][j] += lr * err_projected[i] * input[j]
	for i := 0; i < r && i < len(mod.Wq); i++ {
		var errProj float32
		for d := 0; d < dim && d < len(mod.WqR[0]); d++ {
			if i < len(mod.WqR) {
				errProj += err[d] * mod.WqR[d][i] // backproject error into R-space
			}
		}
		for j := 0; j < dim && j < len(mod.Wq[i]); j++ {
			mod.Wq[i][j] += lr * errProj * input[j]
		}
	}

	// Update Wk: adjust key projection similarly
	for i := 0; i < r && i < len(mod.Wk); i++ {
		var errProj float32
		for d := 0; d < dim && d < len(mod.WoR); d++ {
			if i < len(mod.WoR[d]) {
				errProj += err[d] * mod.WoR[d][i]
			}
		}
		for j := 0; j < dim && j < len(mod.Wk[i]); j++ {
			mod.Wk[i][j] += lr * errProj * input[j]
		}
	}
}

func chunkForTraining(text string, targetLen int) []string {
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
		if current.Len() > 0 { current.WriteString(" ") }
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
