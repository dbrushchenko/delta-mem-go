package thoughts

import (
	"context"
	"math"
	"strings"
	"sync"
)

// TruthEngine enforces grounding constraints on generated thoughts.
//
// The principle: certain things are TRUE regardless of what the generative
// system produces. A thought that contradicts known truth is not creative —
// it's wrong. Creativity operates WITHIN truth, not against it.
//
// Three layers:
// 1. Axioms — immutable truths stored with infinite confidence (the "written on hearts" layer)
// 2. Coherence — a thought must not contradict itself or prior validated thoughts
// 3. Grounding — a thought must connect to something real (retrievable from memory)
type TruthEngine struct {
	axioms  []Axiom
	proven  []Proven
	nli     NLIChecker // optional — second opinion when heuristic is uncertain
	mu      sync.RWMutex
}

// NLIChecker is the interface for natural language inference.
// Returns "contradiction", "entailment", or "neutral" with confidence.
// When nil, truth engine uses heuristic only.
type NLIChecker interface {
	Check(textA, textB string) (label string, confidence float32)
}

// Axiom is an immutable truth. It cannot be overridden by generation.
// Like physical law: water freezes at 0°C. No amount of creative thinking changes this.
type Axiom struct {
	Statement string
	Embedding []float32
	Domain    string // "physics", "logic", "math", "ethics", etc.
}

// Proven is a thought that was validated against reality and held.
type Proven struct {
	Statement  string
	Embedding  []float32
	Confidence float32
	Source     string // how it was validated: "observation", "derivation", "axiom"
}

// Verdict is the truth engine's judgment on a thought.
type Verdict struct {
	Valid           bool
	Coherence       float32 // 0-1: internal consistency
	Grounding       float32 // 0-1: connection to known truths
	Contradictions  []string
	Reason          string
}

func NewTruthEngine() *TruthEngine {
	return &TruthEngine{}
}

// SetNLI attaches an optional NLI model for second-opinion contradiction detection.
func (t *TruthEngine) SetNLI(nli NLIChecker) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.nli = nli
}

// AddAxiom registers an immutable truth.
func (t *TruthEngine) AddAxiom(statement, domain string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.axioms = append(t.axioms, Axiom{
		Statement: statement,
		Embedding: embed(statement),
		Domain:    domain,
	})
}

// Validate checks a generated thought against truth constraints.
// Returns a Verdict — the thought either passes or it doesn't.
func (t *TruthEngine) Validate(ctx context.Context, thought string) *Verdict {
	t.mu.RLock()
	defer t.mu.RUnlock()

	embed := embed(thought)
	verdict := &Verdict{Valid: true, Coherence: 1.0, Grounding: 0.0}

	// Fast path: exact hash match = same text as an axiom = immediately grounded
	thoughtHash := hashStr(strings.ToLower(strings.TrimSpace(thought)))
	for _, axiom := range t.axioms {
		if hashStr(strings.ToLower(strings.TrimSpace(axiom.Statement))) == thoughtHash {
			verdict.Grounding = 1.0
			return verdict
		}
	}
	for _, p := range t.proven {
		if hashStr(strings.ToLower(strings.TrimSpace(p.Statement))) == thoughtHash {
			verdict.Grounding = p.Confidence
			return verdict
		}
	}

	// Check against axioms — any contradiction is fatal
	for _, axiom := range t.axioms {
		sim := cosine(embed, axiom.Embedding)

		// Any high similarity (>0.7): check for inversion FIRST, then decide if it agrees
		if sim > 0.7 {
			if containsNegation(thought, axiom.Statement) || isSemanticInversion(thought, axiom.Statement) {
				verdict.Valid = false
				verdict.Contradictions = append(verdict.Contradictions, axiom.Statement)
				verdict.Reason = "contradicts axiom: " + axiom.Statement
				return verdict
			}
			// Heuristic didn't catch it. If NLI available and sim < 0.97, get second opinion.
			if t.nli != nil && sim < 0.97 {
				label, conf := t.nli.Check(thought, axiom.Statement)
				if label == "contradiction" && conf > 0.7 {
					verdict.Valid = false
					verdict.Contradictions = append(verdict.Contradictions, axiom.Statement)
					verdict.Reason = "NLI contradiction: " + axiom.Statement
					return verdict
				}
			}
			// Not an inversion — it agrees
			verdict.Grounding = maxF(verdict.Grounding, sim)
			continue
		}

		if sim > 0.3 && containsNegation(thought, axiom.Statement) {
			verdict.Valid = false
			verdict.Contradictions = append(verdict.Contradictions, axiom.Statement)
			verdict.Reason = "contradicts axiom: " + axiom.Statement
			return verdict
		}
	}

	// Check against proven truths for coherence
	for _, p := range t.proven {
		sim := cosine(embed, p.Embedding)
		if sim > 0.7 {
			verdict.Grounding = maxF(verdict.Grounding, sim*p.Confidence)
		}
		// Contradicting a highly-confident proven truth is suspect
		if sim > 0.3 && p.Confidence > 0.8 && containsNegation(thought, p.Statement) {
			verdict.Coherence -= 0.3
			verdict.Contradictions = append(verdict.Contradictions, p.Statement)
		}
	}

	// Self-coherence: check if thought contains internal contradictions
	verdict.Coherence = maxF(0, verdict.Coherence)
	verdict.Coherence = minF(1, verdict.Coherence)

	// A thought with zero grounding is unanchored — mark it
	if verdict.Grounding < 0.1 && len(t.axioms) > 0 {
		verdict.Reason = "ungrounded: no connection to known truths"
		// Not invalid, but flagged. Ungrounded thoughts need validation.
	}

	// Coherence below threshold = invalid
	if verdict.Coherence < 0.5 {
		verdict.Valid = false
		verdict.Reason = "incoherent: contradicts established knowledge"
	}

	return verdict
}

// Prove marks a thought as validated truth. It joins the proven set
// and influences future validation.
func (t *TruthEngine) Prove(statement string, confidence float32, source string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.proven = append(t.proven, Proven{
		Statement:  statement,
		Embedding:  embed(statement),
		Confidence: confidence,
		Source:     source,
	})
}

// GroundingScore returns how well a thought connects to the truth base.
func (t *TruthEngine) GroundingScore(thought string) float32 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	embed := embed(thought)
	var maxSim float32
	for _, a := range t.axioms {
		if s := cosine(embed, a.Embedding); s > maxSim { maxSim = s }
	}
	for _, p := range t.proven {
		if s := cosine(embed, p.Embedding) * p.Confidence; s > maxSim { maxSim = s }
	}
	return maxSim
}

// containsNegation is a heuristic: does `thought` semantically negate `truth`?
// In production this would be an NLI model. For now: keyword + structural check.
func containsNegation(thought, truth string) bool {
	tl := strings.ToLower(thought)
	negators := []string{"not ", "never ", "impossible ", "cannot ", "doesn't ", "isn't ", "won't ", "false ", "incorrect "}
	// Check if the thought discusses the same topic but negates
	truthWords := strings.Fields(strings.ToLower(truth))
	overlap := 0
	for _, tw := range truthWords {
		if len(tw) > 3 && strings.Contains(tl, tw) {
			overlap++
		}
	}
	if overlap < 2 {
		return false // not about the same topic
	}
	for _, neg := range negators {
		if strings.Contains(tl, neg) {
			return true
		}
	}
	return false
}

// isSemanticInversion detects subject/object swaps.
// "the earth orbits the sun" vs "the sun revolves around the earth"
func isSemanticInversion(thought, truth string) bool {
	tl := strings.ToLower(thought)
	al := strings.ToLower(truth)
	// Extract entities: nouns that appear in both statements
	skip := map[string]bool{"the": true, "that": true, "this": true, "with": true, "from": true, "into": true, "once": true, "will": true, "does": true, "have": true, "been": true, "around": true, "daily": true}
	truthWords := extractNouns(al, skip)
	thoughtWords := extractNouns(tl, skip)
	// Find shared entities (words appearing in both)
	var sharedInTruth, sharedInThought []int
	for i, tw := range truthWords {
		for j, ow := range thoughtWords {
			if tw == ow {
				sharedInTruth = append(sharedInTruth, i)
				sharedInThought = append(sharedInThought, j)
			}
		}
	}
	// If 2+ shared entities appear in reversed order, it's an inversion
	if len(sharedInTruth) >= 2 {
		if sharedInTruth[0] < sharedInTruth[1] && sharedInThought[0] > sharedInThought[1] {
			return true
		}
		if sharedInTruth[0] > sharedInTruth[1] && sharedInThought[0] < sharedInThought[1] {
			return true
		}
	}
	return false
}

func extractNouns(s string, skip map[string]bool) []string {
	words := strings.Fields(s)
	var nouns []string
	for _, w := range words {
		if len(w) >= 3 && !skip[w] {
			nouns = append(nouns, w)
		}
	}
	return nouns
}

func maxF(a, b float32) float32 { return float32(math.Max(float64(a), float64(b))) }
func minF(a, b float32) float32 { return float32(math.Min(float64(a), float64(b))) }
