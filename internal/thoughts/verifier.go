package thoughts

import "strings"

// DefaultVerifier returns a self-consistency verifier that checks thoughts
// against the system's own stored knowledge. No external tools needed.
func (e *Engine) DefaultVerifier(owner string) func(string) (bool, string) {
	return func(idea string) (bool, string) {
		if e.turbo == nil {
			return true, ""
		}

		vec := embed(idea)
		ids, scores, _ := e.turbo.SearchVector(owner, vec, 5)

		for i, id := range ids {
			if scores[i] < 0.7 {
				continue
			}
			// Check inversion/negation
			if isSemanticInversion(idea, id) || containsNegation(idea, id) {
				return false, id
			}
			// Check substitution contradiction: same subject+verb, different object
			if scores[i] > 0.8 && isSubstitutionConflict(idea, id) {
				return false, id
			}
		}
		return true, ""
	}
}

// isSubstitutionConflict detects "X uses A" vs "X uses B" type contradictions.
// If two statements share >60% of words but have different key objects, they conflict.
func isSubstitutionConflict(a, b string) bool {
	wordsA := strings.Fields(strings.ToLower(a))
	wordsB := strings.Fields(strings.ToLower(b))
	if len(wordsA) < 3 || len(wordsB) < 3 { return false }

	// Count shared words
	setB := make(map[string]bool)
	for _, w := range wordsB { setB[w] = true }
	shared := 0
	for _, w := range wordsA { if setB[w] { shared++ } }

	overlap := float64(shared) / float64(max(len(wordsA), len(wordsB)))
	if overlap < 0.5 { return false } // not about the same thing

	// High overlap but not identical = substitution (different value for same predicate)
	if overlap > 0.5 && overlap < 0.95 {
		return true
	}
	return false
}

func max(a, b int) int { if a > b { return a }; return b }
