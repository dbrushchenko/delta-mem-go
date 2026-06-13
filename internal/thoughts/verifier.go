package thoughts

// DefaultVerifier returns a self-consistency verifier that checks thoughts
// against the system's own stored knowledge. No external tools needed.
//
// Logic: embed the idea, search turbogo for high-similarity matches,
// check if any match contradicts (via truth engine inversion detection).
// If contradiction found, return the stored fact as the correction.
func (e *Engine) DefaultVerifier(owner string) func(string) (bool, string) {
	return func(idea string) (bool, string) {
		if e.turbo == nil {
			return true, ""
		}

		vec := embed(idea)
		ids, scores, _ := e.turbo.SearchVector(owner, vec, 5)

		for i, id := range ids {
			if scores[i] < 0.7 {
				continue // not similar enough to matter
			}
			// High similarity — check if it's agreement or contradiction
			if isSemanticInversion(idea, id) || containsNegation(idea, id) {
				return false, id // stored knowledge contradicts this idea
			}
		}
		return true, ""
	}
}
