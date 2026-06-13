package thoughts

import (
	"context"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

// Correction is a single learning event: what was wrong, what's true.
// Designed to be as effortless as possible — one call, instant adaptation.
type Correction struct {
	Wrong    string
	Right    string
	When     time.Time
	Impact   float32 // how much this shifted the state
}

// Adapt is the effortless learning interface.
// No ceremony. No configuration. Just: "that was wrong, this is right."
//
// Philosophy: NEVER create voids. Always REPLACE.
// The wrong pattern doesn't get erased — it gets REDIRECTED toward the right one.
// This preserves the pathway (how you got there) while changing the destination.
func (e *Engine) Adapt(ctx context.Context, owner, wrong, right string) (*Correction, error) {
	wrongVec := embed(wrong)
	rightVec := embed(right)

	// Compute the correction vector: the direction FROM wrong TO right.
	// This is the "rewiring" — same pathway, new destination.
	correctionVec := make([]float32, len(wrongVec))
	for i := range correctionVec {
		correctionVec[i] = rightVec[i] - wrongVec[i]*0.5 // blend: mostly right, partially informed by wrong
	}
	normalize(correctionVec)

	// Store the correction — this overwrites the wrong region with the right direction
	_, err := e.delta.Store(owner, correctionVec)
	if err != nil {
		return nil, err
	}

	// Also store the right pattern at full strength
	e.delta.Store(owner, rightVec)

	// Register as proven truth
	e.truth.Prove(right, 0.9, "correction")

	// In turbogo: replace, don't just remove
	if e.turbo != nil {
		e.turbo.RemoveVector(owner, turbovec.ExtractID(wrong))
		e.turbo.AddVector(owner, turbovec.ExtractID(right), rightVec)
	}

	impact := 1.0 - cosine(wrongVec, rightVec)

	return &Correction{
		Wrong:  wrong,
		Right:  right,
		When:   time.Now(),
		Impact: impact,
	}, nil
}

// Learn is even simpler than Adapt — just absorb something new.
// No wrong/right duality. Just: "here's a fact."
func (e *Engine) Learn(ctx context.Context, owner, fact string) error {
	vec := embed(fact)
	e.delta.Store(owner, vec)
	e.truth.Prove(fact, 0.7, "learned")
	if e.turbo != nil {
		e.turbo.AddVector(owner, turbovec.ExtractID(fact), vec)
	}
	return nil
}

// Forget is intentionally NOT a standalone operation.
// Removing without replacing creates voids that fill with noise.
// Use Adapt(wrong, right) instead — always replace, never just remove.
//
// For the rare case where something must go with no clear replacement,
// Forget fills the void with a neutral dampener rather than leaving a hole.
// The region of state space doesn't become empty — it becomes inert.
func (e *Engine) Forget(ctx context.Context, owner, what string) error {
	vec := embed(what)
	// Don't negate (that creates a void). Instead, overwrite with a
	// dampened neutral vector — same region, zero energy.
	// This is "I acknowledge this existed but it no longer drives anything."
	neutral := make([]float32, len(vec))
	for i := range vec {
		neutral[i] = vec[i] * 0.01 // near-zero but same direction — inert, not void
	}
	e.delta.Store(owner, neutral)
	if e.turbo != nil {
		// Replace in index with the inert version, don't delete
		e.turbo.RemoveVector(owner, turbovec.ExtractID(what))
	}
	return nil
}
