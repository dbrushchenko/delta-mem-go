package session

import "time"

// Session holds per-session state: what was injected, budget, turn tracking.
type Session struct {
	ID         string
	Owner      string
	Type       SessionType
	Policy     Policy
	Injected   map[string]*Injection
	Created    time.Time
	LastActive time.Time
	TurnCount  int
	TokensUsed int
}

// Injection records when and how a memory was surfaced in this session.
type Injection struct {
	Turn      int
	Timestamp time.Time
	Score     float32
	Useful    *bool // nil = unknown, true = reinforced, false = deprioritized
}

// ShouldInject checks if a key should be injected given session policy.
func (s *Session) ShouldInject(key string, score float32) bool {
	// Budget check
	if s.TokensUsed >= s.Policy.TokenBudget {
		return false
	}
	// Max per turn
	turnInjections := 0
	for _, inj := range s.Injected {
		if inj.Turn == s.TurnCount {
			turnInjections++
		}
	}
	if turnInjections >= s.Policy.MaxPerTurn {
		return false
	}
	// Score threshold
	if score < s.Policy.MinScore {
		return false
	}
	// Dedup: already injected and not stale
	if inj, exists := s.Injected[key]; exists {
		turnsAgo := s.TurnCount - inj.Turn
		if turnsAgo < s.Policy.StaleAfterTurns {
			return false // too recent, skip
		}
		// Stale — allow re-injection
	}
	return true
}

// RecordInjection marks a key as injected this turn.
func (s *Session) RecordInjection(key string, score float32, tokenCost int) {
	s.Injected[key] = &Injection{
		Turn:      s.TurnCount,
		Timestamp: time.Now(),
		Score:     score,
	}
	s.TokensUsed += tokenCost
}

// MarkUseful provides feedback on an injected memory.
func (s *Session) MarkUseful(key string, useful bool) {
	if inj, ok := s.Injected[key]; ok {
		inj.Useful = &useful
	}
}

// Summarize produces a summary for persistence on close.
func (s *Session) Summarize() *SessionSummary {
	useful, notUseful, unknown := 0, 0, 0
	for _, inj := range s.Injected {
		switch {
		case inj.Useful == nil: unknown++
		case *inj.Useful: useful++
		default: notUseful++
		}
	}
	return &SessionSummary{
		ID:           s.ID,
		Owner:        s.Owner,
		Type:         s.Type.String(),
		Duration:     time.Since(s.Created),
		Turns:        s.TurnCount,
		Injected:     len(s.Injected),
		TokensUsed:   s.TokensUsed,
		UsefulCount:  useful,
		NotUseful:    notUseful,
		UnknownCount: unknown,
	}
}

// SessionSummary is the final record when a session closes.
type SessionSummary struct {
	ID           string
	Owner        string
	Type         string
	Duration     time.Duration
	Turns        int
	Injected     int
	TokensUsed   int
	UsefulCount  int
	NotUseful    int
	UnknownCount int
}
