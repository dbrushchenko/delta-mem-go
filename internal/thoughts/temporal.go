package thoughts

import (
	"sync"
	"time"
)

// Temporal tracks when memories were stored and provides recency-weighted retrieval.
type Temporal struct {
	events map[string][]Event // owner → events (ordered by time)
	mu     sync.RWMutex
}

// Event is a timestamped memory event.
type Event struct {
	ID      string
	Content string
	When    time.Time
}

// SelfModel tracks what the system knows about its own state.
type SelfModel struct {
	TotalStores   int64
	TotalRecalls  int64
	TotalThinks   int64
	TotalAdapts   int64
	LastActivity  time.Time
	TopConcepts   []string // most frequently recalled
	Uncertainty   float32  // avg confidence (lower = more uncertain)
	mu            sync.Mutex
}

func NewTemporal() *Temporal {
	return &Temporal{events: make(map[string][]Event)}
}

func NewSelfModel() *SelfModel {
	return &SelfModel{}
}

// Record logs a timestamped event.
func (t *Temporal) Record(owner, id, content string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events[owner] = append(t.events[owner], Event{ID: id, Content: content, When: time.Now()})
	// Keep bounded
	if len(t.events[owner]) > 1000 {
		t.events[owner] = t.events[owner][len(t.events[owner])-500:]
	}
}

// Recent returns the N most recent events for an owner.
func (t *Temporal) Recent(owner string, n int) []Event {
	t.mu.RLock()
	defer t.mu.RUnlock()
	events := t.events[owner]
	if len(events) <= n {
		return events
	}
	return events[len(events)-n:]
}

// Since returns events after a given time.
func (t *Temporal) Since(owner string, after time.Time) []Event {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var result []Event
	for _, e := range t.events[owner] {
		if e.When.After(after) {
			result = append(result, e)
		}
	}
	return result
}

// LogStore records a store operation in the self-model.
func (sm *SelfModel) LogStore() {
	sm.mu.Lock()
	sm.TotalStores++
	sm.LastActivity = time.Now()
	sm.mu.Unlock()
}

// LogRecall records a recall and updates uncertainty estimate.
func (sm *SelfModel) LogRecall(confidence float32) {
	sm.mu.Lock()
	sm.TotalRecalls++
	sm.LastActivity = time.Now()
	// Running average of confidence = inverse uncertainty
	sm.Uncertainty = sm.Uncertainty*0.95 + (1.0-confidence)*0.05
	sm.mu.Unlock()
}

// LogThink records a think operation.
func (sm *SelfModel) LogThink() {
	sm.mu.Lock()
	sm.TotalThinks++
	sm.LastActivity = time.Now()
	sm.mu.Unlock()
}

// LogAdapt records an adaptation.
func (sm *SelfModel) LogAdapt() {
	sm.mu.Lock()
	sm.TotalAdapts++
	sm.LastActivity = time.Now()
	sm.mu.Unlock()
}

// Snapshot returns current self-knowledge.
func (sm *SelfModel) Snapshot() map[string]interface{} {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return map[string]interface{}{
		"total_stores":  sm.TotalStores,
		"total_recalls": sm.TotalRecalls,
		"total_thinks":  sm.TotalThinks,
		"total_adapts":  sm.TotalAdapts,
		"uncertainty":   sm.Uncertainty,
		"last_activity": sm.LastActivity,
	}
}
