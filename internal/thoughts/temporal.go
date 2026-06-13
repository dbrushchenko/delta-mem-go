package thoughts

import (
	"strings"
	"sync"
	"time"
)

// Temporal tracks when memories were stored and provides recency queries.
type Temporal struct {
	events map[string][]Event
	mu     sync.RWMutex
}

type Event struct {
	ID      string
	Content string
	When    time.Time
}

// SelfModel tracks per-topic confidence and influences Think depth dynamically.
type SelfModel struct {
	TotalStores  int64
	TotalRecalls int64
	TotalThinks  int64
	TotalAdapts  int64
	LastActivity time.Time
	Uncertainty  float32
	topicConf    map[string]float32
	topicCount   map[string]int
	mu           sync.Mutex
}

func NewTemporal() *Temporal {
	return &Temporal{events: make(map[string][]Event)}
}

func NewSelfModel() *SelfModel {
	return &SelfModel{topicConf: make(map[string]float32), topicCount: make(map[string]int)}
}

// Record logs a timestamped event.
func (t *Temporal) Record(owner, id, content string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events[owner] = append(t.events[owner], Event{ID: id, Content: content, When: time.Now()})
	if len(t.events[owner]) > 1000 {
		t.events[owner] = t.events[owner][len(t.events[owner])-500:]
	}
}

// Recent returns the N most recent events.
func (t *Temporal) Recent(owner string, n int) []Event {
	t.mu.RLock()
	defer t.mu.RUnlock()
	events := t.events[owner]
	if len(events) <= n { return events }
	return events[len(events)-n:]
}

// Since returns events after a given time.
func (t *Temporal) Since(owner string, after time.Time) []Event {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var result []Event
	for _, e := range t.events[owner] {
		if e.When.After(after) { result = append(result, e) }
	}
	return result
}

func (sm *SelfModel) LogStore() {
	sm.mu.Lock()
	sm.TotalStores++
	sm.LastActivity = time.Now()
	sm.mu.Unlock()
}

// LogRecall records a recall with per-topic confidence tracking.
func (sm *SelfModel) LogRecall(confidence float32, topics ...string) {
	sm.mu.Lock()
	sm.TotalRecalls++
	sm.LastActivity = time.Now()
	sm.Uncertainty = sm.Uncertainty*0.95 + (1.0-confidence)*0.05
	for _, t := range topics {
		sm.topicCount[t]++
		sm.topicConf[t] += (confidence - sm.topicConf[t]) / float32(sm.topicCount[t])
	}
	sm.mu.Unlock()
}

func (sm *SelfModel) LogThink()  { sm.mu.Lock(); sm.TotalThinks++; sm.LastActivity = time.Now(); sm.mu.Unlock() }
func (sm *SelfModel) LogAdapt()  { sm.mu.Lock(); sm.TotalAdapts++; sm.LastActivity = time.Now(); sm.mu.Unlock() }

// SurpriseThresholdFor returns a dynamic threshold based on self-knowledge.
// Known topics → higher threshold (exit early). Unknown → lower (dig deeper).
func (sm *SelfModel) SurpriseThresholdFor(seeds []string, baseThreshold float32) float32 {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if len(sm.topicConf) == 0 { return baseThreshold }

	var totalConf float32
	var matched int
	for _, seed := range seeds {
		seedLower := strings.ToLower(seed)
		for topic, conf := range sm.topicConf {
			if strings.Contains(seedLower, topic) || strings.Contains(topic, seedLower) {
				totalConf += conf
				matched++
			}
		}
	}
	if matched == 0 {
		return baseThreshold * 0.5 // unknown → think deeper
	}
	return baseThreshold + (totalConf/float32(matched))*0.5 // known → exit sooner
}

// Snapshot returns current self-knowledge.
func (sm *SelfModel) Snapshot() map[string]interface{} {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	topics := make(map[string]float32)
	for k, v := range sm.topicConf { topics[k] = v }
	return map[string]interface{}{
		"total_stores": sm.TotalStores, "total_recalls": sm.TotalRecalls,
		"total_thinks": sm.TotalThinks, "total_adapts": sm.TotalAdapts,
		"uncertainty": sm.Uncertainty, "last_activity": sm.LastActivity,
		"topic_confidence": topics,
	}
}
