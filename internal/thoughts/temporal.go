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
	domains      []domain // embedding-based knowledge areas
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

// Confidence level for AmIConfident
type Confidence int
const (
	NeverSeen  Confidence = iota // topic has no match in any known domain
	Uncertain                    // seen before but low confidence
	Confident                    // high confidence — answer directly
)

func (c Confidence) String() string {
	switch c {
	case Confident: return "yes"
	case Uncertain: return "no"
	default: return "never_seen"
	}
}

// domain is an embedding-based knowledge area the system has accumulated.
type domain struct {
	centroid []float32
	conf     float32
	count    int
}

// AmIConfident checks if the system knows a topic using embedding similarity.
// Uses pre-computed domain centroids for O(domains) lookup — fast.
// Falls back to string matching if no embeddings available.
func (sm *SelfModel) AmIConfident(topicVec []float32) Confidence {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if len(sm.domains) == 0 {
		return NeverSeen
	}

	// Find closest domain centroid
	var bestSim float32
	var bestConf float32
	for _, d := range sm.domains {
		sim := dotSM(topicVec, d.centroid)
		if sim > bestSim {
			bestSim = sim
			bestConf = d.conf
		}
	}

	if bestSim < 0.5 {
		return NeverSeen
	}
	if bestConf > 0.04 { // tuned to δ-mem confidence range
		return Confident
	}
	return Uncertain
}

// LearnDomain updates or creates a domain centroid from recall activity.
// Call after embedding is already computed (zero additional embed cost).
func (sm *SelfModel) LearnDomain(vec []float32, confidence float32) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Find matching domain (cosine > 0.7)
	for i, d := range sm.domains {
		if dotSM(vec, d.centroid) > 0.7 {
			// Update centroid (running average)
			sm.domains[i].count++
			alpha := 1.0 / float32(sm.domains[i].count)
			for j := range d.centroid {
				sm.domains[i].centroid[j] += alpha * (vec[j] - d.centroid[j])
			}
			sm.domains[i].conf += alpha * (confidence - sm.domains[i].conf)
			return
		}
	}

	// New domain
	centroid := make([]float32, len(vec))
	copy(centroid, vec)
	sm.domains = append(sm.domains, domain{centroid: centroid, conf: confidence, count: 1})

	// Cap domains (keep top 50 by count)
	if len(sm.domains) > 50 {
		minIdx, minCount := 0, sm.domains[0].count
		for i, d := range sm.domains {
			if d.count < minCount { minIdx = i; minCount = d.count }
		}
		sm.domains[minIdx] = sm.domains[len(sm.domains)-1]
		sm.domains = sm.domains[:len(sm.domains)-1]
	}
}

func dotSM(a, b []float32) float32 {
	var s float32
	for i := range a { if i < len(b) { s += a[i] * b[i] } }
	return s
}
