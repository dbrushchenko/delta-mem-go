package turbovec

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
)

const DedupThreshold = float32(0.92)

type OwnerManager struct {
	dim     int
	indexes map[string]*localIndex
	mu      sync.RWMutex
}

type entry struct {
	vec         []float32
	accessCount int
}

type localIndex struct {
	entries map[string]*entry
}

func NewOwnerManager(dim int) *OwnerManager {
	if dim == 0 { dim = 768 }
	return &OwnerManager{dim: dim, indexes: make(map[string]*localIndex)}
}

func (om *OwnerManager) get(owner string) *localIndex {
	om.mu.RLock()
	if idx, ok := om.indexes[owner]; ok { om.mu.RUnlock(); return idx }
	om.mu.RUnlock()
	om.mu.Lock(); defer om.mu.Unlock()
	if idx, ok := om.indexes[owner]; ok { return idx }
	idx := &localIndex{entries: make(map[string]*entry)}
	om.indexes[owner] = idx
	return idx
}

// AddVector stores with inline dedup. If cosine > 0.92 with existing entry, supersedes it.
// Returns the ID actually stored (may be the existing one if superseded).
func (om *OwnerManager) AddVector(owner, id string, vec []float32) error {
	if len(vec) != om.dim { return fmt.Errorf("dim mismatch: got %d want %d", len(vec), om.dim) }
	idx := om.get(owner)

	// Inline dedup: check if near-duplicate exists
	for existingID, e := range idx.entries {
		if dot(vec, e.vec) > DedupThreshold {
			// Supersede: replace the old entry with new content, keep access count
			delete(idx.entries, existingID)
			idx.entries[id] = &entry{vec: vec, accessCount: e.accessCount}
			return nil
		}
	}

	idx.entries[id] = &entry{vec: vec}
	return nil
}

// SearchVector returns top-k results and bumps access count on each hit.
func (om *OwnerManager) SearchVector(owner string, query []float32, k int) ([]string, []float32, error) {
	idx := om.get(owner)
	type scored struct{ id string; s float32 }
	var results []scored
	for id, e := range idx.entries {
		results = append(results, scored{id, dot(query, e.vec)})
	}
	// Top-k selection
	ids := make([]string, 0, k); scores := make([]float32, 0, k)
	for i := 0; i < k && i < len(results); i++ {
		best := i
		for j := i + 1; j < len(results); j++ { if results[j].s > results[best].s { best = j } }
		results[i], results[best] = results[best], results[i]
		ids = append(ids, results[i].id)
		scores = append(scores, results[i].s)
		// Bump access count on retrieval
		if e, ok := idx.entries[results[i].id]; ok { e.accessCount++ }
	}
	return ids, scores, nil
}

func (om *OwnerManager) RemoveVector(owner, id string) {
	idx := om.get(owner)
	delete(idx.entries, id)
}

// Count returns total entries for an owner.
func (om *OwnerManager) Count(owner string) int {
	return len(om.get(owner).entries)
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a { if i < len(b) { s += a[i] * b[i] } }
	return s
}

// ExtractID produces a readable ID from text using entity extraction.
// Falls back to first 60 chars if no entity found.
var reServer = regexp.MustCompile(`\b([a-z]{2,}\d{2,}[a-z]*\d*)\b`)
var reProject = regexp.MustCompile(`\b([a-z]+-[a-z][\w-]*)\b`) // any hyphenated identifier (e.g. delta-mem-go, my-service)

func ExtractID(text string) string {
	if m := reProject.FindString(text); m != "" { return m }
	if m := reServer.FindString(text); m != "" { return m }
	// Fallback: first 60 chars, cleaned
	id := strings.TrimSpace(text)
	if len(id) > 60 { id = id[:60] }
	return id
}
