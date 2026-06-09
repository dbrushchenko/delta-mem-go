package turbovec

import (
	"fmt"
	"sync"
)

type OwnerManager struct {
	dim     int
	indexes map[string]*localIndex
	mu      sync.RWMutex
}

type localIndex struct {
	vectors map[string][]float32
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
	idx := &localIndex{vectors: make(map[string][]float32)}
	om.indexes[owner] = idx
	return idx
}

func (om *OwnerManager) AddVector(owner, id string, vec []float32) error {
	if len(vec) != om.dim { return fmt.Errorf("dim mismatch: got %d want %d", len(vec), om.dim) }
	idx := om.get(owner); idx.vectors[id] = vec; return nil
}

func (om *OwnerManager) SearchVector(owner string, query []float32, k int) ([]string, []float32, error) {
	idx := om.get(owner)
	type scored struct{ id string; s float32 }
	var results []scored
	for id, vec := range idx.vectors {
		var dot float32; for i := range query { if i < len(vec) { dot += query[i] * vec[i] } }
		results = append(results, scored{id, dot})
	}
	// simple top-k
	ids := make([]string, 0, k); scores := make([]float32, 0, k)
	for i := 0; i < k && i < len(results); i++ {
		best := i
		for j := i + 1; j < len(results); j++ { if results[j].s > results[best].s { best = j } }
		results[i], results[best] = results[best], results[i]
		ids = append(ids, results[i].id); scores = append(scores, results[i].s)
	}
	return ids, scores, nil
}
