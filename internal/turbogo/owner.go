package turbogo

import (
	"fmt"
	"path/filepath"
	"sync"
)

type OwnerManager struct {
	cfg     Config
	dataDir string
	indexes map[string]*Index
	mu      sync.RWMutex
}

func NewOwnerManager(cfg Config, dataDir string) *OwnerManager {
	if cfg.Dim == 0 {
		cfg.Dim = 768
	}
	return &OwnerManager{cfg: cfg, dataDir: dataDir, indexes: make(map[string]*Index)}
}

func (om *OwnerManager) Get(owner string) (*Index, error) {
	om.mu.RLock()
	if idx, ok := om.indexes[owner]; ok {
		om.mu.RUnlock()
		return idx, nil
	}
	om.mu.RUnlock()
	om.mu.Lock()
	defer om.mu.Unlock()
	if idx, ok := om.indexes[owner]; ok {
		return idx, nil
	}
	idx := NewIndex(om.cfg)
	if err := idx.Load(filepath.Join(om.dataDir, owner+".turbogo")); err != nil {
		return nil, fmt.Errorf("load turbogo for %s: %w", owner, err)
	}
	om.indexes[owner] = idx
	return idx, nil
}

func (om *OwnerManager) save(owner string) error {
	idx, ok := om.indexes[owner]
	if !ok {
		return nil
	}
	return idx.Save(filepath.Join(om.dataDir, owner+".turbogo"))
}

func (om *OwnerManager) AddVector(owner, id string, vec []float32) error {
	idx, err := om.Get(owner)
	if err != nil {
		return err
	}
	if err := idx.Add(id, vec); err != nil {
		return err
	}
	om.mu.RLock()
	defer om.mu.RUnlock()
	return om.save(owner)
}

func (om *OwnerManager) SearchVector(owner string, query []float32, k int) ([]string, []float32, error) {
	idx, err := om.Get(owner)
	if err != nil {
		return nil, nil, err
	}
	return idx.Search(query, k)
}
