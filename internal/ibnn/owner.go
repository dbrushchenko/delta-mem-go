package ibnn

import (
	"fmt"
	"path/filepath"
	"sync"
)

type OwnerManager struct {
	cfg     Config
	dataDir string
	modules map[string]*Layer
	mu      sync.RWMutex
	stats   struct{ TotalForwards int64 }
}

func NewOwnerManager(cfg Config, dataDir string) *OwnerManager {
	if cfg.HiddenDim == 0 { cfg.HiddenDim = 768 }
	if cfg.InputDim == 0 { cfg.InputDim = cfg.HiddenDim }
	return &OwnerManager{cfg: cfg, dataDir: dataDir, modules: make(map[string]*Layer)}
}

func (om *OwnerManager) Get(owner string) (*Layer, error) {
	om.mu.RLock()
	if m, ok := om.modules[owner]; ok { om.mu.RUnlock(); return m, nil }
	om.mu.RUnlock()
	om.mu.Lock(); defer om.mu.Unlock()
	if m, ok := om.modules[owner]; ok { return m, nil }
	m := New(om.cfg)
	if err := m.Load(filepath.Join(om.dataDir, owner+".ibnn.state")); err != nil {
		return nil, fmt.Errorf("load ibnn for %s: %w", owner, err)
	}
	om.modules[owner] = m
	return m, nil
}

func (om *OwnerManager) ForwardBatch(owner string, inputs [][]float32) ([][]float32, error) {
	m, err := om.Get(owner); if err != nil { return nil, err }
	out := m.ForwardBatch(inputs); om.stats.TotalForwards++
	statePath := filepath.Join(om.dataDir, owner+".ibnn.state")
	m.Save(statePath)
	return out, nil
}
