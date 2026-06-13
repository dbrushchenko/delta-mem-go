package deltamem

import (
	"fmt"
	"path/filepath"
	"sync"
)

type OwnerManager struct {
	cfg     Config
	dataDir string
	modules map[string]*MultiRes
	mu      sync.RWMutex
	stats   Stats
}

type Stats struct{ TotalStores, TotalRecalls int64 }

func NewOwnerManager(cfg Config, dataDir string) *OwnerManager {
	if cfg.HiddenDim == 0 { cfg.HiddenDim = 768 }
	return &OwnerManager{cfg: cfg, dataDir: dataDir, modules: make(map[string]*MultiRes)}
}

func (om *OwnerManager) Get(owner string) (*Module, error) {
	mr, err := om.GetMultiRes(owner)
	if err != nil { return nil, err }
	return mr.Module, nil
}

func (om *OwnerManager) GetMultiRes(owner string) (*MultiRes, error) {
	om.mu.RLock()
	if m, ok := om.modules[owner]; ok { om.mu.RUnlock(); return m, nil }
	om.mu.RUnlock()
	om.mu.Lock(); defer om.mu.Unlock()
	if m, ok := om.modules[owner]; ok { return m, nil }
	mr := NewMultiRes(om.cfg.HiddenDim)
	if err := mr.Module.LoadState(filepath.Join(om.dataDir, owner+".state")); err != nil {
		return nil, fmt.Errorf("load state for %s: %w", owner, err)
	}
	om.modules[owner] = mr
	return mr, nil
}

func (om *OwnerManager) Save(owner string) error {
	om.mu.RLock(); mr, ok := om.modules[owner]; om.mu.RUnlock()
	if !ok { return nil }
	return mr.Module.SaveState(filepath.Join(om.dataDir, owner+".state"))
}

func (om *OwnerManager) Store(owner string, hidden []float32) (float32, error) {
	mr, err := om.GetMultiRes(owner); if err != nil { return 0, err }
	mr.ForwardAll(hidden); om.stats.TotalStores++
	if err := om.Save(owner); err != nil { return 0, err }
	return mr.Module.StateNorm(), nil
}

func (om *OwnerManager) Recall(owner string, query []float32) ([]float32, []float32, float32, error) {
	mr, err := om.GetMultiRes(owner); if err != nil { return nil, nil, 0, err }
	dq, do, conf := mr.RecallAll(query)
	om.stats.TotalRecalls++
	return dq, do, conf, nil
}

func (om *OwnerManager) ActiveOwners() int { om.mu.RLock(); defer om.mu.RUnlock(); return len(om.modules) }
func (om *OwnerManager) GetStats() Stats { return om.stats }
func (om *OwnerManager) ResetOwner(owner string) error {
	mr, err := om.GetMultiRes(owner); if err != nil { return err }
	mr.Module.ResetState(); mr.Fast.ResetState(); mr.Deep.ResetState()
	return om.Save(owner)
}
