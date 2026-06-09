package deltamem

import (
	"fmt"
	"math"
	"path/filepath"
	"sync"
)

type OwnerManager struct {
	cfg     Config
	dataDir string
	modules map[string]*Module
	mu      sync.RWMutex
	stats   Stats
}

type Stats struct{ TotalStores, TotalRecalls int64 }

func NewOwnerManager(cfg Config, dataDir string) *OwnerManager {
	if cfg.HiddenDim == 0 { cfg.HiddenDim = 768 }
	return &OwnerManager{cfg: cfg, dataDir: dataDir, modules: make(map[string]*Module)}
}

func (om *OwnerManager) Get(owner string) (*Module, error) {
	om.mu.RLock()
	if m, ok := om.modules[owner]; ok { om.mu.RUnlock(); return m, nil }
	om.mu.RUnlock()
	om.mu.Lock(); defer om.mu.Unlock()
	if m, ok := om.modules[owner]; ok { return m, nil }
	m := New(om.cfg)
	if err := m.LoadState(filepath.Join(om.dataDir, owner+".state")); err != nil {
		return nil, fmt.Errorf("load state for %s: %w", owner, err)
	}
	om.modules[owner] = m
	return m, nil
}

func (om *OwnerManager) Save(owner string) error {
	om.mu.RLock(); m, ok := om.modules[owner]; om.mu.RUnlock()
	if !ok { return nil }
	return m.SaveState(filepath.Join(om.dataDir, owner+".state"))
}

func (om *OwnerManager) Store(owner string, hidden []float32) (float32, error) {
	m, err := om.Get(owner); if err != nil { return 0, err }
	m.Forward(hidden); om.stats.TotalStores++
	if err := om.Save(owner); err != nil { return 0, err }
	return m.StateNorm(), nil
}

func (om *OwnerManager) Recall(owner string, query []float32) ([]float32, []float32, float32, error) {
	m, err := om.Get(owner); if err != nil { return nil, nil, 0, err }
	m.mu.Lock()
	mq := layerNorm(tanhVec(matVecMul(m.Wq, query)))
	rt := matVecMul(m.S, mq)
	deltaQ := matVecMul(m.WqR, rt)
	deltaO := matVecMul(m.WoR, rt)
	m.mu.Unlock()
	var norm float32; for _, v := range deltaO { norm += v * v }
	om.stats.TotalRecalls++
	return deltaQ, deltaO, float32(math.Sqrt(float64(norm))), nil
}

func (om *OwnerManager) ActiveOwners() int { om.mu.RLock(); defer om.mu.RUnlock(); return len(om.modules) }
func (om *OwnerManager) GetStats() Stats { return om.stats }
func (om *OwnerManager) ResetOwner(owner string) error {
	m, err := om.Get(owner); if err != nil { return err }; m.ResetState(); return om.Save(owner)
}
func (om *OwnerManager) SetProjections(tmpl *Module) { om.mu.Lock(); om.cfg = tmpl.cfg; om.mu.Unlock() }
