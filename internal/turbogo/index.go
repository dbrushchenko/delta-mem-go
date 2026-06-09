package turbogo

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

var ErrDimMismatch = fmt.Errorf("dimension mismatch")

type VectorEntry struct {
	Quantized  []byte
	Norm       float32
	Renorm     float32
	ExternalID string
}

type Index struct {
	cfg      Config
	rotation *RotationMatrix
	calib    []float32
	entries  []VectorEntry
	idMap    map[string]int
	mu       sync.RWMutex
}

func NewIndex(cfg Config) *Index {
	if cfg.Dim == 0 {
		cfg.Dim = 768
	}
	if cfg.BitWidth != 2 && cfg.BitWidth != 4 {
		cfg.BitWidth = 4
	}
	return &Index{cfg: cfg, rotation: NewRotationMatrix(cfg.Dim), idMap: make(map[string]int)}
}

func (idx *Index) Add(externalID string, vec []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if len(vec) != idx.cfg.Dim {
		return ErrDimMismatch
	}
	rotated := idx.rotation.Rotate(vec)
	if len(idx.entries) == 0 {
		idx.calib = make([]float32, idx.cfg.Dim)
		copy(idx.calib, rotated)
	}
	for i := range rotated {
		rotated[i] -= idx.calib[i]
	}
	quantized := make([]byte, idx.cfg.Dim)
	for i, v := range rotated {
		quantized[i] = Quantize(v, idx.cfg.BitWidth)
	}
	norm := float32(0)
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	packed := packBits(quantized, idx.cfg.BitWidth)
	entry := VectorEntry{Quantized: packed, Norm: norm, Renorm: 1.0 / (norm*0.92 + 1e-9), ExternalID: externalID}
	idx.entries = append(idx.entries, entry)
	idx.idMap[externalID] = len(idx.entries) - 1
	return nil
}

func (idx *Index) Search(query []float32, k int) ([]string, []float32, error) {
	start := time.Now()
	defer func() { _ = time.Since(start) }()
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(query) != idx.cfg.Dim {
		return nil, nil, ErrDimMismatch
	}
	rotatedQ := idx.rotation.Rotate(query)
	h := &scoredHeap{}
	heap.Init(h)
	for _, e := range idx.entries {
		unpacked := unpackBits(e.Quantized, idx.cfg.BitWidth, idx.cfg.Dim)
		score := float32(0)
		for i := 0; i < idx.cfg.Dim; i++ {
			score += rotatedQ[i] * Dequantize(unpacked[i], idx.cfg.BitWidth)
		}
		score *= e.Norm * e.Renorm
		heap.Push(h, scoredItem{id: e.ExternalID, score: score})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	ids := make([]string, h.Len())
	scores := make([]float32, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(scoredItem)
		ids[i] = item.id
		scores[i] = item.score
	}
	return ids, scores, nil
}

func (idx *Index) Delete(externalID string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	i, exists := idx.idMap[externalID]
	if !exists {
		return nil
	}
	last := len(idx.entries) - 1
	idx.entries[i] = idx.entries[last]
	idx.idMap[idx.entries[i].ExternalID] = i
	delete(idx.idMap, externalID)
	idx.entries = idx.entries[:last]
	return nil
}

func (idx *Index) Save(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(struct {
		Cfg     Config
		Calib   []float32
		Entries []VectorEntry
		IDMap   map[string]int
	}{idx.cfg, idx.calib, idx.entries, idx.idMap})
}

func (idx *Index) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	var data struct {
		Cfg     Config
		Calib   []float32
		Entries []VectorEntry
		IDMap   map[string]int
	}
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return err
	}
	idx.cfg = data.Cfg
	idx.calib = data.Calib
	idx.entries = data.Entries
	idx.idMap = data.IDMap
	return nil
}

type scoredItem struct {
	id    string
	score float32
}

type scoredHeap []scoredItem

func (h scoredHeap) Len() int            { return len(h) }
func (h scoredHeap) Less(i, j int) bool   { return h[i].score < h[j].score }
func (h scoredHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }
func (h *scoredHeap) Push(x any)          { *h = append(*h, x.(scoredItem)) }
func (h *scoredHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
