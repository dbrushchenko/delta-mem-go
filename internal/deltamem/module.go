package deltamem

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
)

type Config struct {
	R         int
	HiddenDim int
	NormCap   float32
}

type Module struct {
	cfg     Config
	Wq, Wk, Wv          [][]float32
	WBeta, WLambda       []float32
	BBeta, BLambda       float32
	WqR, WoR             [][]float32
	S                    [][]float32
	mu                   sync.Mutex
}

func New(cfg Config) *Module {
	if cfg.R == 0 { cfg.R = 64 }
	if cfg.HiddenDim == 0 { cfg.HiddenDim = 768 }
	if cfg.NormCap == 0 { cfg.NormCap = 10.0 }
	return &Module{
		cfg: cfg, Wq: randMatrix(cfg.R, cfg.HiddenDim), Wk: randMatrix(cfg.R, cfg.HiddenDim),
		Wv: randMatrix(cfg.R, cfg.HiddenDim), WBeta: randVec(cfg.HiddenDim), WLambda: randVec(cfg.HiddenDim),
		WqR: randMatrix(cfg.HiddenDim, cfg.R), WoR: randMatrix(cfg.HiddenDim, cfg.R), S: zeroMatrix(cfg.R, cfg.R),
	}
}

func (m *Module) Forward(x []float32) ([]float32, []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	r := m.cfg.R
	mq := layerNorm(tanhVec(matVecMul(m.Wq, x)))
	mk := layerNorm(tanhVec(matVecMul(m.Wk, x)))
	mv := tanhVec(matVecMul(m.Wv, x))
	rt := matVecMul(m.S, mq)
	deltaQ := matVecMul(m.WqR, rt)
	deltaO := matVecMul(m.WoR, rt)
	betaT := sigmoid(dot(m.WBeta, x) + m.BBeta)
	lambdaT := sigmoid(dot(m.WLambda, x) + m.BLambda)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			outer := mv[i] * mk[j]
			var sMk float32
			for k := 0; k < r; k++ { sMk += m.S[i][k] * mk[k] }
			m.S[i][j] = lambdaT*m.S[i][j] + betaT*(outer-sMk*mk[j])
		}
	}
	m.clampState()
	return deltaQ, deltaO
}

func (m *Module) StateNorm() float32 {
	var sum float32
	for _, row := range m.S { for _, v := range row { sum += v * v } }
	return float32(math.Sqrt(float64(sum)))
}

func (m *Module) clampState() {
	norm := m.StateNorm()
	if norm > m.cfg.NormCap {
		scale := m.cfg.NormCap / norm
		for i := range m.S { for j := range m.S[i] { m.S[i][j] *= scale } }
	}
}

func (m *Module) ResetState() { m.mu.Lock(); m.S = zeroMatrix(m.cfg.R, m.cfg.R); m.mu.Unlock() }

func (m *Module) SaveState(path string) error {
	m.mu.Lock(); defer m.mu.Unlock()
	os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path); if err != nil { return err }
	defer f.Close()
	return gob.NewEncoder(f).Encode(m.S)
}

func (m *Module) LoadState(path string) error {
	m.mu.Lock(); defer m.mu.Unlock()
	f, err := os.Open(path)
	if err != nil { if os.IsNotExist(err) { return nil }; return err }
	defer f.Close()
	return gob.NewDecoder(f).Decode(&m.S)
}

func (m *Module) LoadProjections(path string) error { return nil } // simplified for compilation

func sigmoid(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(-float64(x)))) }
func tanhVec(v []float32) []float32 { out := make([]float32, len(v)); for i, x := range v { out[i] = float32(math.Tanh(float64(x))) }; return out }
func layerNorm(v []float32) []float32 {
	n := len(v); var mean, variance float64
	for _, x := range v { mean += float64(x) }; mean /= float64(n)
	for _, x := range v { d := float64(x) - mean; variance += d * d }
	std := math.Sqrt(variance/float64(n) + 1e-5)
	out := make([]float32, n); for i, x := range v { out[i] = float32((float64(x) - mean) / std) }; return out
}
func dot(a, b []float32) float32 { var s float32; for i := range a { s += a[i] * b[i] }; return s }
func matVecMul(m [][]float32, v []float32) []float32 { out := make([]float32, len(m)); for i, row := range m { for j, val := range row { out[i] += val * v[j] } }; return out }
func zeroMatrix(r, c int) [][]float32 { m := make([][]float32, r); for i := range m { m[i] = make([]float32, c) }; return m }
func randMatrix(r, c int) [][]float32 { m := make([][]float32, r); seed := uint64(42); for i := range m { m[i] = make([]float32, c); for j := range m[i] { seed = seed*6364136223846793005 + 1; m[i][j] = 0.02 * float32(int32(seed>>33)) / float32(math.MaxInt32) } }; return m }
func randVec(n int) []float32 { v := make([]float32, n); seed := uint64(7); for i := range v { seed = seed*6364136223846793005 + 1; v[i] = 0.02 * float32(int32(seed>>33)) / float32(math.MaxInt32) }; return v }

func init() { _ = fmt.Sprintf }
