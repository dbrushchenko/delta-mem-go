package ibnn

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
)

type Config struct {
	InputDim, HiddenDim int
	Lambda, P, Tol, NormCap float32
	MaxIter int
}

type Layer struct {
	cfg            Config
	Weights        [][]float32
	Biases         []float32
	lateralUniform bool
	mu             sync.Mutex
}

func New(cfg Config) *Layer {
	if cfg.HiddenDim == 0 { cfg.HiddenDim = 768 }
	if cfg.InputDim == 0 { cfg.InputDim = cfg.HiddenDim }
	if cfg.Lambda == 0 { cfg.Lambda = -0.03 }
	if cfg.P == 0 { cfg.P = 10.0 }
	if cfg.MaxIter == 0 { cfg.MaxIter = 50 }
	if cfg.Tol == 0 { cfg.Tol = 1e-5 }
	return &Layer{cfg: cfg, Weights: randMatrix(cfg.HiddenDim, cfg.InputDim), Biases: randVec(cfg.HiddenDim), lateralUniform: true}
}

func (l *Layer) ForwardBatch(inputs [][]float32) [][]float32 {
	l.mu.Lock(); defer l.mu.Unlock()
	outputs := make([][]float32, len(inputs))
	for b, x := range inputs {
		if len(x) != l.cfg.InputDim { panic(fmt.Sprintf("dim mismatch batch %d: got %d want %d", b, len(x), l.cfg.InputDim)) }
		y := matVecMul(l.Weights, x); for i := range y { y[i] -= l.Biases[i] }
		z := make([]float32, l.cfg.HiddenDim); copy(z, y); l.solveImplicit(z, y)
		v := make([]float32, l.cfg.HiddenDim); for i, zi := range z { if zi > 0 { v[i] = zi } }
		outputs[b] = v
	}
	return outputs
}

func (l *Layer) solveImplicit(z, y []float32) {
	D := l.cfg.HiddenDim
	for iter := 0; iter < l.cfg.MaxIter; iter++ {
		oldZ := make([]float32, D); copy(oldZ, z); maxDiff := float32(0)
		for i := 0; i < D; i++ {
			sum := float32(0); alpha := float32(1.0) / float32(D)
			for k := 0; k < D; k++ { sum += alpha * float32(math.Tanh(float64(l.cfg.P*(z[k]-z[i])))) }
			z[i] = y[i] - l.cfg.Lambda*sum
			d := z[i] - oldZ[i]; if d < 0 { d = -d }; if d > maxDiff { maxDiff = d }
		}
		if maxDiff < l.cfg.Tol { return }
	}
}

func (l *Layer) ResetWeights() { l.mu.Lock(); l.Weights = randMatrix(l.cfg.HiddenDim, l.cfg.InputDim); l.Biases = randVec(l.cfg.HiddenDim); l.mu.Unlock() }

func (l *Layer) Save(path string) error {
	l.mu.Lock(); defer l.mu.Unlock()
	os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path); if err != nil { return err }; defer f.Close()
	return gob.NewEncoder(f).Encode(struct{ W [][]float32; B []float32 }{l.Weights, l.Biases})
}

func (l *Layer) Load(path string) error {
	l.mu.Lock(); defer l.mu.Unlock()
	f, err := os.Open(path); if err != nil { if os.IsNotExist(err) { return nil }; return err }; defer f.Close()
	var d struct{ W [][]float32; B []float32 }
	if err := gob.NewDecoder(f).Decode(&d); err != nil { return err }
	l.Weights = d.W; l.Biases = d.B; return nil
}

func matVecMul(m [][]float32, v []float32) []float32 { out := make([]float32, len(m)); for i, row := range m { for j, val := range row { out[i] += val * v[j] } }; return out }
func randMatrix(r, c int) [][]float32 { m := make([][]float32, r); seed := uint64(42); for i := range m { m[i] = make([]float32, c); for j := range m[i] { seed = seed*6364136223846793005 + 1; m[i][j] = 0.02 * float32(int32(seed>>33)) / float32(math.MaxInt32) } }; return m }
func randVec(n int) []float32 { v := make([]float32, n); seed := uint64(7); for i := range v { seed = seed*6364136223846793005 + 1; v[i] = 0.02 * float32(int32(seed>>33)) / float32(math.MaxInt32) }; return v }
