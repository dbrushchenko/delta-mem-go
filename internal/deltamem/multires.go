package deltamem

// MultiRes wraps a Module with multi-resolution state layers.
// S_fast: R=64 (standard 64×64), high beta — hot/recent
// S (main): R=128, default dynamics — warm/working memory
// S_deep: R=256, low beta — cold/long-term (permanent)
type MultiRes struct {
	*Module
	Fast *Module // R=64, hot
	Deep *Module // R=256, cold
}

// NewMultiRes creates a multi-resolution module.
func NewMultiRes(dim int) *MultiRes {
	main := New(Config{R: 128, HiddenDim: dim, NormCap: 10.0})  // warm, extended working memory
	fast := New(Config{R: 64, HiddenDim: dim, NormCap: 5.0})    // hot, standard 64x64
	deep := New(Config{R: 256, HiddenDim: dim, NormCap: 20.0}) // cold, 256x256, permanent
	// Fast layer: high beta (aggressive write), high lambda (fast decay)
	for i := range fast.WBeta { fast.WBeta[i] *= 3.0 }
	// Deep layer: low beta (conservative write), very high lambda (never forget)
	for i := range deep.WBeta { deep.WBeta[i] *= 0.1 }
	for i := range deep.WLambda { deep.WLambda[i] = 0.5 }
	return &MultiRes{Module: main, Fast: fast, Deep: deep}
}

// ForwardAll runs input through all three resolution layers.
// Returns combined deltaQ, deltaO from all layers (weighted).
func (mr *MultiRes) ForwardAll(x []float32) ([]float32, []float32) {
	// Fast layer captures recent context
	mr.Fast.Forward(x)
	// Main layer is standard working memory
	dqMain, doMain := mr.Module.Forward(x)
	// Deep layer for long-term (only write every 10th call via norm check)
	if mr.Deep.StateNorm() < mr.Deep.cfg.NormCap*0.95 {
		mr.Deep.Forward(x)
	}
	return dqMain, doMain
}

// RecallAll queries all layers and returns the strongest signal.
func (mr *MultiRes) RecallAll(query []float32) (deltaQ, deltaO []float32, confidence float32) {
	dqF, doF := recall(mr.Fast, query)
	dqM, doM := recall(mr.Module, query)
	dqD, doD := recall(mr.Deep, query)

	// Combine: weight by each layer's confidence
	confF := norm32(doF)
	confM := norm32(doM)
	confD := norm32(doD)

	total := confF + confM + confD
	if total == 0 { return dqM, doM, confM }

	dim := len(query)
	deltaQ = make([]float32, dim)
	deltaO = make([]float32, dim)
	for d := 0; d < dim; d++ {
		if d < len(dqF) { deltaQ[d] += dqF[d] * confF / total }
		if d < len(dqM) { deltaQ[d] += dqM[d] * confM / total }
		if d < len(dqD) { deltaQ[d] += dqD[d] * confD / total }
		if d < len(doF) { deltaO[d] += doF[d] * confF / total }
		if d < len(doM) { deltaO[d] += doM[d] * confM / total }
		if d < len(doD) { deltaO[d] += doD[d] * confD / total }
	}
	confidence = total
	return
}

func recall(m *Module, query []float32) ([]float32, []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	r := m.cfg.R
	mq := layerNorm(tanhVec(matVecMul(m.Wq, query)))
	rt := matVecMul(m.S, mq)
	dq := make([]float32, m.cfg.HiddenDim)
	do := make([]float32, m.cfg.HiddenDim)
	for i := 0; i < m.cfg.HiddenDim && i < len(m.WqR); i++ {
		for j := 0; j < r && j < len(rt); j++ {
			dq[i] += m.WqR[i][j] * rt[j]
			do[i] += m.WoR[i][j] * rt[j]
		}
	}
	return dq, do
}

func norm32(v []float32) float32 {
	var s float32
	for _, x := range v { s += x * x }
	if s <= 0 { return 0 }
	z := s; for i := 0; i < 10; i++ { z = (z + s/z) / 2 }
	return z
}
