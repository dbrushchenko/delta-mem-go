package embeddings

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const Dimensions = 768

type Embedder struct {
	session   *ort.DynamicAdvancedSession
	vocab     map[string]int
	targetDim int
	mu        sync.Mutex
}

func Get(modelPath, libPath string) (*Embedder, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path required")
	}
	if libPath == "" {
		candidates := []string{
			filepath.Join(filepath.Dir(modelPath), "onnxruntime.dll"),
			filepath.Join(filepath.Dir(modelPath), "libonnxruntime.so"),
		}
		// Platform-specific fallbacks via environment
		if ortDir := os.Getenv("ORT_LIB_DIR"); ortDir != "" {
			candidates = append(candidates, filepath.Join(ortDir, "onnxruntime.dll"))
			candidates = append(candidates, filepath.Join(ortDir, "libonnxruntime.so"))
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				libPath = c
				break
			}
		}
		if libPath == "" {
			return nil, fmt.Errorf("onnxruntime library not found (set ORT_LIB_DIR or place next to model)")
		}
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"}, nil)
	if err != nil {
		return nil, fmt.Errorf("ort session: %w", err)
	}

	e := &Embedder{session: session, targetDim: Dimensions}
	tokPath := filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	if data, err := os.ReadFile(tokPath); err == nil {
		e.vocab = parseVocab(data)
	}
	return e, nil
}

func (e *Embedder) Embed(text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	ids, mask := e.tokenize("search_document: " + text)
	seqLen := int64(len(ids))

	idsTensor, err := ort.NewTensor(ort.NewShape(1, seqLen), ids)
	if err != nil { return nil, err }
	defer idsTensor.Destroy()

	maskTensor, err := ort.NewTensor(ort.NewShape(1, seqLen), mask)
	if err != nil { return nil, err }
	defer maskTensor.Destroy()

	outData := make([]float32, seqLen*Dimensions)
	outTensor, err := ort.NewTensor(ort.NewShape(1, seqLen, Dimensions), outData)
	if err != nil { return nil, err }
	defer outTensor.Destroy()

	if err := e.session.Run(
		[]ort.ArbitraryTensor{idsTensor, maskTensor},
		[]ort.ArbitraryTensor{outTensor},
	); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	// Mean pool over non-padding tokens
	raw := outTensor.GetData()
	result := make([]float32, e.targetDim)
	var count float32
	for t := int64(0); t < seqLen; t++ {
		if mask[t] == 0 { continue }
		off := t * Dimensions
		for d := 0; d < e.targetDim; d++ { result[d] += raw[off+int64(d)] }
		count++
	}
	if count > 0 { for d := range result { result[d] /= count } }

	// L2 normalize
	var norm float32
	for _, v := range result { norm += v * v }
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 { for i := range result { result[i] /= norm } }
	return result, nil
}

func (e *Embedder) EmbedText(text string) []float32 {
	v, err := e.Embed(text)
	if err != nil { return make([]float32, e.targetDim) }
	return v
}

func (e *Embedder) SetTargetDim(dim int) {
	if dim < 64 || dim > Dimensions { dim = Dimensions }
	e.targetDim = dim
}

func (e *Embedder) Close() {
	if e.session != nil { e.session.Destroy() }
	ort.DestroyEnvironment()
}

func (e *Embedder) tokenize(text string) ([]int64, []int64) {
	maxLen := 512
	var tokens []int64
	if e.vocab != nil {
		tokens = e.wordpiece(text)
	} else {
		b := []byte(text)
		if len(b) > maxLen-2 { b = b[:maxLen-2] }
		tokens = make([]int64, len(b))
		for i, c := range b { tokens[i] = int64(c) + 1000 }
	}
	if len(tokens) > maxLen-2 { tokens = tokens[:maxLen-2] }
	ids := make([]int64, len(tokens)+2)
	mask := make([]int64, len(tokens)+2)
	ids[0] = 101; mask[0] = 1
	for i, t := range tokens { ids[i+1] = t; mask[i+1] = 1 }
	ids[len(tokens)+1] = 102; mask[len(tokens)+1] = 1
	return ids, mask
}

func (e *Embedder) wordpiece(text string) []int64 {
	text = strings.ToLower(text)
	words := strings.Fields(text)
	var tokens []int64
	for _, word := range words {
		if id, ok := e.vocab[word]; ok { tokens = append(tokens, int64(id)); continue }
		remaining := word
		first := true
		for len(remaining) > 0 {
			found := false
			for end := len(remaining); end > 0; end-- {
				sub := remaining[:end]
				if !first { sub = "##" + sub }
				if id, ok := e.vocab[sub]; ok {
					tokens = append(tokens, int64(id)); remaining = remaining[end:]; first = false; found = true; break
				}
			}
			if !found { tokens = append(tokens, 100); break }
		}
	}
	return tokens
}

func parseVocab(data []byte) map[string]int {
	var tok struct{ Model struct{ Vocab map[string]int `json:"vocab"` } `json:"model"` }
	if json.Unmarshal(data, &tok) != nil { return nil }
	return tok.Model.Vocab
}
