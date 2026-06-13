package nli

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Labels for NLI output (DeBERTa cross-encoder/nli-deberta-v3-xsmall)
var Labels = [3]string{"contradiction", "entailment", "neutral"}

// Checker implements thoughts.NLIChecker using DeBERTa ONNX.
type Checker struct {
	session *ort.DynamicAdvancedSession
	vocab   map[string]int
	mu      sync.Mutex
}

// New loads the DeBERTa NLI ONNX model. Returns nil if model not found.
func New(modelPath, tokenizerPath string) (*Checker, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("nli model not found: %s", modelPath)
	}

	// ORT should already be initialized by embeddings. If not, this is a no-op.
	if !ort.IsInitialized() {
		return nil, fmt.Errorf("ort not initialized (embeddings must load first)")
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"}, nil)
	if err != nil {
		return nil, fmt.Errorf("nli session: %w", err)
	}

	c := &Checker{session: session}
	if data, err := os.ReadFile(tokenizerPath); err == nil {
		c.vocab = parseVocab(data)
	}
	return c, nil
}

// Check runs NLI inference on a text pair.
// Returns label ("contradiction", "entailment", "neutral") and confidence.
func (c *Checker) Check(textA, textB string) (string, float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	ids, mask := c.tokenizePair(textA, textB)
	seqLen := int64(len(ids))

	idsTensor, _ := ort.NewTensor(ort.NewShape(1, seqLen), ids)
	maskTensor, _ := ort.NewTensor(ort.NewShape(1, seqLen), mask)
	outData := make([]float32, 3)
	outTensor, _ := ort.NewTensor(ort.NewShape(1, 3), outData)
	defer idsTensor.Destroy()
	defer maskTensor.Destroy()
	defer outTensor.Destroy()

	err := c.session.Run(
		[]ort.ArbitraryTensor{idsTensor, maskTensor},
		[]ort.ArbitraryTensor{outTensor},
	)
	if err != nil {
		return "neutral", 0
	}

	// Softmax
	logits := outTensor.GetData()
	maxL := logits[0]
	for _, v := range logits[1:] { if v > maxL { maxL = v } }
	var sum float64
	probs := make([]float64, 3)
	for i, v := range logits {
		probs[i] = math.Exp(float64(v - maxL))
		sum += probs[i]
	}
	for i := range probs { probs[i] /= sum }

	// Find best label
	bestIdx := 0
	for i := 1; i < 3; i++ { if probs[i] > probs[bestIdx] { bestIdx = i } }

	return Labels[bestIdx], float32(probs[bestIdx])
}

// tokenizePair creates [CLS] textA [SEP] textB [SEP] input
func (c *Checker) tokenizePair(textA, textB string) ([]int64, []int64) {
	maxLen := 256
	tokA := c.tokenize(textA)
	tokB := c.tokenize(textB)
	// Truncate to fit
	maxPerSide := (maxLen - 3) / 2
	if len(tokA) > maxPerSide { tokA = tokA[:maxPerSide] }
	if len(tokB) > maxPerSide { tokB = tokB[:maxPerSide] }

	// [CLS]=1, [SEP]=2 for DeBERTa
	ids := make([]int64, 0, len(tokA)+len(tokB)+3)
	ids = append(ids, 1) // [CLS]
	ids = append(ids, tokA...)
	ids = append(ids, 2) // [SEP]
	ids = append(ids, tokB...)
	ids = append(ids, 2) // [SEP]

	mask := make([]int64, len(ids))
	for i := range mask { mask[i] = 1 }
	return ids, mask
}

func (c *Checker) tokenize(text string) []int64 {
	if c.vocab == nil {
		// Byte fallback
		b := []byte(strings.ToLower(text))
		out := make([]int64, len(b))
		for i, ch := range b { out[i] = int64(ch) + 1000 }
		return out
	}
	return wordpiece(text, c.vocab)
}

func wordpiece(text string, vocab map[string]int) []int64 {
	text = strings.ToLower(text)
	words := strings.Fields(text)
	var tokens []int64
	for _, word := range words {
		if id, ok := vocab[word]; ok { tokens = append(tokens, int64(id)); continue }
		remaining := word
		first := true
		for len(remaining) > 0 {
			found := false
			for end := len(remaining); end > 0; end-- {
				sub := remaining[:end]
				if !first { sub = "##" + sub }
				if id, ok := vocab[sub]; ok {
					tokens = append(tokens, int64(id)); remaining = remaining[end:]; first = false; found = true; break
				}
			}
			if !found { tokens = append(tokens, 3); break } // [UNK]=3
		}
	}
	return tokens
}

func parseVocab(data []byte) map[string]int {
	// Try HuggingFace tokenizer.json format: model.vocab as [[token, score], ...]
	var tok struct {
		Model struct {
			Vocab json.RawMessage `json:"vocab"`
		} `json:"model"`
	}
	if json.Unmarshal(data, &tok) != nil { return nil }

	// Try list format: [[token, score], ...]
	var listVocab [][]interface{}
	if json.Unmarshal(tok.Model.Vocab, &listVocab) == nil && len(listVocab) > 0 {
		vocab := make(map[string]int, len(listVocab))
		for i, entry := range listVocab {
			if len(entry) >= 1 {
				if token, ok := entry[0].(string); ok {
					vocab[token] = i
				}
			}
		}
		if len(vocab) > 0 { return vocab }
	}

	// Try dict format: {token: id}
	var dictVocab map[string]int
	if json.Unmarshal(tok.Model.Vocab, &dictVocab) == nil && len(dictVocab) > 0 {
		return dictVocab
	}

	return nil
}
