//go:build !cgo

package embeddings

const Dimensions = 768

type Embedder struct{ targetDim int }

func Get(modelPath, libPath string) (*Embedder, error) { return &Embedder{targetDim: Dimensions}, nil }
func (e *Embedder) Embed(text string) ([]float32, error) { return make([]float32, e.targetDim), nil }
func (e *Embedder) EmbedText(text string) []float32      { return make([]float32, e.targetDim) }
func (e *Embedder) SetTargetDim(dim int) {
	if dim < 64 || dim > Dimensions {
		dim = Dimensions
	}
	e.targetDim = dim
}
