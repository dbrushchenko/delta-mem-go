package thoughts

import (
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
)

// embedder is the package-level embedding backend.
// When nil, falls back to the hash function (textToHidden).
// When set, produces real semantic vectors.
var embedder *embeddings.Embedder

// SetEmbedder injects the real nomic-embed-text ONNX embedder.
// Call this once at startup when the model is available.
// After this call, ALL thought operations use real semantics.
func SetEmbedder(e *embeddings.Embedder) {
	embedder = e
}

// embed produces a vector for text — real semantics if available, hash fallback if not.
func embed(text string) []float32 {
	if embedder != nil {
		return embedder.EmbedText(text)
	}
	return textToHidden(text)
}
