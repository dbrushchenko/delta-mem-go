package thoughts

// VectorStore is the interface for any vector retrieval backend.
// Both turbovec (in-memory, simple) and turbogo (quantized, persistent) satisfy this.
type VectorStore interface {
	AddVector(owner, id string, vec []float32) error
	SearchVector(owner string, query []float32, k int) ([]string, []float32, error)
	RemoveVector(owner, id string)
}
