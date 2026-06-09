package turbogo

import (
	"math/rand"
	"sync"
)

type RotationMatrix struct {
	matrix [][]float32
	once   sync.Once
	dim    int
}

func NewRotationMatrix(dim int) *RotationMatrix {
	return &RotationMatrix{dim: dim}
}

func (r *RotationMatrix) Rotate(vec []float32) []float32 {
	r.once.Do(r.generateMatrix)
	rotated := make([]float32, len(vec))
	for i := 0; i < r.dim; i++ {
		for j := 0; j < r.dim; j++ {
			rotated[i] += r.matrix[i][j] * vec[j]
		}
	}
	return rotated
}

func (r *RotationMatrix) generateMatrix() {
	r.matrix = make([][]float32, r.dim)
	for i := range r.matrix {
		r.matrix[i] = make([]float32, r.dim)
		for j := range r.matrix[i] {
			r.matrix[i][j] = rand.Float32()*2 - 1
		}
	}
}
