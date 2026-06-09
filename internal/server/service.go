package server

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/gemma"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

const Dimensions = 768

type Service struct {
	gemma       *gemma.Client
	turbovecCli *turbovec.Client
	started     time.Time
	stores      int64
	recalls     int64
}

func New(gemmaClient *gemma.Client, turbovecClient *turbovec.Client) *Service {
	return &Service{
		gemma:       gemmaClient,
		turbovecCli: turbovecClient,
		started:     time.Now(),
	}
}

func (s *Service) Store(ctx context.Context, owner, key, content string) (float32, error) {
	s.stores++
	return 1.0, nil
}

func (s *Service) Recall(ctx context.Context, owner, query string) ([]float32, float32, error) {
	s.recalls++
	return nil, 0.0, nil
}

func (s *Service) StoreHidden(ctx context.Context, owner string, hidden []float32) (float32, error) {
	s.stores++
	return 1.0, nil
}

func (s *Service) RecallHidden(ctx context.Context, owner string, query []float32) ([]float32, []float32, float32, error) {
	s.recalls++
	return nil, nil, 0.0, nil
}

func (s *Service) Health() (int, float32, int64, int64, string) {
	return 1, 1.0, s.stores, s.recalls, time.Since(s.started).String()
}

func (s *Service) ResetState(ctx context.Context, owner string) error {
	return nil
}

func (s *Service) IBNNForward(ctx context.Context, owner, text string) ([]float32, error) {
	hidden := textToHidden(text)
	return hidden, nil
}

func (s *Service) IBNNForwardHidden(ctx context.Context, owner string, hidden []float32) ([]float32, error) {
	return hidden, nil
}

func (s *Service) TurboAdd(ctx context.Context, owner, id string, vector []float32) error {
	if s.turbovecCli == nil {
		return fmt.Errorf("turbovec client not initialized")
	}
	return s.turbovecCli.Add(ctx, owner, id, vector)
}

func (s *Service) TurboSearch(ctx context.Context, owner string, query []float32, k int) ([]string, []float32, error) {
	if s.turbovecCli == nil {
		return nil, nil, fmt.Errorf("turbovec client not initialized")
	}
	return s.turbovecCli.Search(ctx, owner, query, k)
}

func (s *Service) Generate(ctx context.Context, owner, prompt string) (string, error) {
	if s.gemma == nil {
		return "", fmt.Errorf("gemma client not initialized")
	}
	return s.gemma.Generate(ctx, prompt)
}

func textToHidden(text string) []float32 {
	hidden := make([]float32, Dimensions)
	seed := uint64(0)
	for _, c := range text {
		seed = seed*31 + uint64(c)
	}
	for i := range hidden {
		seed = seed*6364136223846793005 + 1
		hidden[i] = float32(math.Sin(float64(seed)/1e15)) * 0.1
	}
	var norm float32
	for _, v := range hidden {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range hidden {
			hidden[i] /= norm
		}
	}
	return hidden
}
