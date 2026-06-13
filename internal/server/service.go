package server

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/gemma"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

const Dimensions = 768

type Service struct {
	om          *deltamem.OwnerManager
	ibnnOM      *ibnn.OwnerManager
	turboOM     *turbovec.OwnerManager
	gemma       *gemma.Client
	turbovecCli *turbovec.Client
	thoughts    *thoughts.Engine
	embedder    *embeddings.Embedder
	started     time.Time
}

func New(deltaOM *deltamem.OwnerManager, ibnnOM *ibnn.OwnerManager, turboOM *turbovec.OwnerManager, gemmaClient *gemma.Client, turbovecClient *turbovec.Client, emb ...*embeddings.Embedder) *Service {
	var e *embeddings.Embedder
	if len(emb) > 0 { e = emb[0] }
	return &Service{
		om: deltaOM, ibnnOM: ibnnOM, turboOM: turboOM, gemma: gemmaClient, turbovecCli: turbovecClient,
		thoughts: thoughts.New(deltaOM, ibnnOM, turboOM, gemmaClient),
		embedder: e,
		started:  time.Now(),
	}
}

func (s *Service) Store(ctx context.Context, owner, key, content string) (float32, error) {
	hidden := s.embed(key + " " + content)
	if s.om != nil {
		norm, err := s.om.Store(owner, hidden)
		if err != nil { return 0, err }
		// Index for retrieval
		if s.turboOM != nil {
			id := key
			if len(id) > 60 { id = id[:60] }
			s.turboOM.AddVector(owner, id, hidden)
			s.turboOM.Save(owner)
		}
		// Self-model learns from every store
		s.thoughts.Self().LearnDomain(hidden, 0.03)
		return norm, nil
	}
	return 1.0, nil
}

func (s *Service) Recall(ctx context.Context, owner, query string) ([]float32, float32, error) {
	hidden := s.embed(query)
	if s.om != nil { _, deltaO, conf, err := s.om.Recall(owner, hidden); return deltaO, conf, err }
	return nil, 0, nil
}

func (s *Service) StoreHidden(ctx context.Context, owner string, h []float32) (float32, error) {
	if s.om != nil { return s.om.Store(owner, h) }; return 1.0, nil
}

func (s *Service) RecallHidden(ctx context.Context, owner string, q []float32) ([]float32, []float32, float32, error) {
	if s.om != nil { return s.om.Recall(owner, q) }; return nil, nil, 0, nil
}

func (s *Service) Health() (int, float32, int64, int64, string) {
	if s.om != nil {
		stats := s.om.GetStats()
		return s.om.ActiveOwners(), 0, stats.TotalStores, stats.TotalRecalls, time.Since(s.started).String()
	}
	return 0, 0, 0, 0, time.Since(s.started).String()
}

func (s *Service) ResetState(ctx context.Context, owner string) error {
	if s.om != nil { return s.om.ResetOwner(owner) }; return nil
}

func (s *Service) IBNNForward(ctx context.Context, owner, text string) ([]float32, error) {
	hidden := s.embed(text)
	if s.ibnnOM != nil { out, err := s.ibnnOM.ForwardBatch(owner, [][]float32{hidden}); if err != nil { return nil, err }; return out[0], nil }
	return hidden, nil
}

func (s *Service) IBNNForwardHidden(ctx context.Context, owner string, h []float32) ([]float32, error) {
	if s.ibnnOM != nil { out, err := s.ibnnOM.ForwardBatch(owner, [][]float32{h}); if err != nil { return nil, err }; return out[0], nil }
	return h, nil
}

func (s *Service) TurboAdd(ctx context.Context, owner, id string, vec []float32) error {
	if s.turboOM != nil { return s.turboOM.AddVector(owner, id, vec) }
	if s.turbovecCli != nil { return s.turbovecCli.Add(ctx, owner, id, vec) }
	return fmt.Errorf("no turbovec backend")
}

func (s *Service) TurboSearch(ctx context.Context, owner string, query []float32, k int) ([]string, []float32, error) {
	if s.turboOM != nil { return s.turboOM.SearchVector(owner, query, k) }
	if s.turbovecCli != nil { return s.turbovecCli.Search(ctx, owner, query, k) }
	return nil, nil, fmt.Errorf("no turbovec backend")
}

func (s *Service) Generate(ctx context.Context, owner, prompt string) (string, error) {
	if s.gemma == nil { return "", fmt.Errorf("gemma not initialized") }
	return s.gemma.Generate(ctx, prompt)
}

func (s *Service) Think(ctx context.Context, owner string, seeds []string) (*thoughts.Thought, error) {
	return s.thoughts.Think(ctx, owner, seeds)
}

func (s *Service) StartWander(owner string) {
	s.thoughts.StartWander(owner)
}

func (s *Service) StopWander(owner string) {
	s.thoughts.StopWander(owner)
}

func (s *Service) HarvestWander(owner string) []*thoughts.Thought {
	return s.thoughts.HarvestWander(owner)
}

func (s *Service) AddAxiom(statement, domain string) {
	s.thoughts.Truth().AddAxiom(statement, domain)
}

func (s *Service) Adapt(ctx context.Context, owner, wrong, right string) (float32, error) {
	c, err := s.thoughts.Adapt(ctx, owner, wrong, right)
	if err != nil { return 0, err }
	return c.Impact, nil
}

func (s *Service) Learn(ctx context.Context, owner, fact string) error {
	return s.thoughts.Learn(ctx, owner, fact)
}

// SetThoughtsVectorStore overrides the thoughts engine's vector store (e.g. turbogo for production).
func (s *Service) SetThoughtsVectorStore(vs thoughts.VectorStore) {
	s.thoughts = thoughts.New(s.om, s.ibnnOM, vs, s.gemma)
}

// SetNLI attaches an NLI checker to the truth engine for second-opinion contradiction detection.
func (s *Service) SetNLI(checker thoughts.NLIChecker) {
	s.thoughts.Truth().SetNLI(checker)
}

// SaveAll persists all layer state to disk. Call on shutdown or periodically.
func (s *Service) SaveAll(owner, dataDir string) {
	s.thoughts.Self().Save(filepath.Join(dataDir, owner+".self"))
	if s.turboOM != nil { s.turboOM.Save(owner) }
	if s.om != nil { s.om.Save(owner) }
}

func (s *Service) Forget(ctx context.Context, owner, what string) error {
	return s.thoughts.Forget(ctx, owner, what)
}

// embed uses real embeddings if available, falls back to hash.
func (s *Service) embed(text string) []float32 {
	if s.embedder != nil {
		return s.embedder.EmbedText(text)
	}
	hidden := make([]float32, Dimensions)
	seed := uint64(0)
	for _, c := range text { seed = seed*31 + uint64(c) }
	for i := range hidden { seed = seed*6364136223846793005 + 1; hidden[i] = float32(math.Sin(float64(seed)/1e15)) * 0.1 }
	var norm float32
	for _, v := range hidden { norm += v * v }
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 { for i := range hidden { hidden[i] /= norm } }
	return hidden
}
