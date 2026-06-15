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
	"github.com/dbrushchenko/delta-mem-go/internal/session"
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
	sessions    *session.Manager
	started     time.Time
}

func New(deltaOM *deltamem.OwnerManager, ibnnOM *ibnn.OwnerManager, turboOM *turbovec.OwnerManager, gemmaClient *gemma.Client, turbovecClient *turbovec.Client, emb ...*embeddings.Embedder) *Service {
	var e *embeddings.Embedder
	if len(emb) > 0 { e = emb[0] }
	return &Service{
		om: deltaOM, ibnnOM: ibnnOM, turboOM: turboOM, gemma: gemmaClient, turbovecCli: turbovecClient,
		thoughts: thoughts.New(deltaOM, ibnnOM, turboOM, gemmaClient),
		embedder: e,
		sessions: session.NewManager(30 * time.Minute),
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
			if len(id) > 200 { id = id[:200] }
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

// StoreDeep chains ALL layers: embed → δ-mem → turbovec → turbogo → IBNN reinforce → temporal → self-model.
func (s *Service) StoreDeep(ctx context.Context, owner, key, content string) (float32, error) {
	hidden := s.embed(key + " " + content)

	// δ-mem state accumulation
	var norm float32
	if s.om != nil {
		var err error
		norm, err = s.om.Store(owner, hidden)
		if err != nil { return 0, err }
	}

	// turbovec (service layer index)
	id := key
	if len(id) > 200 { id = id[:200] }
	if s.turboOM != nil {
		s.turboOM.AddVector(owner, id, hidden)
		s.turboOM.Save(owner)
	}

	// turbogo (thoughts engine production index) + IBNN reinforce + temporal via Learn
	if s.thoughts != nil {
		s.thoughts.Learn(ctx, owner, content)
	}

	// Self-model domain learning
	s.thoughts.Self().LearnDomain(hidden, 0.03)

	return norm, nil
}

// TurbogoSearch searches the production quantized vector store (turbogo) used by the thoughts engine.
func (s *Service) TurbogoSearch(ctx context.Context, owner string, query []float32, k int) ([]string, []float32, error) {
	if s.thoughts == nil {
		return nil, nil, fmt.Errorf("thoughts engine not initialized")
	}
	// Access turbogo through the thoughts engine's vector store interface
	return s.thoughts.SearchVector(owner, query, k)
}

// Validate runs a statement through the truth engine (axioms + NLI + coherence).
func (s *Service) Validate(ctx context.Context, owner, statement string) (bool, float32, float32, string, []string) {
	if s.thoughts == nil {
		return true, 0, 0, "no truth engine", nil
	}
	v := s.thoughts.Truth().Validate(ctx, statement)
	return v.Valid, v.Grounding, v.Coherence, v.Reason, v.Contradictions
}

// QueryTemporal returns recent temporal events for an owner.
func (s *Service) QueryTemporal(owner string, limit int) []thoughts.Event {
	if s.thoughts == nil { return nil }
	return s.thoughts.Temporal().Recent(owner, limit)
}

// AmIConfident checks the self-model's confidence on a topic.
func (s *Service) AmIConfident(ctx context.Context, owner, text string) (int, float32) {
	if s.thoughts == nil { return 0, 0 }
	hidden := s.embed(text)
	conf := s.thoughts.Self().AmIConfident(hidden)
	// Return raw score approximation based on level
	var raw float32
	switch conf {
	case thoughts.Confident: raw = 1.0
	case thoughts.Uncertain: raw = 0.5
	default: raw = 0.0
	}
	return int(conf), raw
}

// SessionSearch performs a session-aware search: respects budget, dedup, score threshold.
func (s *Service) SessionSearch(ctx context.Context, sessionID, owner string, st session.SessionType, query []float32, k int) ([]string, []float32, error) {
	sess := s.sessions.GetOrCreate(sessionID, owner, st)

	var ids []string
	var scores []float32
	var err error
	if s.turboOM != nil {
		ids, scores, err = s.turboOM.SearchVector(owner, query, k)
	}
	if err != nil { return nil, nil, err }

	// Filter by session policy
	var filteredIDs []string
	var filteredScores []float32
	for i, id := range ids {
		if sess.ShouldInject(id, scores[i]) {
			label := id
			if len(label) > sess.Policy.MaxChars { label = label[:sess.Policy.MaxChars] }
			filteredIDs = append(filteredIDs, label)
			filteredScores = append(filteredScores, scores[i])
			// Estimate token cost (~4 chars per token)
			tokenCost := len(label) / 4
			sess.RecordInjection(id, scores[i], tokenCost)
		}
	}
	return filteredIDs, filteredScores, nil
}

// Sessions returns the session manager for direct access (e.g. from gRPC interceptor).
func (s *Service) Sessions() *session.Manager { return s.sessions }
