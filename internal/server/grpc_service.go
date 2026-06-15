package server

import (
	"context"

	"github.com/dbrushchenko/delta-mem-go/internal/session"
	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// GRPCService implements the full DeltaMem gRPC service (old + new methods).
type GRPCService struct {
	pb.UnimplementedDeltaMemServer
	Svc *Service
}

// === Original δ-mem methods (unchanged) ===
func (g *GRPCService) Store(ctx context.Context, req *pb.StoreRequest) (*pb.StoreResponse, error) {
	norm, err := g.Svc.Store(ctx, req.Owner, req.Key, req.Content)
	if err != nil {
		return nil, err
	}
	return &pb.StoreResponse{Ok: true, StateNorm: norm}, nil
}

func (g *GRPCService) Recall(ctx context.Context, req *pb.RecallRequest) (*pb.RecallResponse, error) {
	correction, confidence, err := g.Svc.Recall(ctx, req.Owner, req.Query)
	if err != nil {
		return nil, err
	}
	return &pb.RecallResponse{Correction: correction, Confidence: confidence}, nil
}

func (g *GRPCService) StoreHidden(ctx context.Context, req *pb.HiddenStoreRequest) (*pb.StoreResponse, error) {
	norm, err := g.Svc.StoreHidden(ctx, req.Owner, req.HiddenState)
	if err != nil {
		return nil, err
	}
	return &pb.StoreResponse{Ok: true, StateNorm: norm}, nil
}

func (g *GRPCService) RecallHidden(ctx context.Context, req *pb.HiddenRecallRequest) (*pb.HiddenRecallResponse, error) {
	dq, do, conf, err := g.Svc.RecallHidden(ctx, req.Owner, req.QueryState)
	if err != nil {
		return nil, err
	}
	return &pb.HiddenRecallResponse{DeltaQ: dq, DeltaO: do, Confidence: conf}, nil
}

func (g *GRPCService) Health(ctx context.Context, _ *pb.Empty) (*pb.HealthResponse, error) {
	owners, avgNorm, stores, recalls, uptime := g.Svc.Health()
	return &pb.HealthResponse{
		OwnersActive: int32(owners), AvgStateNorm: avgNorm,
		TotalStores: stores, TotalRecalls: recalls, Uptime: uptime,
	}, nil
}

func (g *GRPCService) ResetState(ctx context.Context, req *pb.OwnerRequest) (*pb.Empty, error) {
	return &pb.Empty{}, g.Svc.ResetState(ctx, req.Owner)
}

// === NEW: IBNN methods ===
func (g *GRPCService) IBNNForward(ctx context.Context, req *pb.IBNNForwardRequest) (*pb.IBNNForwardResponse, error) {
	out, err := g.Svc.IBNNForward(ctx, req.Owner, req.Text)
	if err != nil {
		return nil, err
	}
	return &pb.IBNNForwardResponse{Output: out, Dim: int32(len(out))}, nil
}

func (g *GRPCService) IBNNForwardHidden(ctx context.Context, req *pb.IBNNForwardHiddenRequest) (*pb.IBNNForwardResponse, error) {
	out, err := g.Svc.IBNNForwardHidden(ctx, req.Owner, req.HiddenState)
	if err != nil {
		return nil, err
	}
	return &pb.IBNNForwardResponse{Output: out, Dim: int32(len(out))}, nil
}

// === NEW: turbovec methods ===
func (g *GRPCService) TurboAdd(ctx context.Context, req *pb.TurboAddRequest) (*pb.TurboAddResponse, error) {
	if err := g.Svc.TurboAdd(ctx, req.Owner, req.Id, req.Vector); err != nil {
		return nil, err
	}
	return &pb.TurboAddResponse{Ok: true}, nil
}

func (g *GRPCService) TurboSearch(ctx context.Context, req *pb.TurboSearchRequest) (*pb.TurboSearchResponse, error) {
	ids, scores, err := g.Svc.TurboSearch(ctx, req.Owner, req.Query, int(req.K))
	if err != nil {
		return nil, err
	}
	return &pb.TurboSearchResponse{Ids: ids, Scores: scores}, nil
}

// === NEW: Gemma 4 QAT generation ===
func (g *GRPCService) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	resp, err := g.Svc.Generate(ctx, req.Owner, req.Prompt)
	if err != nil {
		return nil, err
	}
	return &pb.GenerateResponse{Response: resp}, nil
}

// === NEW: Thoughts — novel idea synthesis ===
func (g *GRPCService) Think(ctx context.Context, req *pb.ThinkRequest) (*pb.ThinkResponse, error) {
	thought, err := g.Svc.Think(ctx, req.Owner, req.Seeds)
	if err != nil {
		return nil, err
	}
	return &pb.ThinkResponse{
		Idea:       thought.Idea,
		Seeds:      thought.Seeds,
		Neighbors:  thought.Neighbors,
		Confidence: thought.Confidence,
		Novelty:    thought.Novelty,
		Grounding:  thought.Grounding,
		Depth:      int32(thought.Depth),
		Valid:      thought.Valid,
	}, nil
}

func (g *GRPCService) StartWander(ctx context.Context, req *pb.OwnerRequest) (*pb.Empty, error) {
	g.Svc.StartWander(req.Owner)
	return &pb.Empty{}, nil
}

func (g *GRPCService) StopWander(ctx context.Context, req *pb.OwnerRequest) (*pb.Empty, error) {
	g.Svc.StopWander(req.Owner)
	return &pb.Empty{}, nil
}

func (g *GRPCService) HarvestWander(ctx context.Context, req *pb.OwnerRequest) (*pb.HarvestResponse, error) {
	raw := g.Svc.HarvestWander(req.Owner)
	resp := &pb.HarvestResponse{}
	for _, t := range raw {
		resp.Thoughts = append(resp.Thoughts, &pb.ThinkResponse{
			Idea: t.Idea, Seeds: t.Seeds, Neighbors: t.Neighbors,
			Confidence: t.Confidence, Novelty: t.Novelty,
			Grounding: t.Grounding, Depth: int32(t.Depth), Valid: t.Valid,
		})
	}
	return resp, nil
}

func (g *GRPCService) AddAxiom(ctx context.Context, req *pb.AxiomRequest) (*pb.Empty, error) {
	g.Svc.AddAxiom(req.Statement, req.Domain)
	return &pb.Empty{}, nil
}

func (g *GRPCService) Adapt(ctx context.Context, req *pb.AdaptRequest) (*pb.AdaptResponse, error) {
	impact, err := g.Svc.Adapt(ctx, req.Owner, req.Wrong, req.Right)
	if err != nil { return nil, err }
	return &pb.AdaptResponse{Impact: impact}, nil
}

func (g *GRPCService) Learn(ctx context.Context, req *pb.LearnRequest) (*pb.Empty, error) {
	return &pb.Empty{}, g.Svc.Learn(ctx, req.Owner, req.Fact)
}

func (g *GRPCService) Forget(ctx context.Context, req *pb.ForgetRequest) (*pb.Empty, error) {
	return &pb.Empty{}, g.Svc.Forget(ctx, req.Owner, req.What)
}

// === Full-pipeline RPCs ===

func (g *GRPCService) StoreDeep(ctx context.Context, req *pb.StoreRequest) (*pb.StoreDeepResponse, error) {
	norm, err := g.Svc.StoreDeep(ctx, req.Owner, req.Key, req.Content)
	if err != nil { return nil, err }
	id := req.Key
	if len(id) > 60 { id = id[:60] }
	return &pb.StoreDeepResponse{
		Ok: true, StateNorm: norm,
		IbnnReinforced: 0.03,
		TurbogoId: id, TurbovecId: id,
	}, nil
}

func (g *GRPCService) TurbogoSearch(ctx context.Context, req *pb.TurboSearchRequest) (*pb.TurboSearchResponse, error) {
	ids, scores, err := g.Svc.TurbogoSearch(ctx, req.Owner, req.Query, int(req.K))
	if err != nil { return nil, err }
	return &pb.TurboSearchResponse{Ids: ids, Scores: scores}, nil
}

func (g *GRPCService) Validate(ctx context.Context, req *pb.ValidateRequest) (*pb.ValidateResponse, error) {
	valid, grounding, coherence, reason, contradictions := g.Svc.Validate(ctx, req.Owner, req.Statement)
	return &pb.ValidateResponse{
		Valid: valid, Grounding: grounding, Coherence: coherence,
		Reason: reason, Contradictions: contradictions,
	}, nil
}

func (g *GRPCService) QueryTemporal(ctx context.Context, req *pb.TemporalRequest) (*pb.TemporalResponse, error) {
	events := g.Svc.QueryTemporal(req.Owner, int(req.Limit))
	resp := &pb.TemporalResponse{}
	for _, e := range events {
		resp.Events = append(resp.Events, &pb.TemporalEvent{
			Id: e.ID, Content: e.Content, When: e.When.Format("2006-01-02T15:04:05Z"),
		})
	}
	return resp, nil
}

func (g *GRPCService) AmIConfident(ctx context.Context, req *pb.ConfidenceRequest) (*pb.ConfidenceResponse, error) {
	level, raw := g.Svc.AmIConfident(ctx, req.Owner, req.Text)
	return &pb.ConfidenceResponse{Level: int32(level), RawScore: raw}, nil
}

func (g *GRPCService) SessionSearch(ctx context.Context, req *pb.SessionSearchRequest) (*pb.TurboSearchResponse, error) {
	st := parseSessionType(req.SessionType)
	ids, scores, err := g.Svc.SessionSearch(ctx, req.SessionId, req.Owner, st, req.Query, int(req.K))
	if err != nil { return nil, err }
	return &pb.TurboSearchResponse{Ids: ids, Scores: scores}, nil
}

func parseSessionType(s string) session.SessionType {
	switch s {
	case "hook-light": return session.HookLight
	case "hook-standard": return session.HookStandard
	case "hook-deep": return session.HookDeep
	case "agent-mcp": return session.AgentMCP
	case "think": return session.ThinkSession
	case "cli": return session.CLISession
	default: return session.HookStandard
	}
}
