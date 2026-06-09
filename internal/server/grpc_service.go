package server

import (
	"context"

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
