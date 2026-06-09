package server

import (
	"context"

	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

// grpcService implements the full DeltaMem gRPC service (old + new methods).
type grpcService struct {
	pb.UnimplementedDeltaMemServer
	svc *Service
}

// === Original δ-mem methods (unchanged) ===
func (g *grpcService) Store(ctx context.Context, req *pb.StoreRequest) (*pb.StoreResponse, error) {
	norm, err := g.svc.Store(ctx, req.Owner, req.Key, req.Content)
	if err != nil {
		return nil, err
	}
	return &pb.StoreResponse{Ok: true, StateNorm: norm}, nil
}

func (g *grpcService) Recall(ctx context.Context, req *pb.RecallRequest) (*pb.RecallResponse, error) {
	correction, confidence, err := g.svc.Recall(ctx, req.Owner, req.Query)
	if err != nil {
		return nil, err
	}
	return &pb.RecallResponse{Correction: correction, Confidence: confidence}, nil
}

func (g *grpcService) StoreHidden(ctx context.Context, req *pb.HiddenStoreRequest) (*pb.StoreResponse, error) {
	norm, err := g.svc.StoreHidden(ctx, req.Owner, req.HiddenState)
	if err != nil {
		return nil, err
	}
	return &pb.StoreResponse{Ok: true, StateNorm: norm}, nil
}

func (g *grpcService) RecallHidden(ctx context.Context, req *pb.HiddenRecallRequest) (*pb.HiddenRecallResponse, error) {
	dq, do, conf, err := g.svc.RecallHidden(ctx, req.Owner, req.QueryState)
	if err != nil {
		return nil, err
	}
	return &pb.HiddenRecallResponse{DeltaQ: dq, DeltaO: do, Confidence: conf}, nil
}

func (g *grpcService) Health(ctx context.Context, _ *pb.Empty) (*pb.HealthResponse, error) {
	owners, avgNorm, stores, recalls, uptime := g.svc.Health()
	return &pb.HealthResponse{
		OwnersActive: int32(owners), AvgStateNorm: avgNorm,
		TotalStores: stores, TotalRecalls: recalls, Uptime: uptime,
	}, nil
}

func (g *grpcService) ResetState(ctx context.Context, req *pb.OwnerRequest) (*pb.Empty, error) {
	return &pb.Empty{}, g.svc.ResetState(ctx, req.Owner)
}

// === NEW: IBNN methods ===
func (g *grpcService) IBNNForward(ctx context.Context, req *pb.IBNNForwardRequest) (*pb.IBNNForwardResponse, error) {
	out, err := g.svc.IBNNForward(ctx, req.Owner, req.Text)
	if err != nil {
		return nil, err
	}
	return &pb.IBNNForwardResponse{Output: out, Dim: int32(len(out))}, nil
}

func (g *grpcService) IBNNForwardHidden(ctx context.Context, req *pb.IBNNForwardHiddenRequest) (*pb.IBNNForwardResponse, error) {
	out, err := g.svc.IBNNForwardHidden(ctx, req.Owner, req.HiddenState)
	if err != nil {
		return nil, err
	}
	return &pb.IBNNForwardResponse{Output: out, Dim: int32(len(out))}, nil
}

// === NEW: turbovec methods ===
func (g *grpcService) TurboAdd(ctx context.Context, req *pb.TurboAddRequest) (*pb.TurboAddResponse, error) {
	if err := g.svc.TurboAdd(ctx, req.Owner, req.Id, req.Vector); err != nil {
		return nil, err
	}
	return &pb.TurboAddResponse{Ok: true}, nil
}

func (g *grpcService) TurboSearch(ctx context.Context, req *pb.TurboSearchRequest) (*pb.TurboSearchResponse, error) {
	ids, scores, err := g.svc.TurboSearch(ctx, req.Owner, req.Query, int(req.K))
	if err != nil {
		return nil, err
	}
	return &pb.TurboSearchResponse{Ids: ids, Scores: scores}, nil
}

// === NEW: Gemma 4 QAT generation ===
func (g *grpcService) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	resp, err := g.svc.Generate(ctx, req.Owner, req.Prompt)
	if err != nil {
		return nil, err
	}
	return &pb.GenerateResponse{Response: resp}, nil
}
