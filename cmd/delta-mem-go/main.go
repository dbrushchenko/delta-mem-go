package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"github.com/dbrushchenko/delta-mem-go/internal/auth"
	"github.com/dbrushchenko/delta-mem-go/internal/config"
	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/gemma"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/logger"
	mcpkg "github.com/dbrushchenko/delta-mem-go/internal/mcp"
	"github.com/dbrushchenko/delta-mem-go/internal/metrics"
	"github.com/dbrushchenko/delta-mem-go/internal/nli"
	"github.com/dbrushchenko/delta-mem-go/internal/server"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbogo"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
	pb "github.com/dbrushchenko/delta-mem-go/proto"
)

func main() {
	cfg := config.Load()
	l := logger.New(cfg.LogLevel)

	os.MkdirAll(cfg.DataDir, 0755)

	// Real embeddings (ONNX in-process)
	var emb *embeddings.Embedder
	if cfg.ModelPath != "" || os.Getenv("EMBEDDER_URL") != "" {
		url := cfg.ModelPath
		if v := os.Getenv("EMBEDDER_URL"); v != "" {
			url = v
		}
		var err error
		emb, err = embeddings.Get(url, "")
		if err != nil {
			l.Warn("embeddings disabled", slog.String("err", err.Error()))
		} else {
			emb.SetTargetDim(cfg.EmbedDim)
			thoughts.SetEmbedder(emb)
			l.Info("embeddings configured", slog.String("url", url), slog.Int("dim", cfg.EmbedDim))
		}
	}

	// δ-mem
	deltaCfg := deltamem.Config{R: 64, HiddenDim: cfg.EmbedDim, NormCap: 10.0}
	deltaOM := deltamem.NewOwnerManager(deltaCfg, cfg.DataDir)

	// IBNN
	ibnnCfg := ibnn.Config{InputDim: cfg.EmbedDim, HiddenDim: cfg.EmbedDim}
	ibnnOM := ibnn.NewOwnerManager(ibnnCfg, cfg.DataDir)

	// turbogo (quantized ANN — production vector store for thoughts engine)
	turbogoOM := turbogo.NewOwnerManager(turbogo.Config{Dim: cfg.EmbedDim, BitWidth: 4}, cfg.DataDir)

	// turbovec (simple in-memory — for service API endpoints)
	turboOM := turbovec.NewOwnerManager(cfg.EmbedDim, cfg.DataDir)

	// Gemma client (optional)
	var gemmaClient *gemma.Client
	if cfg.GemmaModelPath != "" || cfg.GemmaModel != "" {
		gemmaClient = gemma.NewClient(cfg.GemmaModel)
		l.Info("gemma enabled", slog.String("model", cfg.GemmaModel))
	}

	// turbovec HTTP sidecar client (optional, fallback)
	turbovecCli := turbovec.NewClient(cfg.TurbovecURL)

	svc := server.New(deltaOM, ibnnOM, turboOM, gemmaClient, turbovecCli, emb)
	svc.SetThoughtsVectorStore(turbogoOM) // thoughts engine uses production quantized index

	// NLI model (optional — supplements truth detection)
	nliModelPath := filepath.Join(filepath.Dir(cfg.ModelPath), "nli-deberta.onnx")
	nliTokPath := filepath.Join(filepath.Dir(cfg.ModelPath), "nli-tokenizer.json")
	if _, err := os.Stat(nliModelPath); err == nil {
		nliChecker, err := nli.New(nliModelPath, nliTokPath)
		if err != nil {
			l.Warn("NLI disabled", slog.String("err", err.Error()))
		} else {
			svc.SetNLI(nliChecker)
			l.Info("NLI enabled (DeBERTa)")
		}
	}

	// HTTP
	mux := http.NewServeMux()
	metrics.RegisterMetricsHandler(mux)
	svc.RegisterHTTP(mux)

	// MCP endpoint — exposes all tools via streamable-http
	mcpHandler := mcpkg.New()
	mcpkg.RegisterTools(mcpHandler, svc)
	mux.Handle("POST /mcp", mcpHandler)
	l.Info("MCP registered", slog.Int("tools", 14))

	handler := metrics.Middleware(auth.HTTPMiddleware(mux, cfg.APIKeys))
	httpAddr := fmt.Sprintf(":%d", cfg.HTTPPort)
	httpServer := &http.Server{Addr: httpAddr, Handler: handler}

	// gRPC
	grpcServer := grpc.NewServer(grpc.ChainUnaryInterceptor(auth.GRPCUnaryInterceptor(cfg.APIKeys)))
	pb.RegisterDeltaMemServer(grpcServer, &server.GRPCService{Svc: svc})
	reflection.Register(grpcServer)

	grpcAddr := fmt.Sprintf(":%d", cfg.GRPCPort)
	lis, err := net.Listen("tcp", grpcAddr)
	if err != nil { l.Error("listen failed", slog.String("err", err.Error())); os.Exit(1) }

	go func() { l.Info("HTTP", slog.String("addr", httpAddr)); httpServer.ListenAndServe() }()
	go func() { l.Info("gRPC", slog.String("addr", grpcAddr)); grpcServer.Serve(lis) }()

	l.Info("δ-mem-go full stack ready", slog.Int("dim", cfg.EmbedDim))

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	l.Info("shutting down")
	// Save state for all active owners
	// (owners are auto-discovered from δ-mem OwnerManager)
	grpcServer.GracefulStop()
	httpServer.Shutdown(context.Background())
}
