# delta-mem-go

Production-grade per-owner memory service with nomic-embed-text-v1.5, δ-mem, IBNN, turbogo, and Gemma 4 QAT.

## Quick Start

```bash
go mod tidy
CGO_ENABLED=1 go build -tags cgo -o delta-mem-go ./cmd/delta-mem-go
./delta-mem-go --port 8080 --grpc-port 9090
```

## Components

- **nomic-embed-text-v1.5** — 768-dim Matryoshka embeddings (ONNX)
- **δ-mem** — gated delta-rule memory
- **IBNN** — Implicit Bias Neural Network
- **turbogo** — pure-Go scalar-quantized ANN index
- **Gemma 4 QAT** — generation via Ollama/llama.cpp

## Deploy

```bash
docker compose up --build -d
# or Kubernetes
kubectl apply -f k8s/
```
