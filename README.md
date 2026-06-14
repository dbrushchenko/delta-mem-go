# δ-mem-go

Persistent memory and thought synthesis engine for AI agents. A 12-layer cognitive architecture that stores knowledge, detects truth, generates novel ideas, and self-trains on every interaction.

Written in Go. Single binary. Runs as a Windows service or Kubernetes pod.

## Key Capabilities

- **Store** — Encode facts into a gated delta-rule memory substrate (20ms)
- **Recall** — Retrieve and interference-correct against stored knowledge
- **Think** — Synthesize novel ideas through iterative re-entry across layers (1–4s)
- **Adapt** — Correct misconceptions without catastrophic forgetting (150ms)
- **Learn** — Absorb new facts and update domain confidence
- **Self-train** — Every recall updates projection weights (lr=0.001); IBNN reinforces/weakens on truth pass/fail

## Quick Start (Windows)

### Option A: Installer (recommended)

```powershell
# Download or build the installer
.\delta-mem-go-setup.exe
```

The installer auto-detects admin status:
- **Administrator** → System-wide NSSM service, auto-start on boot
- **Standard user** → `%APPDATA%\mem-go`, startup shortcut, no admin required

### Option B: Manual

```powershell
# 1. Build
set CGO_ENABLED=1
go build -o delta-mem-go.exe ./cmd/delta-mem-go

# 2. Download models to .\models\
#    - nomic-embed-text-v1.5.onnx + .onnx.data (549 MB)
#    - onnxruntime.dll (v1.26.0 win-x64)
#    - nli-deberta.onnx + nli-tokenizer.json (284 MB, optional)

# 3. Run
.\delta-mem-go.exe --model .\models\nomic-embed-text-v1.5.onnx --port 18080 --grpc-port 19090 --data .\data

# 4. Test
curl http://localhost:18080/health
```

## Quick Start (Docker)

```bash
# Full stack: delta-mem + ollama (Gemma 4) + turbovec sidecar
docker compose up --build -d

# Verify
curl http://localhost:8080/health
```

Models must be mounted at `/models`:
```
models/
├── nomic-embed-text-v1.5.onnx
├── nomic-embed-text-v1.5.onnx.data
├── nli-deberta.onnx
├── nli-tokenizer.json
└── tokenizer.json
```

## CLI Client

```powershell
# Build
go build -o mem-cli.exe ./cmd/mem-cli

# Core
mem-cli store --key "go-concurrency" --content "goroutines are M:N scheduled"
mem-cli store-deep --key "k8s-mesh" --content "compute mesh uses Flagger canary"
mem-cli recall "concurrency model in Go"
mem-cli think "distributed systems" "eventual consistency"
mem-cli adapt "Python is compiled" "Python is interpreted"
mem-cli learn "Kubernetes uses etcd for state storage"
mem-cli forget "outdated fact"
mem-cli health

# New (full-pipeline)
mem-cli search-deep "data collection kubernetes"
mem-cli validate "LoggerNet runs on Kubernetes pods"
mem-cli temporal 20
mem-cli confident "LoRaWAN IoT sensors"
```

## API Reference

### HTTP API (default :8080 / Windows :18080)

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/store` | POST | `{"owner","key","content"}` | `{"ok":true,"state_norm":float}` |
| `/recall` | POST | `{"owner","query"}` | `{"confidence":float,"correction_dim":int}` |
| `/health` | GET | — | `{"owners_active","avg_state_norm","total_stores","total_recalls","uptime"}` |
| `/ibnn-forward` | POST | `{"owner","text"}` | `{"output":[float],"dim":int}` |
| `/turbo-add` | POST | `{"owner","id","vector"}` | `{"ok":true}` |
| `/turbo-search` | POST | `{"owner","query","k"}` | `{"ids":[],"scores":[]}` |
| `/generate` | POST | `{"owner","prompt"}` | `{"response":string}` |
| `/metrics` | GET | — | Prometheus format |

Authentication: `X-API-Key` header (configured via `API_KEYS` env var, comma-separated).

### gRPC API (default :9090 / Windows :19090)

Proto: `proto/deltamem.proto`

```protobuf
service DeltaMem {
  // Core
  rpc Store(StoreRequest) returns (StoreResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
  rpc Think(ThinkRequest) returns (ThinkResponse);
  rpc Adapt(AdaptRequest) returns (AdaptResponse);
  rpc Learn(LearnRequest) returns (Empty);
  rpc Forget(ForgetRequest) returns (Empty);
  rpc Health(Empty) returns (HealthResponse);

  // Vector stores
  rpc IBNNForward(IBNNForwardRequest) returns (IBNNForwardResponse);
  rpc IBNNForwardHidden(IBNNForwardHiddenRequest) returns (IBNNForwardResponse);
  rpc TurboAdd(TurboAddRequest) returns (TurboAddResponse);
  rpc TurboSearch(TurboSearchRequest) returns (TurboSearchResponse);
  rpc Generate(GenerateRequest) returns (GenerateResponse);

  // Wander
  rpc StartWander(OwnerRequest) returns (Empty);
  rpc StopWander(OwnerRequest) returns (Empty);
  rpc HarvestWander(OwnerRequest) returns (HarvestResponse);
  rpc AddAxiom(AxiomRequest) returns (Empty);

  // Full-pipeline (added 2026-06-14)
  rpc StoreDeep(StoreRequest) returns (StoreDeepResponse);
  rpc TurbogoSearch(TurboSearchRequest) returns (TurboSearchResponse);
  rpc Validate(ValidateRequest) returns (ValidateResponse);
  rpc QueryTemporal(TemporalRequest) returns (TemporalResponse);
  rpc AmIConfident(ConfidenceRequest) returns (ConfidenceResponse);
}
```

gRPC auth: `x-api-key` metadata header.

### MCP API (POST /mcp on HTTP port)

JSON-RPC 2.0 over HTTP POST. Same port as REST API. Protected by same `X-API-Key`.

Agent config (mesh):
```json
{ "dmem": { "url": "https://dmem.mesh.gs.doi.net/mcp" } }
```

Agent config (localhost, no auth):
```json
{ "dmem": { "url": "http://localhost:18080/mcp" } }
```

14 tools exposed:

| Tool | Description |
|------|-------------|
| `dmem_store` | Store fact (embed → δ-mem → turbovec → self-model) |
| `dmem_store_deep` | Store ALL layers (Store + Learn in one call) |
| `dmem_recall` | Query δ-mem confidence (triggers self-training) |
| `dmem_learn` | Absorb deeply (thoughts engine path) |
| `dmem_think` | Full 12-layer synthesis (Gemma + Wander included) |
| `dmem_adapt` | Correct misconception (replace-not-remove) |
| `dmem_forget` | Decay a memory |
| `dmem_validate` | Truth engine check (axioms + NLI) |
| `dmem_search` | Vector search (turbovec or turbogo) |
| `dmem_confident` | Self-model confidence check |
| `dmem_temporal` | Query recent events |
| `dmem_wander` | Start/stop/harvest spontaneous thoughts |
| `dmem_axiom` | Set immutable truth |
| `dmem_health` | Server status |

## Architecture Summary

12 processing layers orchestrated by the thoughts engine:

1. **Embeddings** — nomic-embed-text-v1.5, 768-dim, ONNX in-process
2. **δ-mem** — Gated delta-rule memory (R=64 rank, adaptive expansion)
3. **IBNN** — Lateral inhibition sharpens dominant patterns
4. **TurboGo** — 4-bit quantized ANN for fast neighbor retrieval
5. **Truth Engine** — Heuristic axiom validation + NLI DeBERTa contradiction detection
6. **Self-Model** — Domain confidence tracking, dynamic thresholds
7. **Temporal** — Event sequencing and causal ordering
8. **Adaptation** — Replace-not-remove error correction
9. **Synthesis** — Substrate gap-finding (centroid → gap → insight)
10. **Verifier** — Self-consistency checking across iterations
11. **Wander** — Opportunistic residual-based adjacent discovery
12. **Iterative Re-entry** — Surprise-gated depth control (max 5 iterations)

## Configuration

Flags:
```
--model         Path to nomic ONNX model
--port          HTTP port (default: 8080)
--grpc-port     gRPC port (default: 9090)
--data          State directory (default: ./data/states)
--embed-dim     Matryoshka dimension, 64–768 (default: 768)
--rate-limit    Requests/min per owner (default: 1000)
--log-level     debug|info|warn|error (default: info)
```

Environment:
```
API_KEYS=key1,key2        Comma-separated valid API keys
DATA_DIR=/data/states     Override --data
GEMMA_URL=http://...      Ollama endpoint for Gemma 4
ORT_LIB_DIR=/usr/local/lib  ONNX Runtime library path
```

## Project Structure

```
cmd/
  delta-mem-go/    Server binary (HTTP + gRPC + MCP)
  mem-cli/         gRPC CLI client
internal/
  auth/            API key middleware (HTTP + gRPC)
  config/          Flag + env configuration
  deltamem/        δ-mem gated delta-rule module (MultiRes: hot/warm/cold)
  embeddings/      ONNX nomic embedder
  ibnn/            Inhibition-Based Neural Network
  mcp/             MCP streamable-http handler (JSON-RPC 2.0)
  nli/             DeBERTa NLI contradiction checker
  server/          HTTP + gRPC service layer
  state/           Persistence backends (File, Redis, PostgreSQL, Hybrid)
  thoughts/        12-layer synthesis engine
  turbogo/         4-bit quantized ANN index (production)
  turbovec/        Simple vector store + HTTP sidecar (service API)
  gemma/           Ollama Gemma client
  metrics/         Prometheus instrumentation
  observability/   OpenTelemetry tracing
installer/         Windows installer (NSSM + per-user)
proto/             Protobuf definitions
scripts/delivery/  Kiro CLI hooks (Go gRPC binaries)
k8s/               Kubernetes manifests (Istio mesh)
models/            ONNX model files (gitignored)
data/              Persistent state (gitignored)
```

## Deployment

- **Windows** — Single `.exe`, NSSM service or startup shortcut
- **Docker** — `docker compose up --build`
- **Kubernetes** — `k8s/deployment-full.yaml` (Istio sidecar, Prometheus scrape, PVC-backed)

## License

Internal USGS project. Not for public distribution.
