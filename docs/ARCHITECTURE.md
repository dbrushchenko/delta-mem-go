# Architecture

## 12-Layer Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         δ-mem-go Cognitive Stack                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT (text)                                                           │
│    │                                                                    │
│    ▼                                                                    │
│  ┌─────────────────┐                                                    │
│  │ 1. EMBEDDINGS   │  nomic-embed-text-v1.5 (ONNX, 768-dim)           │
│  │    (Layer 1)    │  Matryoshka: configurable 64–768 dims             │
│  └────────┬────────┘                                                    │
│           │ float32[768]                                                │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 2. δ-MEM        │  Gated delta-rule, R=64 rank                      │
│  │    (Layer 2)    │  Store: outer-product accumulation                 │
│  │                 │  Recall: sparse top-k correction signal            │
│  │                 │  Adaptive rank expansion when saturated            │
│  └────────┬────────┘                                                    │
│           │ correction vector + confidence                              │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 3. IBNN         │  Lateral inhibition sharpening                    │
│  │    (Layer 3)    │  Reinforced +0.001 on truth pass                  │
│  │                 │  Weakened -0.001 on truth fail                     │
│  └────────┬────────┘                                                    │
│           │ crystallized activation                                     │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 4. TURBOGO      │  4-bit quantized ANN index                        │
│  │    (Layer 4)    │  Rotation → quantization → packed search          │
│  │                 │  k=3 normal, k=8 on surprise                      │
│  └────────┬────────┘                                                    │
│           │ neighbor IDs + scores                                       │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 5. TRUTH ENGINE │  Heuristic axiom check (registered axioms)        │
│  │    (Layer 5)    │  + NLI DeBERTa-v3-xsmall contradiction detection  │
│  │                 │  Validates every generated thought                 │
│  └────────┬────────┘                                                    │
│           │ valid/invalid + grounding score + reason                    │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 6. SELF-MODEL   │  Domain confidence map (per-owner)                │
│  │    (Layer 6)    │  Dynamic surprise thresholds                      │
│  │                 │  Tracks store/recall/think activity                │
│  └────────┬────────┘                                                    │
│           │ threshold adjustments                                       │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 7. TEMPORAL     │  Event sequencing per owner                       │
│  │    (Layer 7)    │  Records: thought IDs, wander events, adaptations │
│  │                 │  Feeds recent history into synthesis               │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 8. ADAPTATION   │  Replace-not-remove correction                    │
│  │    (Layer 8)    │  Wrong embedding suppressed, right strengthened   │
│  │                 │  δ-mem state updated in-place                     │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 9. SYNTHESIS    │  Gap-finding: centroid of neighbors →             │
│  │    (Layer 9)    │  search for what centroid points to but           │
│  │                 │  isn't stored → the gap IS the insight            │
│  │                 │  IBNN re-ranks neighbors by activation alignment  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 10. VERIFIER    │  Self-consistency: compare iteration N vs N-1     │
│  │    (Layer 10)   │  Convergence threshold: cosine > 0.95 → stop     │
│  │                 │  External verifier hook for reality grounding     │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 11. WANDER      │  Opportunistic residual discovery                 │
│  │    (Layer 11)   │  Residual = raw_probe − δ_mem_output             │
│  │                 │  Search residual for "nearby but not recalled"    │
│  │                 │  Salience threshold: score > 0.6                  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 12. ITERATIVE   │  Surprise-gated depth (max 5 iterations)         │
│  │    RE-ENTRY     │  Low confidence → go deeper (more retrieval)     │
│  │    (Layer 12)   │  Previous thought → new seed → re-enter loop     │
│  └─────────────────┘                                                    │
│                                                                         │
│  OUTPUT: Thought {idea, seeds, neighbors, confidence, novelty,          │
│          grounding, depth, valid}                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Think Operation

```
seeds[]
  │
  ├── parallel embed ──→ float32[768] per seed
  │
  ├── δ-mem Store (accumulate outer products)
  │
  ├── δ-mem SparseRecall (top-k rows, k=R/2)
  │       └── inline updateProjections(lr=0.001) ← SELF-TRAINING
  │
  ├── IBNN crystallize (lateral inhibition on δ output)
  │
  ├── TurboGo ANN search (raw embeddings, k=3 or k=8)
  │
  ├── Gemma articulation OR substrate synthesis
  │       ├── With Gemma: prompt = seeds + neighbors + top activations
  │       └── Without Gemma: IBNN-weighted centroid → gap search → extractive output
  │
  ├── Truth validation (axioms + NLI)
  │       ├── valid → IBNN reinforce +0.001
  │       └── invalid → IBNN weaken -0.001, retry with "Avoid: reason"
  │
  ├── Verifier (optional external reality check)
  │       └── invalid → Adapt(wrong, correction), retry
  │
  ├── Convergence check (cosine > 0.95 with previous iteration → done)
  │
  ├── Surprise gate (confidence >= dynamic threshold && depth > 0 → done)
  │
  ├── Wander: residual search for adjacent insights
  │
  └── Store thought back into δ-mem + TurboGo + Temporal
```

## Self-Training Loop

The system trains itself on every interaction without external supervision:

```
┌──────────────────────────────────────────────────────────┐
│                   Continuous Learning                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Every Store:                                            │
│    • δ-mem accumulates outer products                    │
│    • Self-model learns domain confidence                 │
│    • TurboGo indexes for future retrieval                │
│                                                          │
│  Every Recall:                                           │
│    • updateProjections(lr=0.001)                         │
│      Adjusts δ-mem Q/K projections toward better recall  │
│    • Self-model updates surprise thresholds              │
│                                                          │
│  Every Think:                                            │
│    • Truth pass → IBNN reinforce(+0.001)                 │
│    • Truth fail → IBNN weaken(-0.001)                    │
│    • Generated thought stored back (self-seeding)        │
│    • Temporal records event sequence                     │
│                                                          │
│  Every Adapt:                                            │
│    • Wrong embedding suppressed in δ-mem                 │
│    • Right embedding strengthened                        │
│    • No catastrophic forgetting (replace-not-remove)     │
│                                                          │
│  Initiation (first-time):                                │
│    • 5 epochs over training corpus                       │
│    • lr=0.01 (10x normal for rapid bootstrap)            │
│    • ~45s for 90KB corpus                                │
│    • Builds initial δ-mem state + IBNN weights           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Persistence Model

All state is persisted to the `data/` directory, partitioned by owner:

| File Pattern | Layer | Contents |
|---|---|---|
| `{owner}.state` | δ-mem | Gated delta-rule weight matrices (R×D float32) |
| `{owner}.ibnn.state` | IBNN | Lateral inhibition weights (~3.4 MB per owner) |
| `{owner}.self` or `self_model.gob` | Self-Model | Domain confidence map, activity counters |
| `{owner}.turbo` | TurboGo | 4-bit quantized vectors + ID index |

### State Lifecycle

```
Start → Load existing .state/.ibnn.state from disk (if present)
      → Otherwise create fresh with default config
      → Process requests (state mutates in memory)
      → Save on: explicit save, graceful shutdown, periodic checkpoint
```

### Disk Layout Example

```
data/
├── states/
│   ├── dabrush.state          # δ-mem weights
│   ├── dabrush.ibnn.state     # IBNN weights
│   └── dabrush.self           # Self-model
├── training_data/
│   ├── training_3.txt         # Raw training corpus
│   └── training_2.jsonl       # Structured training pairs
└── initiated/
    └── dabrush.state          # Post-initiation snapshot
```

## Component Interactions

```
┌────────────┐     ┌─────────────────┐     ┌──────────────┐
│  HTTP API  │────▶│     Service     │────▶│   Thoughts   │
│  :8080     │     │                 │     │    Engine     │
└────────────┘     └─────────────────┘     └──────┬───────┘
                                                   │
┌────────────┐            │                        │
│  gRPC API  │────────────┘               ┌────────┴────────┐
│  :9090     │                            │                  │
└────────────┘                     ┌──────┴─────┐    ┌──────┴──────┐
                                   │  δ-mem OM  │    │   IBNN OM   │
┌────────────┐                     │ (per-owner)│    │ (per-owner) │
│  mem-cli   │──gRPC──────────┐    └────────────┘    └─────────────┘
│            │                │           │
└────────────┘                │    ┌──────┴──────┐    ┌─────────────┐
                              │    │  TurboGo OM │    │ Truth Engine│
┌────────────┐                │    │ (per-owner) │    │ + NLI Model │
│ kiro hook  │──gRPC/HTTP─────┘    └─────────────┘    └─────────────┘
└────────────┘
```

## Kubernetes Deployment

```yaml
# Single pod, Istio mesh, PVC-backed storage
Namespace: mesh
Image: code.usgs.gov:5050/daniel_brushchenko/mem-go:latest
Resources: 500m–2 CPU, 1–4Gi RAM
Probes: /health on :8080
Volumes:
  - delta-mem-data (PVC) → /data
  - delta-mem-models (PVC) → /models
Annotations:
  - sidecar.istio.io/inject: "true"
  - prometheus.io/scrape: "true"
```

## Protocol Interfaces

Single binary serves three protocol paths to the same service layer:

```
delta-mem-go.exe
  ├── :18080       HTTP REST API (/store, /recall, /health, /metrics, /enroll)
  ├── :18080/mcp   MCP (JSON-RPC 2.0, 14 tools, streamable-http)
  └── :19090       gRPC (protobuf, 24 RPCs)
```

All three share the same in-process `*server.Service` — same δ-mem state, IBNN weights,
turbovec/turbogo indexes. A Store via gRPC and dmem_store via MCP modify the same matrices.

### Protocol Selection Guide

| Use case | Protocol | Why |
|----------|----------|-----|
| Kiro CLI hooks (Go binaries) | gRPC :19090 | Fastest (~5ms), compiled client |
| Agent on-demand tools (MCP) | HTTP :18080/mcp | Standard agent tool protocol |
| Manual testing / scripts | HTTP REST :18080 | curl-friendly |
| Mesh-to-mesh (pod-to-pod) | gRPC :19090 | Protobuf efficiency |
| Remote agent (no local install) | MCP URL | `"url": "https://dmem.mesh.gs.doi.net/mcp"` |

### Authentication

Both HTTP and gRPC use service-level API key (not per-owner):
- HTTP: `X-API-Key` header
- gRPC: `x-api-key` metadata
- MCP: Same `X-API-Key` (goes through HTTP middleware)
- No keys configured → passthrough (localhost dev mode)

### New RPCs (2026-06-14)

| RPC | Layers | Purpose |
|-----|--------|---------|
| `StoreDeep` | All (Store + Learn) | Single-call full-pipeline storage |
| `TurbogoSearch` | turbogo (4) | Direct production index query |
| `Validate` | Truth (10) + NLI (7) | Statement truth-checking |
| `QueryTemporal` | Temporal (8) | Recent event history |
| `AmIConfident` | Self-model (9) | Topic confidence check |
