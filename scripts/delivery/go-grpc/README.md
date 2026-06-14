# δ-mem-go gRPC Hooks for Kiro CLI

Compiled Go binaries that integrate Kiro CLI agents with δ-mem-go via gRPC (port 19090).
Zero dependencies, ~5ms startup, proper protobuf serialization.

## Server Architecture (main.go)

The running δ-mem-go server initializes 12 active layers:

### Core Computation Layers
```
 1. Embeddings    nomic-embed-text-v1.5 (ONNX, 768-dim, in-process)
 2. δ-mem         Accumulative state matrix (R=64, NormCap=10.0, adaptive rank)
 3. IBNN          Inference-Based Neural Network (768→768, sparse, reinforcing)
 4. turbogo       Quantized 4-bit ANN (production vector store, thoughts engine)
 5. turbovec      Simple in-memory ANN (service API endpoints)
 6. Gemma         LLM for articulation (gemma-4-e4b-it-q4, optional)
 7. NLI           DeBERTa contradiction detection (truth engine)
```

### Thoughts Engine Sub-Layers
```
 8. Temporal      Event timeline, recency tracking, wander history
 9. Self-model    Per-topic confidence, uncertainty, domain awareness
10. Truth         Axiom storage, heuristic + NLI validation
11. Verifier      Self-consistency checking against stored knowledge
12. Wander        Spontaneous thought from δ-mem residuals (opportunistic)
```

### State Backend (Persistence)
```
13. State         Underlies all layers — controls how state is saved/loaded
    ├── FileBackend   Local gob files (Windows desktop, current)
    ├── RedisBackend  Fast access, 24h TTL (mesh deployment)
    ├── PGBackend     PostgreSQL durable store (long-term)
    └── HybridBackend Redis + PG (production: write-both, read-Redis-first)
```

Configured via `--backend file|redis` + `--redis-url` + `--pg-url`.
Current localhost runs FileBackend → `%APPDATA%\mem-go\data\`.
Mesh target: HybridBackend (Redis speed + PostgreSQL durability).

### Vector Store Roles
- **turbovec** → service API layer (`Store`, `TurboSearch` RPCs) — hooks interact here
- **turbogo** → thoughts engine (`Think`, `Learn`, `Wander`) — deeper operations, quantized

## Hook Tiers

Three tiers of hooks, increasing depth and latency:

| Tier | Store | Enrich | Layers (Store) | Layers (Enrich) | Latency |
|------|-------|--------|----------------|-----------------|---------|
| Light | `dmem-store-turn.exe` | `dmem-enrich.exe` | 5 (1,2,5,9,13) | 3 (2,3,5) | ~5ms |
| Standard | `dmem-store-turn-standard.exe` | `dmem-enrich-standard.exe` | 10 (1-5,8,9,13) | 5 (2,3,5,9,13) | ~50ms |
| Deep | `dmem-store-deep.exe` | `dmem-enrich-deep.exe` | 8 (1-4,8,9,13) | All 12+ (Think) | ~1-5s |

### Light Tier

Fastest. Uses service-layer RPCs only.

**Store** (`Store` RPC):
```
embed(1) → δ-mem.Store(2) → turbovec.AddVector(5) → self.LearnDomain(9) → persist(13)
```

**Enrich** (`Recall` + `IBNNForward` + `TurboSearch`):
```
Recall(prompt) → δ-mem(2) confidence gate
IBNNForward(prompt) → embed(1) + IBNN(3) sharpened embedding
TurboSearch(embedding) → turbovec(5) neighbor IDs
```

### Standard Tier

Full data flow matching main.go wiring. Both vector stores populated. All core layers active.

**Store** (`Store` + `Learn` RPCs):
```
Store RPC:  embed(1) → δ-mem(2) → turbovec(5) → self-model(9) → persist(13)
Learn RPC:  embed(1) → δ-mem(2) → turbogo(4) → IBNN reinforce(3) → temporal(8) → self-model(9) → persist(13)
```
Every fact enters BOTH vector stores, δ-mem accumulates twice (reinforcement), IBNN weights update.

**Enrich** (`Recall` + `IBNNForward` + `IBNNForwardHidden` + `TurboSearch` ×2):
```
Recall(prompt)           → δ-mem(2) confidence + correction vector
IBNNForward(prompt)      → embed(1) + IBNN(3) sharpened representation
IBNNForwardHidden(corr)  → IBNN(3) crystallize δ-mem correction
TurboSearch(sharpened)   → turbovec(5) primary neighbors (δ[])
TurboSearch(crystallized)→ turbovec(5) secondary neighbors (δ°[])
```
Dual search: raw embedding finds direct matches, crystallized correction finds deeper associations.

### Deep Tier

Heaviest. Invokes the full thoughts engine including Gemma and NLI.

**Store** (`Learn` RPC):
```
Learn → embed(1) → δ-mem(2) → turbogo(4) → IBNN reinforce(3) → temporal(8) → self-model(9) → persist(13)
```

**Enrich** (`Recall` + `Think` RPC):
```
Recall → δ-mem(2) confidence gate
Think(seeds) → δ-mem(2) + turbogo(4) + IBNN crystallize(3) + Gemma/substrate(6)
           → Truth validation(10) + NLI check(7) + Wander residuals(12)
           → Temporal record(8) + self-model(9) + store thought back(2,4,13)
```
Produces synthesized insights, not just retrieved IDs. Output format: `δ-think[conf|novelty] idea (via: neighbors)`

## Write Loop

```
┌──────────────────────────────────────────────────────────────┐
│  USER PROMPT                                                  │
│    ↓ userPromptSubmit                                        │
│    → dmem-enrich-*.exe                                       │
│        Search stored knowledge → inject as context           │
│                                                               │
│  ASSISTANT RESPONSE                                           │
│    ↓ stop                                                    │
│    → dmem-store-turn-*.exe                                   │
│        Extract facts → store into δ-mem-go                   │
│                                                               │
│  NEXT PROMPT → enrichment finds what was just stored          │
└──────────────────────────────────────────────────────────────┘
```

## All Binaries

| Binary | Trigger | Tier | Description |
|--------|---------|------|-------------|
| `dmem-enrich.exe` | userPromptSubmit | Light | TurboSearch retrieval |
| `dmem-enrich-standard.exe` | userPromptSubmit | Standard | Dual TurboSearch + crystallization |
| `dmem-enrich-deep.exe` | userPromptSubmit | Deep | Think-based synthesis |
| `dmem-store-turn.exe` | stop | Light | Store RPC (turbovec) |
| `dmem-store-turn-standard.exe` | stop | Standard | Store + Learn (both vector stores) |
| `dmem-store-deep.exe` | stop | Deep | Learn RPC (turbogo + IBNN) |
| `dmem-store-turn-all.exe` | both | Light | Single binary, prompts + responses |
| `dmem-store-turn-all-prompt.exe` | userPromptSubmit | Light | Split pair — user prompts |
| `dmem-store-turn-all-stop.exe` | stop | Light | Split pair — assistant responses |
| `hook.go` | stop | Legacy | Original hook (hash keys) |

## Authentication

All hooks use **auto-auth**:
- If `~/.memgo/token` exists → sends `x-api-key` gRPC metadata
- If no token file → connects without auth (localhost, no keys)

Generate a token:
```
mem-cli create-token dbrushchenko
```

## Key Design: Readable Keys

Store hooks use `content[:60]` as the TurboVec ID. TurboSearch returns these directly as context:

```
δ[I created the sentinel agent as a full clone of personal.js|0.87]
δ°[The server Store RPC triggers all 5 layers including turbo|0.72]
```

- `δ[]` = primary match (direct similarity)
- `δ°[]` = secondary match (crystallized association, standard tier only)
- `δ-think[]` = synthesized insight (deep tier only)

## Fact Extraction Heuristic

Lines from assistant responses are stored if they contain keywords:

- **Actions:** created, fixed, deployed, configured, installed, built, updated, deleted, moved, copied, cloned, wired
- **Decisions:** decided, chose, switched, set, changed, renamed
- **State:** running, listening, connected, verified, confirmed
- **Knowledge:** because, means, requires, uses, stores, loads, the issue, the problem, the fix, works by

Max 3 facts per response. Lines starting with ``` or --- or | are skipped.

## Agent Configuration

### Light (fastest, default for new agents)
```json
{
  "hooks": {
    "userPromptSubmit": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-enrich.exe", "timeout_ms": 5000 }
    ],
    "stop": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-store-turn.exe", "timeout_ms": 5000 }
    ]
  }
}
```

### Standard (recommended — full layer coverage)
```json
{
  "hooks": {
    "userPromptSubmit": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-enrich-standard.exe", "timeout_ms": 5000 }
    ],
    "stop": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-store-turn-standard.exe", "timeout_ms": 10000 }
    ]
  }
}
```

### Deep (heaviest — synthesized insights)
```json
{
  "hooks": {
    "userPromptSubmit": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-enrich-deep.exe", "timeout_ms": 10000 }
    ],
    "stop": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-store-deep.exe", "timeout_ms": 10000 }
    ]
  }
}
```

### Mixed (standard store + light enrich for speed)
```json
{
  "hooks": {
    "userPromptSubmit": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-enrich.exe", "timeout_ms": 5000 }
    ],
    "stop": [
      { "command": "C:\\Users\\dabrush\\.kiro\\hooks\\dmem-store-turn-standard.exe", "timeout_ms": 10000 }
    ]
  }
}
```

## Build

```bash
cd scripts/delivery/go-grpc

# Light
cd enrich && go build -o dmem-enrich.exe .
cd store-turn && go build -o dmem-store-turn.exe .

# Standard
cd enrich-standard && go build -o dmem-enrich-standard.exe .
cd store-turn-standard && go build -o dmem-store-turn-standard.exe .

# Deep
cd enrich-deep && go build -o dmem-enrich-deep.exe .
cd store-deep && go build -o dmem-store-deep.exe .

# Bidirectional (light tier)
cd store-turn-all && go build -o dmem-store-turn-all.exe .
cd store-turn-all/prompt && go build -o dmem-store-turn-all-prompt.exe .
cd store-turn-all/stop && go build -o dmem-store-turn-all-stop.exe .
```

Install: copy binaries to `~/.kiro/hooks/`

## Directory Structure

```
go-grpc/
├── README.md
├── hook.go                          ← legacy (hash keys)
├── hookauth/auth.go                 ← shared auth reference
├── enrich/main.go                   ← light enrichment
├── enrich-standard/main.go          ← standard enrichment (dual search)
├── enrich-deep/main.go              ← deep enrichment (Think)
├── store-turn/main.go               ← light store (responses only)
├── store-turn-standard/main.go      ← standard store (Store + Learn)
├── store-deep/main.go               ← deep store (Learn only)
└── store-turn-all/
    ├── main.go                      ← bidirectional single binary
    ├── prompt/main.go               ← split: user prompts
    └── stop/main.go                 ← split: assistant responses
```

## Session Dedup Files

Each tier uses its own session file to track what's been provided:
- Light: `~/.kiro/session/dmem-provided-grpc.json`
- Standard: `~/.kiro/session/dmem-provided-standard.json`
- Deep: `~/.kiro/session/dmem-think-provided.json`

## Comparison

| | Go gRPC Hooks | Python HTTP Hooks |
|---|--------------|-------------------|
| Dependencies | None (compiled) | Python 3 + chromadb |
| Protocol | gRPC + protobuf | HTTP + JSON |
| Startup | ~5ms | ~200ms |
| Port | 19090 | 18080 |
| Auth | Auto (token file) | Manual header |
| Layers | All 7 available | Vector search only |
| Vector stores | turbovec + turbogo | ChromaDB |
