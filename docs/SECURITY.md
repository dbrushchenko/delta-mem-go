# Security

## Authentication

### API Key Authentication

δ-mem-go uses shared API keys for both HTTP and gRPC interfaces. When API keys are configured, all requests must include a valid key.

**Configuration:**

```bash
# Environment variable (comma-separated)
API_KEYS=sk-prod-abc123,sk-prod-def456

# Keys are loaded at startup and stored in memory as map[string]bool
```

When `API_KEYS` is empty or unset, authentication is **disabled** (open access). This is the default for local development.

### HTTP Authentication

```
Header: X-API-Key: <your-key>
```

```bash
# Example
curl -H "X-API-Key: sk-prod-abc123" http://localhost:18080/health
```

Unauthorized requests receive HTTP 401.

### gRPC Authentication

```
Metadata: x-api-key: <your-key>
```

```go
// Go client example
md := metadata.New(map[string]string{"x-api-key": "sk-prod-abc123"})
ctx := metadata.NewOutgoingContext(context.Background(), md)
client.Store(ctx, &pb.StoreRequest{...})
```

Unauthorized requests receive gRPC status `UNAUTHENTICATED`.

### Implementation

```
internal/auth/auth.go
├── HTTPMiddleware()         — Wraps http.Handler, checks X-API-Key header
└── GRPCUnaryInterceptor()   — grpc.UnaryServerInterceptor, checks metadata
```

Both bypass authentication entirely when `validKeys` map is empty (no keys configured).

## Multi-Tenancy Isolation

### Owner-Based Isolation

Every API call requires an `owner` field. All state is partitioned by owner:

```
owner="alice"  →  alice.state, alice.ibnn.state, alice.self, alice.turbo
owner="bob"    →  bob.state, bob.ibnn.state, bob.self, bob.turbo
```

### Isolation Guarantees

| Layer | Isolation Mechanism |
|-------|-------------------|
| δ-mem | Separate `Module` instance per owner via `OwnerManager` |
| IBNN | Separate weight matrices per owner via `OwnerManager` |
| TurboGo | Separate quantized index per owner via `OwnerManager` |
| TurboVec | Separate in-memory vector store per owner |
| Self-Model | Shared (domain confidence is global) |
| Truth Engine | Shared (axioms apply globally) |
| Temporal | Events tagged with owner, filtered on retrieval |
| Wander | Separate `Wanderer` goroutine per owner |

### What is NOT isolated

- **Truth axioms** — Registered axioms apply to all owners. An axiom added by one owner constrains all thought generation.
- **NLI model** — The DeBERTa contradiction checker is a shared inference resource.
- **Self-model** — Domain confidence is aggregated across all owners (by design — it represents the system's overall competence map).
- **Rate limiter** — Per-owner rate limiting (1000 req/min default).

### Owner Manager Pattern

```go
// Each layer follows this pattern:
type OwnerManager struct {
    mu      sync.RWMutex
    owners  map[string]*Module  // isolated state per owner
    cfg     Config
    dataDir string
}
```

State is lazily loaded from disk on first access and saved on explicit save or shutdown.

## NLI Truth Constraints

### Purpose

The Natural Language Inference (NLI) model provides a second-opinion on truth validation. It detects logical contradictions between generated thoughts and established knowledge.

### Model

- **Model:** nli-deberta-v3-xsmall
- **Size:** 284 MB (ONNX)
- **Tokenizer:** nli-tokenizer.json (8.6 MB)
- **Inference:** In-process ONNX Runtime (no network calls)

### How It Works

```
Generated thought  ─┐
                    ├── NLI DeBERTa ──→ {entailment, neutral, contradiction}
Stored axiom/fact  ─┘
```

- **Entailment** — Thought is consistent with known facts
- **Neutral** — No relationship detected
- **Contradiction** — Thought violates known facts → marked invalid

### Truth Validation Pipeline

1. **Heuristic check** — Fast axiom matching (registered statements, domain rules)
2. **NLI check** — DeBERTa inference against relevant stored knowledge
3. **Verdict** — `{valid: bool, grounding: float32, reason: string}`

### Consequences of Truth Failure

- Thought marked `valid=false`
- IBNN weights weakened by -0.001 (discourages similar outputs)
- If iterations remain, system retries with "Avoid: {reason}" appended to seeds
- Thought still returned to caller (with `valid=false` flag)

### Limitations

- NLI operates on text pairs — it cannot detect multi-hop contradictions
- Small model (xsmall) trades accuracy for speed
- No automatic axiom registration — axioms must be explicitly added via `AddAxiom` RPC

## Data at Rest

### State Files

All persistent state is stored as binary files in the configured data directory:

| File | Format | Contains | Sensitivity |
|------|--------|----------|-------------|
| `*.state` | Custom binary (gob) | δ-mem weight matrices | Medium — embeddings of stored knowledge |
| `*.ibnn.state` | Custom binary (gob) | IBNN connection weights | Low — no plaintext |
| `*.self` | Gob-encoded struct | Domain confidence map | Low — statistical metadata |
| `*.turbo` | Custom binary | Quantized vectors + ID strings | Medium — vector IDs may reveal topic names |

### What's Stored in State Files

- **δ-mem `.state`:** Float32 weight matrices encoding all stored knowledge as interference patterns. Not human-readable, but an adversary with the embedding model could reconstruct approximate content.
- **IBNN `.ibnn.state`:** Neural network weights. No plaintext content, only learned inhibition patterns.
- **TurboGo index:** 4-bit quantized vectors + string IDs. IDs are derived from the `key` parameter of Store calls (first 60 chars).

### Encryption

**Currently: No encryption at rest.**

State files are written as raw binary to the filesystem. Protection relies on:
- OS-level file permissions (installer sets per-user directory ACLs)
- Full-disk encryption (BitLocker, LUKS) at the OS level
- Kubernetes secrets for API keys in cluster deployments

### Recommendations

For sensitive deployments:
1. Enable BitLocker on the data volume (Windows)
2. Use encrypted PVCs in Kubernetes (StorageClass with encryption)
3. Set restrictive ACLs on the data directory
4. Use `API_KEYS` to prevent unauthorized access
5. Run behind an Istio gateway with mTLS for network encryption

### Model Files

ONNX models are read-only and not sensitive (publicly available weights):
- `nomic-embed-text-v1.5.onnx` — Public model from HuggingFace
- `nli-deberta.onnx` — Public model from HuggingFace
- `onnxruntime.dll` — Public Microsoft runtime

### Network Security

| Protocol | Default Port | Encryption |
|----------|-------------|------------|
| HTTP | 8080 (container) / 18080 (Windows) | None (add reverse proxy for TLS) |
| gRPC | 9090 (container) / 19090 (Windows) | None (insecure credentials) |

**In Kubernetes:** Istio sidecar provides mTLS between pods automatically.

**On Windows:** For production use behind a network boundary, add a reverse proxy (NGINX, Caddy) with TLS termination, or use the gRPC client with TLS credentials.

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Unauthorized access | API key authentication on all endpoints |
| Cross-tenant data leak | Owner-partitioned state, separate OwnerManager instances |
| Knowledge extraction via recall | Rate limiting (1000 req/min per owner) |
| Model poisoning via adapt/learn | Truth engine validates before IBNN reinforcement |
| State file theft | OS-level encryption, restrictive permissions |
| Network eavesdropping | Istio mTLS (K8s), reverse proxy TLS (Windows) |
| Denial of service | Rate limiter, resource limits in K8s deployment |
