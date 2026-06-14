# User Guide

## Installation

### Option 1: Windows Installer (recommended)

```powershell
# Run the installer
.\delta-mem-go-setup.exe
```

The installer detects whether you have administrator privileges:

**Administrator mode** (right-click → Run as Administrator):
- Installs to `C:\Program Files\DeltaMemGo\`
- Registers NSSM Windows service (`DeltaMemGo`)
- Auto-starts on boot
- Manages service with: `nssm status|start|stop|restart DeltaMemGo`

**Standard user mode** (double-click):
- Installs to `%APPDATA%\mem-go\`
- Creates startup shortcut (runs at login)
- No admin rights required

The installer prompts for:
1. Install directory
2. Path to `nomic-embed-text-v1.5.onnx` model
3. HTTP port (default: 18080)
4. gRPC port (default: 19090)
5. Owner name (default: your Windows username)
6. Training data file (optional)
7. Hook delivery method (Go gRPC binary or Python HTTP script)

### Option 2: Docker

```bash
# Clone the repository
git clone https://code.usgs.gov/daniel_brushchenko/mem-go.git
cd mem-go

# Place models in ./models/
#   nomic-embed-text-v1.5.onnx
#   nomic-embed-text-v1.5.onnx.data
#   tokenizer.json
#   nli-deberta.onnx (optional)
#   nli-tokenizer.json (optional)

# Start full stack
docker compose up --build -d

# Verify
curl http://localhost:8080/health
```

Services started:
- `delta-mem` — Main server on :8080 (HTTP) and :9090 (gRPC)
- `ollama` — Gemma 4 language model on :11434
- `turbovec` — Vector search sidecar on :8001

### Option 3: Manual Build

Prerequisites:
- Go 1.25+
- CGO enabled (C compiler: GCC/MinGW on Windows)
- ONNX Runtime 1.26.0 DLL/SO

```powershell
# Build server
set CGO_ENABLED=1
go build -o delta-mem-go.exe ./cmd/delta-mem-go

# Build CLI
go build -o mem-cli.exe ./cmd/mem-cli

# Run
.\delta-mem-go.exe --model C:\path\to\models\nomic-embed-text-v1.5.onnx ^
    --port 18080 --grpc-port 19090 --data .\data
```

Required model files (place in a `models\` directory):
| File | Size | Source |
|------|------|--------|
| `nomic-embed-text-v1.5.onnx` | 2 MB | [HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| `nomic-embed-text-v1.5.onnx.data` | 549 MB | Same repo |
| `tokenizer.json` | 711 KB | Same repo |
| `onnxruntime.dll` | 14.9 MB | [GitHub](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0) |
| `nli-deberta.onnx` | 284 MB | [HuggingFace](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) (optional) |
| `nli-tokenizer.json` | 8.6 MB | Same repo (optional) |

## First-Time Initiation

Initiation bootstraps the δ-mem substrate with your domain knowledge. It runs 5 epochs over your training data at lr=0.01 (10× normal learning rate) to rapidly establish a knowledge base.

### Preparing Training Data

Training data is a plain text file containing your domain knowledge. One fact/concept per line works best, but paragraphs are also fine.

```
# Example: training_data.txt
Kubernetes uses etcd for distributed state storage
Pods are the smallest deployable unit in Kubernetes
Services provide stable networking for pod sets
Istio injects Envoy sidecars for traffic management
...
```

Size guideline: ~90 KB of text produces a well-initiated substrate in ~45 seconds.

### Running Initiation

```powershell
# Windows (service must be stopped first)
nssm stop DeltaMemGo
.\delta-mem-go.exe --initiate --owner dabrush --training-data .\data\training_data\training_3.txt --model .\models\nomic-embed-text-v1.5.onnx --data .\data\initiated
nssm start DeltaMemGo
```

```bash
# Docker (exec into container)
docker compose exec delta-mem delta-mem-go \
    --initiate --owner dabrush \
    --training-data /data/training_data/training_3.txt \
    --model /models/nomic-embed-text-v1.5.onnx \
    --data /data/states
```

### What Initiation Does

1. Loads the training corpus
2. Splits into chunks
3. Embeds each chunk via nomic (768-dim)
4. Runs 5 epochs of δ-mem store + recall + projection update at lr=0.01
5. Builds IBNN weights from the training signal
6. Saves `.state` and `.ibnn.state` to disk
7. Total time: ~45 seconds for 90 KB corpus

After initiation, the service can be restarted and will load the trained state.

### Initiation on the Mesh (Microservice Mode)

When running as a mesh microservice, each owner initiates via API call — not at startup.

**Via mem-cli (gRPC):**
```bash
mem-cli --addr delta-mem.mesh.svc:9090 initiate --owner jsmith --training-data ./my-corpus.txt
```

**Via HTTP API:**
```bash
curl -X POST http://delta-mem.mesh.svc:8080/initiate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "owner": "jsmith",
    "text": "...your domain training corpus...",
    "epochs": 5,
    "learning_rate": 0.01
  }'
```

**Response:**
```json
{
  "chunks": 546,
  "epochs": 5,
  "duration": "45.2s",
  "state_norm": 0.12,
  "avg_confidence": 0.054
}
```

**How it works on the mesh:**
1. Any pod can serve the initiation request (stateless — state goes to Redis+PG)
2. Training runs on the pod that received the request (~45s for 90 KB)
3. Resulting state is stored in Redis (shared cache) + PostgreSQL (durable)
4. After initiation, any pod can serve Store/Recall/Think for that owner
5. No restart needed — the owner is immediately active

**When to initiate:**
- First time an owner starts using the system
- When switching domains (new team, new project)
- To refresh after significant knowledge changes

**Size guidelines:**
- 10 KB corpus → ~5s initiation (light)
- 90 KB corpus → ~45s initiation (standard)
- 500 KB corpus → ~4 min initiation (comprehensive)

## Using mem-cli

`mem-cli` is a compiled Go binary that communicates with δ-mem-go over gRPC.

### Connection

```powershell
# Default: localhost:19090
mem-cli health

# Custom address
mem-cli --addr 10.0.1.50:9090 health

# Custom owner
mem-cli --owner alice store --key "fact1" --content "some knowledge"
```

The `--owner` flag defaults to your Windows `%USERNAME%`.

### Commands

#### store — Encode knowledge

```powershell
mem-cli store --key "k8s-networking" --content "Kubernetes Services use iptables rules for packet routing"
# Output: stored: norm=3.2451

# Shorthand (key = first arg, content = rest)
mem-cli store k8s-networking Kubernetes Services use iptables rules for packet routing
```

#### recall — Retrieve related knowledge

```powershell
mem-cli recall "how does Kubernetes route traffic"
# Output: confidence=0.0234
```

Confidence indicates how strongly the substrate resonates with the query. Higher values mean more relevant stored knowledge was activated.

#### think — Synthesize novel ideas

```powershell
mem-cli think "microservices" "event sourcing" "CQRS"
# Output:
# idea: microservices ∩ event-sourcing ∩ CQRS → distributed-saga + compensating-transactions
# depth=3 conf=0.0312 novelty=0.7821 valid=true
```

The Think operation:
- Takes 1+ seed concepts
- Runs iterative re-entry (up to 5 depth)
- Returns a novel synthesis with confidence and novelty scores
- `valid=true` means it passed truth constraints

#### adapt — Correct misconceptions

```powershell
mem-cli adapt "Python is compiled to native code" "Python is interpreted via CPython bytecode"
# Output: adapted: impact=0.1523
```

Adapt replaces wrong knowledge without catastrophic forgetting. The impact score indicates how much the substrate changed.

#### learn — Absorb new facts

```powershell
mem-cli learn "gRPC uses HTTP/2 for transport and Protocol Buffers for serialization"
# Output: learned
```

Learn stores a fact and updates the self-model's domain confidence.

#### health — Check server status

```powershell
mem-cli health
# Output: owners=3 stores=1247 recalls=892 uptime=4h32m15s
```

## kiro-cli Hook Setup

δ-mem-go integrates with kiro-cli to automatically store knowledge from AI conversations.

### Go gRPC Hook (recommended)

The installer places `dmem-hook.exe` in `%USERPROFILE%\.kiro\hooks\`:

```
~/.kiro/hooks/dmem-hook.exe
```

This binary reads kiro-cli hook events from stdin and stores assistant responses into δ-mem-go via gRPC. No configuration needed if the server runs on `localhost:19090`.

### Python HTTP Hook (fallback)

If you chose Python during installation, the hook lives at:

```
~/.kiro/hooks/dmem-store.py
```

```python
# Auto-generated by installer — stores assistant responses via HTTP
import sys, json, urllib.request
MEMORY_URL = "http://localhost:18080"
def main():
    data = json.loads(sys.stdin.read())
    if data.get("hook_event_name") != "stop": return
    resp = data.get("assistant_response", "")
    if len(resp) < 100: return
    body = json.dumps({"owner": "dabrush", "key": "hook-auto", "content": resp[:500]}).encode()
    try: urllib.request.urlopen(urllib.request.Request(MEMORY_URL + "/store", body, {"Content-Type": "application/json"}), timeout=3)
    except: pass
if __name__ == "__main__": main()
```

### Manual Hook Setup

If you didn't use the installer:

```powershell
# Create hooks directory
mkdir %USERPROFILE%\.kiro\hooks

# Build the Go hook
go build -o %USERPROFILE%\.kiro\hooks\dmem-hook.exe ./scripts/delivery/go-grpc

# Or copy the Python script
copy scripts\delivery\http-python\hook.py %USERPROFILE%\.kiro\hooks\dmem-store.py
```

## Adaptation and Learning

### How the System Learns

δ-mem-go learns continuously through three mechanisms:

1. **Passive learning** (every recall) — Projection matrices are updated with lr=0.001 to improve future recall quality.

2. **Active learning** (store/learn) — New knowledge is encoded into the δ-mem substrate and indexed in TurboGo for retrieval.

3. **Reinforcement learning** (think) — IBNN weights are reinforced (+0.001) when generated thoughts pass truth validation, and weakened (-0.001) when they fail.

### Adapt vs Learn

| Operation | Purpose | Mechanism |
|-----------|---------|-----------|
| `learn` | Add new knowledge | Store embedding, update self-model |
| `adapt` | Correct wrong knowledge | Suppress wrong embedding, strengthen right one |

Use `adapt` when the system has stored incorrect information. It doesn't delete — it overwrites the wrong pattern with the correct one.

### Domain Confidence

The self-model tracks which domains the system knows well (high confidence on recall) and which are weak. This affects:
- **Surprise threshold** — Known domains have higher thresholds (less deep processing needed)
- **Retrieval depth** — Unknown domains trigger k=8 retrieval (vs k=3 for known)

### Wander Mode

Wander mode runs background thought synthesis, exploring connections the system hasn't been explicitly asked about:

```powershell
# Start background wandering (via gRPC)
# Use the gRPC API directly — no CLI command yet
grpcurl -plaintext localhost:19090 deltamem.DeltaMem/StartWander -d '{"owner":"dabrush"}'

# Harvest spontaneous insights
grpcurl -plaintext localhost:19090 deltamem.DeltaMem/HarvestWander -d '{"owner":"dabrush"}'

# Stop wandering
grpcurl -plaintext localhost:19090 deltamem.DeltaMem/StopWander -d '{"owner":"dabrush"}'
```

## Monitoring

### Prometheus Metrics

Available at `GET /metrics`:
- `deltamem_stores_total` — Total store operations
- `deltamem_recalls_total` — Total recall operations
- `deltamem_think_duration_seconds` — Think operation latency histogram
- Standard Go runtime metrics

### Health Check

```bash
curl http://localhost:18080/health
```

```json
{
  "owners_active": 2,
  "avg_state_norm": 3.45,
  "total_stores": 5432,
  "total_recalls": 3210,
  "uptime": "12h45m30s"
}
```

### Service Management (Windows)

```powershell
nssm status DeltaMemGo      # Check service state
nssm restart DeltaMemGo     # Restart after config change
nssm stop DeltaMemGo        # Stop service
nssm edit DeltaMemGo        # GUI editor for service config

# Logs
type "C:\Program Files\DeltaMemGo\service.log"
# or
type "%APPDATA%\mem-go\service.log"
```

## Performance Reference

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Store | ~20 ms | Embed + δ-mem write + TurboGo index |
| Recall | ~15 ms | Embed + sparse recall |
| Think | 1–4 s | Depends on depth (1–5 iterations) |
| Adapt | ~150 ms | Two embeds + δ-mem update |
| Learn | ~25 ms | Embed + store + self-model update |
| Health | <1 ms | In-memory counters |
| Initiation | ~45 s | 5 epochs, 90 KB corpus |

## Troubleshooting

### Service won't start

```powershell
# Check NSSM log
type "C:\Program Files\DeltaMemGo\service.log"

# Common issues:
# - onnxruntime.dll not found → place next to model files
# - Model file missing → download nomic-embed-text-v1.5.onnx
# - Port in use → change --port/--grpc-port
```

### Low confidence on recall

- System needs more training data — run initiation with a larger corpus
- Domain not well-covered — use `learn` to add relevant facts
- Embedding dimension too low — increase `--embed-dim` (default 768 is best)

### Think returns invalid thoughts

- Add axioms for your domain: `AddAxiom("statement", "domain")`
- Ensure NLI model is loaded (check startup log for "NLI enabled")
- The system will self-correct over time (IBNN weakening on failures)

### State corruption

```powershell
# Reset a specific owner's state
curl -X POST http://localhost:18080/reset -d '{"owner":"dabrush"}'

# Or delete state files and restart
del data\states\dabrush.state
del data\states\dabrush.ibnn.state
nssm restart DeltaMemGo
```

## Using MCP Tools (Agent Integration)

δ-mem-go exposes 14 tools via MCP at `/mcp` on the HTTP port.

### Agent Configuration

Localhost (no auth):
```json
{
  "dmem": { "url": "http://localhost:18080/mcp" }
}
```

Mesh (with API key):
```json
{
  "dmem": {
    "url": "https://dmem.mesh.gs.doi.net/mcp",
    "headers": { "X-API-Key": "your-service-key" }
  }
}
```

Local stdio (Python wrapper, for per-session process):
```json
{
  "dmem": {
    "command": "C:\\Users\\dabrush\\kiro-dmem\\venv\\Scripts\\python.exe",
    "args": ["C:\\Users\\dabrush\\kiro-dmem\\server.py"],
    "env": { "MCP_TRANSPORT": "stdio" }
  }
}
```

### Example Tool Usage

```
@dmem/dmem_store_deep key="k8s-mesh" content="compute mesh uses Flagger canary"
@dmem/dmem_think seeds="LoRaWAN, data pipeline, Kubernetes"
@dmem/dmem_validate statement="LoggerNet runs on Kubernetes pods"
@dmem/dmem_search query="certificate management" deep=true
@dmem/dmem_confident text="InSAR processing"
@dmem/dmem_wander action="harvest"
```

### Kiro CLI Hooks (Automatic Background Memory)

Hooks fire automatically on triggers — no agent action needed:

| Tier | Store Hook (stop) | Enrich Hook (userPromptSubmit) |
|------|-------------------|-------------------------------|
| Light | `dmem-store-turn-light.exe` (Store) | `dmem-enrich-light.exe` (Recall+IBNN+TurboSearch) |
| Standard | `dmem-store-turn-standard.exe` (Store+Learn) | `dmem-enrich-turn-standard.exe` (dual TurboSearch) |
| Deep | `dmem-store-turn-deep.exe` (StoreDeep) | `dmem-enrich-turn-deep.exe` (all layers individually) |
| Think | — | `dmem-enrich-deep.exe` (Think RPC, black box) |
| Wander | `dmem-store-turn-wander.exe` (StartWander) | `dmem-enrich-turn-wander.exe` (HarvestWander) |

Hooks are compiled Go binaries in `~/.kiro/hooks/`. See `scripts/delivery/go-grpc/README.md` for full details.
