# δ-mem-go Delivery Methods

## Option A: Go gRPC (recommended)

Compiled Go binary that speaks gRPC directly to δ-mem-go (port 19090).
Zero dependencies, fast startup, proper protobuf serialization.

**Build:**
```
cd scripts/delivery/go-grpc
go build -o dmem-hook.exe .
```

**Install:**
Copy `dmem-hook.exe` to `~/.kiro/hooks/` and configure as a `stop` hook.

## Option B: Python HTTP (fallback)

Python script that calls δ-mem-go HTTP API (port 18080).
Uses only stdlib (`urllib`). Works without compilation.

**Install:**
Copy `dmem-store.py` to `~/.kiro/hooks/`

## Comparison

| | Go gRPC | Python HTTP |
|---|---------|-------------|
| Dependencies | None (compiled) | Python 3 |
| Protocol | gRPC + protobuf | HTTP + JSON |
| Startup time | ~5ms | ~200ms |
| Port | 19090 | 18080 |
| Size | ~5MB binary | ~3KB script |
