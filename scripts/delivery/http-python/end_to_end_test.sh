#!/usr/bin/env bash
set -euo pipefail

echo "=== δ-mem-go End-to-End Test ==="

OWNER="e2e-test-$(date +%s)"
BASE="http://localhost:8080"

curl -s -X POST "$BASE/store" -H "Content-Type: application/json" \
  -d "{\"owner\":\"$OWNER\",\"key\":\"test\",\"content\":\"The quick brown fox jumps over the lazy dog.\"}" | jq

curl -s -X POST "$BASE/ibnn-forward" -H "Content-Type: application/json" \
  -d "{\"owner\":\"$OWNER\",\"text\":\"The quick brown fox\"}" | jq

curl -s -X POST "$BASE/generate" -H "Content-Type: application/json" \
  -d "{\"owner\":\"$OWNER\",\"prompt\":\"Explain delta-mem in one sentence.\"}" | jq

curl -s "$BASE/health" | jq
curl -s "$BASE/metrics" | head -n 20

echo "✅ End-to-end test PASSED"
