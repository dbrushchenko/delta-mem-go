#!/usr/bin/env bash
set -euo pipefail

OWNER="self-train-$(date +%s)"
ITERATIONS=${1:-5}

echo "Starting full self-training for owner $OWNER ($ITERATIONS iterations)"

python scripts/merge_text_files.py data/raw/ --output data/merged.txt

python scripts/prepare_training_data.py data/merged.txt \
  --output data/train.jsonl \
  --synthetic \
  --service-url http://localhost:8080/generate

go run ./internal/examples/selftrain.go --owner "$OWNER" --iterations "$ITERATIONS"

echo "✅ Full self-training pipeline completed successfully!"
echo "   Owner: $OWNER"
