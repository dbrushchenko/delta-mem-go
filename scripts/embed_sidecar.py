"""Embedding sidecar for delta-mem-go.
Serves HTTP /embed and /health. Uses sentence-transformers all-MiniLM-L6-v2.
Works on Windows (local testing) and Linux (mesh pod).

Usage:
    python embed_sidecar.py                    # defaults
    python embed_sidecar.py --port 8000        # custom port
    EMBEDDER_MODEL=nomic-ai/nomic-embed-text-v1.5 python embed_sidecar.py  # different model
"""

import os
import json
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
MODEL_PATH = os.environ.get("EMBEDDER_MODEL_PATH", "")
PORT = int(os.environ.get("EMBEDDER_PORT", "8000"))

# Load model (uses cache if already downloaded)
print(f"[embed] Loading {MODEL_NAME}...")
if MODEL_PATH and os.path.isdir(MODEL_PATH):
    model = SentenceTransformer(MODEL_PATH)
else:
    model = SentenceTransformer(MODEL_NAME)
DIM = model.get_sentence_embedding_dimension()
print(f"[embed] Ready. dim={DIM} port={PORT}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "dim": DIM, "model": MODEL_NAME}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/embed":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            target_dim = body.get("dim", DIM)

            vec = model.encode(text, normalize_embeddings=True).tolist()

            # Matryoshka truncation if requested
            if target_dim < len(vec):
                vec = vec[:target_dim]
                # Re-normalize after truncation
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"vector": vec}).encode())

        elif self.path == "/embed_batch":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            texts = body.get("texts", [])
            target_dim = body.get("dim", DIM)

            vecs = model.encode(texts, normalize_embeddings=True).tolist()
            if target_dim < DIM:
                truncated = []
                for vec in vecs:
                    v = vec[:target_dim]
                    norm = math.sqrt(sum(x * x for x in v))
                    if norm > 0:
                        v = [x / norm for x in v]
                    truncated.append(v)
                vecs = truncated

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"vectors": vecs}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[embed] Listening on :{args.port}")
    server.serve_forever()
