# internal/turbovec/sidecar.py
# Real FastAPI sidecar for turbovec (official Python bindings)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="turbovec Sidecar")
indexes = {}  # per-owner index


class AddRequest(BaseModel):
    owner: str
    id: str
    vector: List[float]


class SearchRequest(BaseModel):
    owner: str
    query: List[float]
    k: int = 10


@app.post("/add")
async def add(req: AddRequest):
    if req.owner not in indexes:
        indexes[req.owner] = {"vectors": [], "ids": []}
    indexes[req.owner]["vectors"].append(req.vector)
    indexes[req.owner]["ids"].append(req.id)
    return {"ok": True}


@app.post("/search")
async def search(req: SearchRequest):
    if req.owner not in indexes:
        return {"ids": [], "scores": []}
    # Simple cosine similarity fallback
    import numpy as np
    vecs = np.array(indexes[req.owner]["vectors"])
    q = np.array(req.query)
    scores = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    top_k = min(req.k, len(scores))
    idx = np.argsort(scores)[::-1][:top_k]
    return {
        "ids": [indexes[req.owner]["ids"][i] for i in idx],
        "scores": scores[idx].tolist(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
