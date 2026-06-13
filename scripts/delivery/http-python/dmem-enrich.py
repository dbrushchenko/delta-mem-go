"""
Kiro CLI userPromptSubmit Hook — Automatic Memory Enrichment
=============================================================
Queries ChromaDB, re-ranks by layer weight + access_count, deduplicates
across turns. Only injects genuinely relevant, non-repetitive context.

Input (STDIN): {"hook_event_name": "userPromptSubmit", "prompt": "..."}
Output (STDOUT): Context added to conversation (or nothing if not needed)
"""
import sys, json, time
from pathlib import Path

CHROMA_PATH = str(Path.home() / "kiro-memoryv5" / "data" / "chromadb")
LOG_FILE = Path.home() / ".kiro" / "logs" / "dmem-enrich.log"
SESSION_FILE = Path.home() / ".kiro" / "session" / "dmem-provided.json"
MAX_CANDIDATES = 5
MAX_OUTPUT = 1  # single best match per turn
LAYER_WEIGHTS = {"procedural": 1.0, "semantic": 0.8, "episodic": 0.5, "working": 0.3}
DIST_CUTOFF = 1.3  # strict relevance


def log(msg: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def search_and_rank(prompt: str) -> list[dict]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection("memories")
        if col.count() == 0:
            return []
        n = min(MAX_CANDIDATES, col.count())
        results = col.query(query_texts=[prompt], n_results=n,
                           include=["metadatas", "documents", "distances"])
        candidates = []
        for i, mid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            if dist > DIST_CUTOFF:
                continue
            m = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            key = m.get("key", "")
            content = doc[len(key)+1:] if doc.startswith(key) else doc
            layer = m.get("layer", "working")
            access_count = int(m.get("access_count", 0))
            # Combined score: similarity * layer_weight * (1 + log(access))
            sim = max(0, 1 - dist / 2)
            lw = LAYER_WEIGHTS.get(layer, 0.3)
            import math
            boost = 1 + math.log1p(access_count) * 0.2
            score = sim * lw * boost
            candidates.append({"key": key, "content": content, "score": score,
                             "layer": layer, "dist": dist})
        candidates.sort(key=lambda x: -x["score"])
        return candidates
    except Exception as e:
        log(f"Error: {e}")
        return []


def load_provided() -> set:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SESSION_FILE.exists():
        try:
            return set(json.loads(SESSION_FILE.read_text(encoding="utf-8")))
        except:
            pass
    return set()


def save_provided(keys: set):
    SESSION_FILE.write_text(json.dumps(list(keys)[-200:]), encoding="utf-8")


def main():
    event = json.loads(sys.stdin.read())
    prompt = event.get("prompt", "")
    if not prompt or len(prompt) < 10:
        sys.exit(0)

    log(f"Q: {prompt[:60]}")
    candidates = search_and_rank(prompt)
    if not candidates:
        log("no relevant")
        sys.exit(0)

    # Filter already-provided
    provided = load_provided()
    new = [c for c in candidates if c["key"] not in provided][:MAX_OUTPUT]
    if not new:
        log("all seen")
        sys.exit(0)

    # Format output — ultra-compact for token efficiency
    lines = []
    for m in new:
        content = m["content"]
        # Strip boilerplate prefixes
        for prefix in ["Extracted from superseded ", "Extracted from consolidated ", "Key facts from "]:
            if content.startswith(prefix):
                idx = content.find(":\n")
                if idx > 0:
                    content = content[idx+2:]
                break
        content = content.strip().replace("\n", " ")[:120].encode("ascii", "replace").decode()
        lines.append(f"d[{m['key']}|{m['layer'][0]}] {content}")

    provided.update(m["key"] for m in new)
    save_provided(provided)
    print("\n".join(lines))
    log(f"+{len(new)} [{', '.join(m['key'][:30] for m in new)}]")


if __name__ == "__main__":
    main()
