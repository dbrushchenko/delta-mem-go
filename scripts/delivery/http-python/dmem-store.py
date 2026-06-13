"""
Kiro CLI stop Hook — Auto-Store Facts from Responses
=====================================================
Runs after every assistant response. Extracts key facts/decisions
and stores them in memory automatically.

Input (STDIN): {"hook_event_name": "stop", "assistant_response": "...", "cwd": "..."}
Exit 0: success (storage is fire-and-forget)
"""
import sys, json, time, re, hashlib
from pathlib import Path

LOG_FILE = Path.home() / ".kiro" / "logs" / "dmem-store.log"
STORE_FILE = Path.home() / ".kiro" / "session" / "dmem-auto-stored.json"
MEMORY_URL = "http://localhost:18080"

# Only store responses that contain actionable/factual content
MIN_RESPONSE_LENGTH = 100
MAX_STORE_PER_TURN = 2


def log(msg: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def extract_facts(response: str) -> list[dict]:
    """Extract storable facts from assistant response. Lightweight heuristics."""
    facts = []

    # Pattern 1: Decisions made (logged with base_log_decision or stated)
    if "decision" in response.lower() or "chose" in response.lower() or "selected" in response.lower():
        # Extract first sentence after decision keyword
        for line in response.split("\n"):
            if any(w in line.lower() for w in ["decided", "chose", "selected", "going with", "using"]):
                facts.append({"key": f"decision-{hashlib.md5(line.encode()).hexdigest()[:8]}", "content": line.strip()[:200]})
                break

    # Pattern 2: Completed actions (file created, deployed, configured)
    action_patterns = [
        r"(?:Created|Deployed|Configured|Installed|Updated|Fixed|Built|Saved)\s+(.+?)(?:\.|$)",
    ]
    for pat in action_patterns:
        matches = re.findall(pat, response, re.MULTILINE)
        for m in matches[:1]:
            facts.append({"key": f"action-{hashlib.md5(m.encode()).hexdigest()[:8]}", "content": m.strip()[:200]})

    # Pattern 3: Important state (PID, IP, path, version discovered)
    state_patterns = [
        r"(PID \d+.+?)(?:\n|$)",
        r"((?:running|listening|deployed) (?:on|at) .+?)(?:\n|$)",
    ]
    for pat in state_patterns:
        matches = re.findall(pat, response)
        for m in matches[:1]:
            facts.append({"key": f"state-{hashlib.md5(m.encode()).hexdigest()[:8]}", "content": m.strip()[:200]})

    return facts[:MAX_STORE_PER_TURN]


def store_to_memory(facts: list[dict]):
    """Store facts via memory MCP server."""
    import urllib.request

    for fact in facts:
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "memory_store", "arguments": {
                "key": fact["key"], "content": fact["content"],
                "tags": ["auto-captured"]
            }}
        }).encode()
        req = urllib.request.Request(
            f"{MEMORY_URL}/mcp", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            log(f"Store failed for {fact['key']}: {e}")


def already_stored(fact_key: str) -> bool:
    """Check if we already stored this fact (dedup)."""
    STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    stored = set()
    if STORE_FILE.exists():
        try:
            stored = set(json.loads(STORE_FILE.read_text(encoding="utf-8")))
        except:
            pass
    if fact_key in stored:
        return True
    stored.add(fact_key)
    # Keep last 500
    if len(stored) > 500:
        stored = set(list(stored)[-500:])
    STORE_FILE.write_text(json.dumps(list(stored)), encoding="utf-8")
    return False


def main():
    event = json.loads(sys.stdin.read())
    response = event.get("assistant_response", "")

    if len(response) < MIN_RESPONSE_LENGTH:
        sys.exit(0)

    facts = extract_facts(response)
    new_facts = [f for f in facts if not already_stored(f["key"])]

    if new_facts:
        store_to_memory(new_facts)
        log(f"Auto-stored {len(new_facts)} facts: {[f['key'] for f in new_facts]}")

    sys.exit(0)


if __name__ == "__main__":
    main()
