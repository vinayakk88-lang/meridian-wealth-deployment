"""smoke_test.py — hit /health, /agent/info, /chat against a local server.

Usage:
    python smoke_test.py                       # http://localhost:8000
    python smoke_test.py http://localhost:9000
"""

from __future__ import annotations

import sys

import httpx


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main(base_url: str) -> None:
    print(f"Smoke-testing {base_url}\n")

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        # /health
        r = client.get("/health")
        check("GET /health status 200", r.status_code == 200, f"got {r.status_code}")
        h = r.json()
        check(
            "/health fields present",
            {"status", "db_connected", "vectorstore_loaded", "tavily_configured"} <= h.keys(),
            str(h),
        )
        print(f"       status={h['status']}  db={h['db_connected']}  "
              f"vs={h['vectorstore_loaded']}  tavily={h['tavily_configured']}\n")

        # /agent/info
        r = client.get("/agent/info")
        check("GET /agent/info status 200", r.status_code == 200, f"got {r.status_code}")
        info = r.json()
        check("/agent/info has tools", len(info.get("tools", [])) > 0, f"tools={info.get('tools')}")
        print(f"       model={info['model']}  tools={[t['name'] for t in info['tools']]}  "
              f"docs={info.get('vectorstore_docs')}\n")

        # /chat
        payload = {
            "message": "What client IDs are available?",
            "include_trace": True,
            "max_iterations": 6,
        }
        r = client.post("/chat", json=payload)
        check("POST /chat status 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
        c = r.json()
        check("/chat returned answer", bool(c.get("answer")), "empty answer")
        check("/chat returned conversation_id", bool(c.get("conversation_id")))
        print(f"       conversation_id={c['conversation_id']}")
        print(f"       iterations={c['iteration_count']}  "
              f"tool_calls={len(c.get('tool_calls') or [])}")
        answer = c["answer"]
        snippet = answer if len(answer) < 240 else answer[:240] + "..."
        print(f"       answer: {snippet}\n")

    print("All checks passed.")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    try:
        main(url.rstrip("/"))
    except httpx.ConnectError:
        print(f"[FAIL] Could not connect to {url}.")
        print("       Is the server running? Start it with:")
        print("           uvicorn app:app --reload")
        sys.exit(2)
    except httpx.ReadTimeout:
        print(f"[FAIL] Request to {url} timed out.")
        print("       The agent may still be starting up (FAISS build on first run).")
        sys.exit(2)
