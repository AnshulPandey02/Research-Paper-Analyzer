"""
Live end-to-end check against a RUNNING backend. Uses the real external
services, so it needs internet access and a valid GEMINI_API_KEY.

Start the backend first, then from the project root run:
    backend\\.venv\\Scripts\\python.exe tests\\smoke_test.py
    backend\\.venv\\Scripts\\python.exe tests\\smoke_test.py path\\to\\your_paper.pdf
"""

import os
import sys
import time

import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sample_paper.pdf")

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def upload(name, data, timeout=200):
    started = time.time()
    resp = requests.post(f"{BASE}/analyze", files={"file": (name, data, "application/pdf")}, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body, time.time() - started


def main():
    try:
        health = requests.get(f"{BASE}/health", timeout=10)
    except requests.RequestException as e:
        print(f"[FAIL] backend not reachable at {BASE}: {e}")
        return 1
    info = health.json()
    check("GET /health", health.status_code == 200 and info.get("status") == "ok", str(info))
    check("GET /docs", requests.get(f"{BASE}/docs", timeout=10).status_code == 200)
    check("Gemini key configured", bool(info.get("gemini_configured")))

    with open(PDF, "rb") as f:
        pdf_bytes = f.read()

    print(f"\nUploading {os.path.basename(PDF)} (this takes 20-60 seconds)...")
    status, body, seconds = upload(os.path.basename(PDF), pdf_bytes)
    check("POST /analyze returns 200", status == 200, f"HTTP {status} in {seconds:.1f}s {body.get('error', '')}")
    if status == 200:
        extraction = body.get("extraction", {})
        plagiarism = body.get("plagiarism", {})
        bias = body.get("bias_analysis", {})
        check("PDF text extracted", extraction.get("words", 0) > 60, str(extraction))
        check("plagiarism analysis completed", plagiarism.get("status") in ("ok", "partial"),
              f"status={plagiarism.get('status')} overall={plagiarism.get('overall_percent')}% "
              f"providers={plagiarism.get('providers')}")
        sections = plagiarism.get("sections", {})
        check("all four sections present", set(sections) == {"Title", "Abstract", "Methodology", "Conclusions"})
        check("at least one section has matches", any(s.get("matches") for s in sections.values()))
        check("bias analysis completed", bias.get("status") == "ok",
              f"status={bias.get('status')} model={bias.get('model')} score={bias.get('overall_score')} "
              f"biases={len(bias.get('biases', []))} error={bias.get('error')}")
        check("overall status", body.get("status") in ("ok", "partial"),
              f"{body.get('status')} warnings={body.get('warnings')}")

        status2, body2, seconds2 = upload(os.path.basename(PDF), pdf_bytes)
        if body.get("status") == "ok":
            check("repeat upload served from cache", status2 == 200 and body2.get("cached") is True, f"{seconds2:.2f}s")

    print("\nBad uploads must be rejected with a clear message:")
    for name, data, expected in [
        ("empty.pdf", b"", 400),
        ("notes.pdf", b"this is plain text, not a pdf", 415),
        ("corrupt.pdf", b"%PDF-1.7\n" + os.urandom(3000), 422),
    ]:
        status, body, _ = upload(name, data, timeout=60)
        check(f"{name} -> HTTP {expected}", status == expected and bool(body.get("error")), body.get("error", ""))

    check("backend still healthy", requests.get(f"{BASE}/health", timeout=10).status_code == 200)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
