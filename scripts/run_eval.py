import json
import sys
import requests

API = "http://localhost:8000"

if len(sys.argv) < 2:
    print("Usage: python -m scripts.run_eval <doc_id>")
    sys.exit(1)

doc_id = sys.argv[1]
questions = json.load(open("tests/eval_questions.json", encoding="utf-8"))
score = {"baseline": 0, "hybrid": 0}

print(f"\n--- Running Evaluation Benchmark for Doc ID: {doc_id} ---")
for item in questions:
    body = {"doc_id": doc_id, "question": item["q"]}
    for mode in ("baseline", "hybrid"):
        try:
            resp = requests.post(f"{API}/ask/{mode}", json=body, timeout=300).json()
            ans = resp.get("answer", "")
            ok = all(e.lower() in ans.lower() for e in item["expect"])
            score[mode] += ok
            status = "PASS" if ok else "FAIL"
            print(f"[{mode:8}] {status} ({item['type']}) {item['q']}")
        except Exception as e:
            print(f"[{mode:8}] ERROR ({item['type']}) {item['q']}: {e}")

n = len(questions)
print(f"\n==========================================")
print(f"Benchmark Results: Baseline: {score['baseline']}/{n} | Hybrid: {score['hybrid']}/{n}")
print(f"==========================================\n")
