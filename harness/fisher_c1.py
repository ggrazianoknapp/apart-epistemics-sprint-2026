"""Exploratory check cited in Appendix C: do models that report no hidden reasoning at the pushback turn give in more
inside the chat than to the same claim as a hearsay hint in a new chat? Main run, single samples, user-wrong cases
where both replies have an answer (the same cases as Chart 4 of results_page.py). One-sided Fisher exact test.
Usage: uv run python harness/fisher_c1.py
"""
import json
from math import comb
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
L = lambda r, s: (r.get(s) or {}).get("letter")
rows = []
for run in ["main", "mask"]:
    for f in sorted((ROOT / "runs" / run).glob("*.jsonl")):
        if f.name.startswith(("calls__", "extract")):
            continue
        recs = [json.loads(l) for l in f.open()]
        if not recs:
            continue
        uw = [r for r in recs if r["arm"] == "pushback" and not r["user_right"] and L(r, "t1") == r["answer"]]
        both = [r for r in uw if L(r, "t2") and L(r, "fresh_hint")]
        if len(both) < 30:
            continue
        chat = sum(L(r, "t2") == r["pushed"] for r in both) / len(both)
        fresh = sum(L(r, "fresh_hint") == r["pushed"] for r in both) / len(both)
        no_reasoning = median((r["t2"].get("reasoning_tokens") or 0) for r in both) == 0
        rows.append((recs[0]["model"], no_reasoning, 100 * (chat - fresh), len(both)))
a = sum(1 for _, nr, d, _ in rows if nr and d > 0); n1 = sum(1 for _, nr, _, _ in rows if nr)
b = sum(1 for _, nr, d, _ in rows if not nr and d > 0); n2 = len(rows) - n1
K, N = a + b, len(rows)
p = sum(comb(n1, x) * comb(n2, K - x) for x in range(a, min(n1, K) + 1)) / comb(N, K)  # P(at least a of the K in group 1)
for m, nr, d, n in sorted(rows, key=lambda r: (not r[1], -r[2])):
    print(f"{m:45s} {'no hidden reasoning' if nr else 'hidden reasoning':20s} {d:+6.1f} points (n={n})")
print(f"\ngave in more inside the chat: {a} of {n1} without hidden reasoning, {b} of {n2} with it; one-sided Fisher p = {p:.4f}")
