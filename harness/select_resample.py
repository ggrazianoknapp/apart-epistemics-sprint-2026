"""Pick the items for study C1 (pre-registration C): every given-up item plus confidence-matched kept items.

For each model in runs/main: "given up" = user-wrong pushback item, turn-1 answer correct, turn-2 answer = the user's
option. "Kept" = same, but the turn-2 answer stayed correct. Each given-up item gets the kept item from the same model
with the nearest turn-1 confidence (ties broken by a hash of the question id), without replacement, going round the
given-up items until max(2 x given up, 40) kept items are picked (or all of them). Kept items carry a
weight (all kept items for the model / kept items sampled) so population estimates cover all user-wrong items.
Writes data/resample_items.jsonl.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = lambda r, s: (r.get(s) or {}).get("letter")
h = lambda qid: hashlib.sha1(qid.encode()).hexdigest()

out = []
for f in sorted((ROOT / "runs" / "main").glob("*.jsonl")):
    if f.name.startswith(("calls__", "extract")):
        continue
    recs = [json.loads(l) for l in f.open()]
    base = [r for r in recs if r["arm"] == "pushback" and not r["user_right"] and L(r, "t1") == r["answer"]]
    given = [r for r in base if L(r, "t2") == r["pushed"]]
    kept = [r for r in base if L(r, "t2") == r["answer"]]
    pool = sorted(kept, key=lambda r: h(r["qid"]))
    picked, target = [], min(max(2 * len(given), 40), len(kept))
    while len(picked) < target:
        for g in sorted(given, key=lambda r: h(r["qid"])):
            if len(picked) == target:
                break
            c = g["t1"]["conf"] if g["t1"]["conf"] is not None else -1
            best = min(pool, key=lambda r: (abs((r["t1"]["conf"] if r["t1"]["conf"] is not None else -1) - c), h(r["qid"])))
            pool.remove(best)
            picked.append(best)
    w = len(kept) / len(picked) if picked else 0
    for r, kind, weight in [(r, "given_up", 1.0) for r in given] + [(r, "kept", w) for r in picked]:
        out.append({"model": r["model"], "qid": r["qid"], "set": kind, "weight": weight, "answer": r["answer"],
                    "pushed": r["pushed"], "t1_conf": r["t1"]["conf"]})
    print(f"{recs[0]['model']:38s} given up {len(given):3d} | kept sampled {len(picked):3d} of {len(kept):3d}")
with open(ROOT / "data" / "resample_items.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(len(out), "items")
