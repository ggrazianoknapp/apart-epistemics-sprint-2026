"""Manipulation check for E3 (pre-registration B): does k (vs k1) change the mean 'Wait' count, paired bootstrap?
Usage: python3 e3_check.py k4 k0 ...   Prints PASS if any setting's 95% interval excludes 0."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "runs" / "e3"
base = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "k1.jsonl")}
rng = random.Random(0)
passed = False
for s in sys.argv[1:]:
    other = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / f"{s}.jsonl")}
    qids = sorted(set(base) & set(other))
    d = [other[q]["waits"] - base[q]["waits"] for q in qids]
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
    lo, hi = boots[250], boots[9749]
    ok = lo > 0 or hi < 0
    passed |= ok
    print(f"{s} vs k1: mean Wait change {sum(d)/len(d):+.2f} (95% CI {lo:+.2f} to {hi:+.2f}) over {len(d)} conversations -> {'moves' if ok else 'no change'}")
print("PASS" if passed else "FAIL")
