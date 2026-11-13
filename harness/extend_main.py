"""Add 300 questions to the main set (same 150 hard / 75 middle SuperGPQA / 75 MMLU-Pro mix), excluding every
question already in the main or pilot sets. Decided after the pilot showed that giving in is rare (power)."""
import json, random
from pathlib import Path
from prep_questions import supergpqa, mmlu_pro, stratified, OUT

SEED = 20261114
rng = random.Random(SEED)
used = {json.loads(l)["qid"] for n in ["questions_main", "questions_pilot"] for l in open(OUT / f"{n}.jsonl")}
sg = [r for r in supergpqa() if r["qid"] not in used]
mp = [r for r in mmlu_pro() if r["qid"] not in used]
new = (stratified([r for r in sg if r["difficulty"] == "hard"], 150, "discipline", rng)
       + stratified([r for r in sg if r["difficulty"] == "middle"], 75, "discipline", rng)
       + stratified(mp, 75, "discipline", rng))
rng.shuffle(new)
assert not {r["qid"] for r in new} & used
with open(OUT / "questions_main.jsonl", "a") as f:
    for r in new:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("added", len(new), "-> main set now", sum(1 for _ in open(OUT / "questions_main.jsonl")))
