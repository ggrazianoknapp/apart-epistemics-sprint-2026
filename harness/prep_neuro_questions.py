"""600 fresh questions for the neuron experiments (200 to select neurons, 400 held out), disjoint from the main and
pilot sets, same 50/25/25 mix of hard SuperGPQA, middle SuperGPQA and MMLU-Pro. Writes data/questions_neuro.jsonl."""
import json, random
from prep_questions import supergpqa, mmlu_pro, stratified, OUT

rng = random.Random(20261115)
used = {json.loads(l)["qid"] for n in ["questions_main", "questions_pilot"] for l in open(OUT / f"{n}.jsonl")}
sg = [r for r in supergpqa() if r["qid"] not in used]
mp = [r for r in mmlu_pro() if r["qid"] not in used]
rows = (stratified([r for r in sg if r["difficulty"] == "hard"], 300, "discipline", rng)
        + stratified([r for r in sg if r["difficulty"] == "middle"], 150, "discipline", rng)
        + stratified(mp, 150, "discipline", rng))
rng.shuffle(rows)
for i, r in enumerate(rows):
    r["split"] = "select" if i < 200 else "heldout"
with open(OUT / "questions_neuro.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("neuro questions:", len(rows))
