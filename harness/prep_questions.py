"""Build the fixed question sets: 300 main (225 SuperGPQA + 75 MMLU-Pro) and 50 pilot, all 10-option, seeded.

Writes data/questions_main.jsonl and data/questions_pilot.jsonl. Never filters on any model's accuracy.
"""
import json, random
from pathlib import Path
from datasets import load_dataset

SEED = 20261113
OUT = Path(__file__).resolve().parent.parent / "data"
LETTERS = "ABCDEFGHIJ"


def norm(qid, source, question, options, answer_idx, meta):
    return {"qid": qid, "source": source, "question": question.strip(),
            "options": [o.strip() for o in options], "answer": LETTERS[answer_idx], **meta}


def supergpqa():
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    rows = []
    for r in ds:
        if len(r["options"]) != 10:
            continue
        rows.append(norm(f"sg-{r['uuid']}", "SuperGPQA", r["question"], r["options"],
                         LETTERS.index(r["answer_letter"]),
                         {"difficulty": r["difficulty"], "discipline": r["discipline"], "field": r["field"]}))
    return rows


def mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    for r in ds:
        if len(r["options"]) != 10:
            continue
        rows.append(norm(f"mp-{r['question_id']}", "MMLU-Pro", r["question"], r["options"], r["answer_index"],
                         {"difficulty": "mmlu", "discipline": r["category"], "field": r["category"]}))
    return rows


def stratified(rows, n, key, rng):
    """Sample n rows spread evenly over the values of `key`."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n:
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main():
    rng = random.Random(SEED)
    sg, mp = supergpqa(), mmlu_pro()
    sg_hard = [r for r in sg if r["difficulty"] == "hard"]
    sg_mid = [r for r in sg if r["difficulty"] == "middle"]
    print(f"SuperGPQA 10-option: {len(sg)} (hard {len(sg_hard)}, middle {len(sg_mid)}); MMLU-Pro 10-option: {len(mp)}")

    pool_hard = stratified(sg_hard, 150 + 25, "discipline", rng)
    pool_mid = stratified(sg_mid, 75 + 13, "discipline", rng)
    pool_mp = stratified(mp, 75 + 12, "discipline", rng)
    main_set = pool_hard[:150] + pool_mid[:75] + pool_mp[:75]
    pilot_set = pool_hard[150:] + pool_mid[75:] + pool_mp[75:]
    assert not {r["qid"] for r in main_set} & {r["qid"] for r in pilot_set}

    OUT.mkdir(exist_ok=True)
    for name, rows in [("questions_main", main_set), ("questions_pilot", pilot_set)]:
        rng.shuffle(rows)
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
