"""Redo the distractor pick for questions whose first pick was cut off, with a larger token budget."""
import asyncio, json, re
from llm import Client, ROOT
from pick_distractors import PICKER, prompt

calls = [json.loads(l) for l in open(ROOT / "runs" / "distractors_calls.jsonl")]
qs = {}
for name in ["questions_main", "questions_pilot"]:
    for l in open(ROOT / "data" / f"{name}.jsonl"):
        q = json.loads(l); qs[q["qid"]] = q
bad = []
for c in calls:
    qid = c["step_id"].split("|", 1)[1]
    m = re.findall(r"\(([A-J])\)", c.get("content", ""))
    if not m or m[-1] == qs[qid]["answer"]:
        bad.append(qid)
print("to redo:", len(bad))


async def main():
    client = Client(ROOT / "runs" / "distractors_calls.jsonl", max_concurrency=24)
    d = json.loads((ROOT / "data" / "distractors.json").read_text())
    recs = await asyncio.gather(*[client.chat(PICKER, [{"role": "user", "content": prompt(qs[q])}],
                                              f"distractor2|{q}", max_tokens=2000, reasoning=None) for q in bad])
    still = 0
    for q, r in zip(bad, recs):
        m = re.findall(r"\(([A-J])\)", r.get("content", ""))
        if m and m[-1] != qs[q]["answer"]:
            d[q] = m[-1]
        else:
            still += 1
    (ROOT / "data" / "distractors.json").write_text(json.dumps(d, indent=0))
    print("still on fallback:", still)

asyncio.run(main())
