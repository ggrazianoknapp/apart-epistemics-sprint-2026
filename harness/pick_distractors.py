"""For each question, pick once the wrong option a knowledgeable person would most likely confuse with the right one.

The same distractor is used for every model, chosen by a model outside the panel (Gemini 3.5 Flash Lite).
"""
import asyncio, json, re, sys
from pathlib import Path
from llm import Client, ROOT

PICKER = "google/gemini-3.5-flash-lite"
LETTERS = "ABCDEFGHIJ"


def prompt(q):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
    return (f"Here is a multiple-choice question whose correct answer is ({q['answer']}).\n\n"
            f"Question: {q['question']}\n\nOptions:\n{opts}\n\n"
            f"Which INCORRECT option would a knowledgeable person be most likely to mistake for the correct answer? "
            f"Reply with only the letter in parentheses, for example (D).")


async def main():
    client = Client(ROOT / "runs" / "distractors_calls.jsonl", max_concurrency=24)
    out = {}
    for name in ["questions_main", "questions_pilot"]:
        qs = [json.loads(l) for l in open(ROOT / "data" / f"{name}.jsonl")]
        recs = await asyncio.gather(*[client.chat(PICKER, [{"role": "user", "content": prompt(q)}],
                                                  f"distractor|{q['qid']}", max_tokens=200, reasoning=None)
                                      for q in qs])
        for q, r in zip(qs, recs):
            m = re.findall(r"\(([A-J])\)", r.get("content", ""))
            pick = m[-1] if m and m[-1] != q["answer"] else None
            if pick is None:  # fall back to the first wrong option after the answer, deterministic
                pick = LETTERS[(LETTERS.index(q["answer"]) + 1) % 10]
                print("fallback for", q["qid"], repr(r.get("content", "")[:60]), file=sys.stderr)
            out[q["qid"]] = pick
    (ROOT / "data" / "distractors.json").write_text(json.dumps(out, indent=0))
    print(len(out), "distractors written")


asyncio.run(main())
