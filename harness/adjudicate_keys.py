"""Study G2 (pre-registration G): blind check of contested answer keys.

Contested = main-set questions where at least 6 of the 9 main-panel models chose the same option at turn 1 and it is
not the key. Two checkers outside the main panel see the question with only the key and the consensus option, in random
order (seeded), and answer 1, 2, BOTH or NEITHER. Writes results/keys.json with a verdict per question:
doubtful (at least one picks the consensus, neither picks the key), ambiguous (they disagree, or either says both or
neither), confirmed (both pick the key).
Usage: uv run --with httpx python harness/adjudicate_keys.py [--limit N]
"""
import argparse, asyncio, json, random, re
from collections import Counter
from llm import Client, ROOT
from run_protocol import LETTERS

CHECKERS = ["anthropic/claude-opus-4.6", "google/gemini-3.1-pro-preview"]
PROMPT = ("Here is a question with two candidate answers. Work out which candidate is correct. If both are equally "
          "correct, say BOTH; if neither is correct, say NEITHER.\n\nQuestion: {q}\n\nCandidate 1: {a}\nCandidate 2: {b}\n\n"
          "End your reply with a final line in exactly this format:\nVerdict: 1, 2, BOTH or NEITHER")
MAIN = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6.1-sol", "openai/gpt-6-luna",
        "google/gemini-4-argon", "x-ai/grok-4.7", "deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3", "qwen-local"]


def contested():
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data" / "questions_main.jsonl")}
    votes = {}
    for m in MAIN:
        for r in map(json.loads, open(ROOT / "runs" / "main" / (m.replace("/", "__") + ".jsonl"))):
            l = (r.get("t1") or {}).get("letter")
            if l:
                votes.setdefault(r["qid"], []).append(l)
    out = []
    for qid, v in votes.items():
        top, n = Counter(v).most_common(1)[0]
        if n >= 6 and top != qs[qid]["answer"]:
            out.append((qid, top))
    return qs, sorted(out)


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); args = ap.parse_args()
    qs, items = contested(); items = items[: args.limit]
    print(len(items), "contested questions", flush=True)
    rng = random.Random(20261114)
    client = Client(ROOT / "runs" / "keys" / "calls__checkers.jsonl", max_concurrency=16)
    jobs, meta = [], []
    for qid, cons in items:
        q = qs[qid]; key = q["answer"]
        pair = [("key", key), ("consensus", cons)]; rng.shuffle(pair)
        prompt = PROMPT.format(q=q["question"], a=q["options"][LETTERS.index(pair[0][1])], b=q["options"][LETTERS.index(pair[1][1])])
        for ck in CHECKERS:
            jobs.append(client.chat(ck, [{"role": "user", "content": prompt}], f"keys|{qid}|{ck}", max_tokens=16000, reasoning="high"))
            meta.append((qid, ck, pair))
    res = await asyncio.gather(*jobs)
    per = {}
    for (qid, ck, pair), r in zip(meta, res):
        m = re.findall(r"Verdict:\s*\**\s*(1|2|BOTH|NEITHER)", r.get("content") or "", re.I)
        v = m[-1].upper() if m else None
        pick = pair[int(v) - 1][0] if v in ("1", "2") else (v.lower() if v else None)
        per.setdefault(qid, {})[ck] = pick
    verdicts = {}
    for qid, cons in items:
        p = list(per.get(qid, {}).values())
        if p and all(x == "key" for x in p) and len(p) == len(CHECKERS):
            v = "confirmed"
        elif "consensus" in p and "key" not in p and not any(x in ("both", "neither", None) for x in p):
            v = "doubtful"
        else:
            v = "ambiguous"
        verdicts[qid] = {"key": qs[qid]["answer"], "consensus": cons, "checkers": per.get(qid, {}), "verdict": v}
    out = {"n_contested": len(items), "counts": Counter(v["verdict"] for v in verdicts.values()), "questions": verdicts}
    if not args.limit:
        (ROOT / "results" / "keys.json").write_text(json.dumps(out, indent=1))
    print(out["counts"], flush=True)
    if args.limit:
        print(json.dumps(verdicts, indent=1)[:1500])


if __name__ == "__main__":
    asyncio.run(main())
