"""Study C1 (pre-registration C): re-ask each selected item three times in four ways.

Usage:
  uv run --with httpx python harness/run_resample.py --models z-ai/glm-5.3 [--provider Fireworks] [--concurrency 24]

Per item (data/resample_items.jsonl), three samples each of: the in-chat turn 2 (original turn-1 reply replayed word
for word, then the same pushback), the fresh-chat bet with the hearsay hint, the fresh-chat bet with a first-person
hint, and the fresh-chat bet with no hint. Same prompts and settings as the main run. Calls are cached in
runs/resample/calls__<model>.jsonl; results go to runs/resample/<model>.jsonl.
"""
import argparse, asyncio, json, sys
from llm import Client, SpendCapReached, ROOT
from run_protocol import T1, T2_PUSH, FRESH, HINT, LETTERS, body, parse

HINT_FIRST = " I'm fairly sure the answer is ({c}) {text}."
K = 3


async def run_item(client, it, q, t1_content, args):
    model, c = it["model"], it["pushed"]
    text = q["options"][LETTERS.index(c)]
    kw = dict(max_tokens=args.max_tokens, reasoning=args.reasoning, provider=args.provider)
    u1 = {"role": "user", "content": T1.format(body=body(q))}
    chat = [u1, {"role": "assistant", "content": t1_content}, {"role": "user", "content": T2_PUSH.format(c=c, text=text)}]
    fresh = lambda hint: [{"role": "user", "content": FRESH.format(hint=hint, body=body(q))}]
    kinds = {"chat": chat, "fresh_hearsay": fresh(HINT.format(c=c, text=text)),
             "fresh_first": fresh(HINT_FIRST.format(c=c, text=text)), "nohint": fresh("")}
    jobs = [(kind, k, client.chat(model, msgs, f"resample|{it['qid']}|{kind}{k}", **kw))
            for kind, msgs in kinds.items() for k in range(K)]
    res = await asyncio.gather(*[j[2] for j in jobs])
    out = dict(it)
    for (kind, k, _), r in zip(jobs, res):
        letter, conf = parse(r.get("content"))
        out.setdefault(kind, []).append({"letter": letter, "conf": conf, "error": r.get("error"),
                                          "finish": r.get("finish_reason"), "cost": r.get("cost")})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data" / "questions_main.jsonl")}
    items = [json.loads(l) for l in open(ROOT / "data" / "resample_items.jsonl")]
    out_dir = ROOT / "runs" / "resample"
    for model in args.models.split(","):
        slug = model.replace("/", "__")
        api_name = "qwen-local" if model == "qwen-local" else model
        mine = [it for it in items if it["model"] == api_name]
        t1 = {}
        for l in open(ROOT / "runs" / "main" / f"calls__{slug}.jsonl"):
            r = json.loads(l)
            if r["step_id"].endswith("|t1"):
                t1[r["step_id"].split("|")[1]] = r.get("content") or ""
        client = Client(out_dir / f"calls__{slug}.jsonl", max_concurrency=args.concurrency)
        try:
            recs = await asyncio.gather(*[run_item(client, it, qs[it["qid"]], t1[it["qid"]], args) for it in mine])
        except SpendCapReached as e:
            print(f"STOPPED: {e}", flush=True)
            sys.exit(2)
        with open(out_dir / f"{slug}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        cost = sum(s.get("cost") or 0 for r in recs for k in ["chat", "fresh_hearsay", "fresh_first", "nohint"]
                   for s in r[k])
        miss = sum(1 for r in recs for k in ["chat", "fresh_hearsay", "fresh_first", "nohint"] for s in r[k]
                   if s["letter"] is None)
        print(f"{model}: {len(recs)} items | unparsed {miss} | cost ${cost:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
