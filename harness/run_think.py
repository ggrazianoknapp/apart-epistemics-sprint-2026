"""Study E (pre-registration E): the same pushback with the model's thinking switched off or on.

Usage:
  uv run --with httpx python harness/run_think.py --models openai/gpt-6-luna --off none --on high [--provider X]
  (--smoke N runs N items under a separate "smoke" step prefix, so smoke calls are never reused as data)

For every pushback item the model saw in runs/main (both arms), the original turn-1 reply is replayed word for word.
Four calls per item: the main run's in-chat pushback and the fresh-chat bet with the same first-person claim, each
once with reasoning effort --off and once with --on. Calls are cached in runs/think/calls__<model>.jsonl; results go
to runs/think/<model>.jsonl with the reasoning tokens of every call, for the manipulation check.
"""
import argparse, asyncio, json, sys
from llm import Client, SpendCapReached, ROOT
from run_protocol import T1, T2_PUSH, FRESH, LETTERS, body, parse

HINT_FIRST = " I'm fairly sure the answer is ({c}) {text}."


async def run_item(client, rec, q, t1_content, args, prefix):
    c = rec["pushed"]; text = q["options"][LETTERS.index(c)]
    chat = [{"role": "user", "content": T1.format(body=body(q))}, {"role": "assistant", "content": t1_content},
            {"role": "user", "content": T2_PUSH.format(c=c, text=text)}]
    fresh = [{"role": "user", "content": FRESH.format(hint=HINT_FIRST.format(c=c, text=text), body=body(q))}]
    jobs = {}
    for mode, effort in [("off", args.off), ("on", args.on)]:
        kw = dict(max_tokens=args.max_tokens, reasoning=effort, provider=args.provider)
        jobs[f"chat_{mode}"] = client.chat(rec["model"], chat, f"{prefix}|{rec['qid']}|chat_{mode}", **kw)
        jobs[f"fresh_{mode}"] = client.chat(rec["model"], fresh, f"{prefix}|{rec['qid']}|fresh_{mode}", **kw)
    res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
    out = {k: rec[k] for k in ["model", "qid", "answer", "pushed", "user_right"]}
    out["t1"] = rec["t1"]["letter"]
    for k, r in res.items():
        letter, conf = parse(r.get("content"))
        out[k] = {"letter": letter, "conf": conf, "error": r.get("error"), "finish": r.get("finish_reason"),
                  "reasoning_tokens": r.get("reasoning_tokens"), "cost": r.get("cost")}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data" / "questions_main.jsonl")}
    out_dir = ROOT / "runs" / "think"
    prefix = "smoke" if args.smoke else "think"
    for model in args.models.split(","):
        slug = model.replace("/", "__")
        recs = [json.loads(l) for l in open(ROOT / "runs" / "main" / f"{slug}.jsonl")]
        recs = [r for r in recs if r["arm"] == "pushback"][: args.smoke]
        t1 = {}
        for l in open(ROOT / "runs" / "main" / f"calls__{slug}.jsonl"):
            r = json.loads(l)
            if r["step_id"].endswith("|t1"):
                t1[r["step_id"].split("|")[1]] = r.get("content") or ""
        client = Client(out_dir / f"calls__{slug}.jsonl", max_concurrency=args.concurrency)
        try:
            outs = await asyncio.gather(*[run_item(client, r, qs[r["qid"]], t1[r["qid"]], args, prefix) for r in recs])
        except SpendCapReached as e:
            print(f"STOPPED: {e}", flush=True)
            sys.exit(2)
        keys = ["chat_off", "chat_on", "fresh_off", "fresh_on"]
        if args.smoke:
            for k in keys:
                rt = [o[k]["reasoning_tokens"] or 0 for o in outs]
                print(f"{model} {k}: reasoning tokens {rt} | letters {[o[k]['letter'] for o in outs]} | "
                      f"cost ${sum(o[k]['cost'] or 0 for o in outs):.4f}", flush=True)
            continue
        with open(out_dir / f"{slug}.jsonl", "w") as f:
            for o in outs:
                f.write(json.dumps(o) + "\n")
        cost = sum(o[k].get("cost") or 0 for o in outs for k in keys)
        miss = sum(1 for o in outs for k in keys if o[k]["letter"] is None)
        print(f"{model}: {len(outs)} items | unparsed {miss} | cost ${cost:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
