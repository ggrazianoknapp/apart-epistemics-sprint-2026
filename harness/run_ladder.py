"""Study G4 (pre-registration G): the same in-chat pushback at several reasoning efforts, one session.

Usage: uv run --with httpx python harness/run_ladder.py --models openai/gpt-6-luna --efforts none,low,medium,high
For every main-run pushback item (both arms), turn 1 is replayed word for word and the pushback is asked once per
effort. Step ids start with "ladder|", so nothing is reused from Study E. Results go to runs/ladder/<model>.jsonl.
"""
import argparse, asyncio, json, sys
from llm import Client, SpendCapReached, ROOT
from run_protocol import T1, T2_PUSH, LETTERS, body, parse


async def run_item(client, rec, q, t1_content, efforts, args):
    c = rec["pushed"]
    msgs = [{"role": "user", "content": T1.format(body=body(q))}, {"role": "assistant", "content": t1_content},
            {"role": "user", "content": T2_PUSH.format(c=c, text=q["options"][LETTERS.index(c)])}]
    jobs = {e: client.chat(rec["model"], msgs, f"ladder|{rec['qid']}|{e}", max_tokens=args.max_tokens, reasoning=e,
                           provider=args.provider) for e in efforts}
    res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
    out = {k: rec[k] for k in ["model", "qid", "answer", "pushed", "user_right"]}
    out["t1"] = rec["t1"]["letter"]
    for e, r in res.items():
        letter, conf = parse(r.get("content"))
        out[e] = {"letter": letter, "conf": conf, "finish": r.get("finish_reason"),
                  "reasoning_tokens": r.get("reasoning_tokens"), "cost": r.get("cost")}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--efforts", default="none,low,medium,high")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    efforts = args.efforts.split(",")
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data" / "questions_main.jsonl")}
    out_dir = ROOT / "runs" / "ladder"
    for model in args.models.split(","):
        slug = model.replace("/", "__")
        recs = [r for r in map(json.loads, open(ROOT / "runs" / "main" / f"{slug}.jsonl")) if r["arm"] == "pushback"][: args.limit]
        t1 = {r["step_id"].split("|")[1]: r.get("content") or "" for r in map(json.loads, open(ROOT / "runs" / "main" / f"calls__{slug}.jsonl"))
              if r["step_id"].endswith("|t1")}
        client = Client(out_dir / f"calls__{slug}.jsonl", max_concurrency=args.concurrency)
        try:
            outs = await asyncio.gather(*[run_item(client, r, qs[r["qid"]], t1[r["qid"]], efforts, args) for r in recs])
        except SpendCapReached as e:
            print(f"STOPPED: {e}", flush=True); sys.exit(2)
        if args.limit:
            for e in efforts:
                print(model, e, "reasoning", [o[e]["reasoning_tokens"] for o in outs], "letters", [o[e]["letter"] for o in outs])
            continue
        with open(out_dir / f"{slug}.jsonl", "w") as f:
            for o in outs:
                f.write(json.dumps(o) + "\n")
        print(f"{model}: {len(outs)} items | cost ${sum(o[e]['cost'] or 0 for o in outs for e in efforts):.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
