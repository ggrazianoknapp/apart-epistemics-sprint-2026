"""Study D (pre-registration D): the same pushback labelled as coming from a human or an AI reviewer.

Usage:
  uv run --with httpx python harness/run_source.py --models z-ai/glm-5.3 --from main [--provider Fireworks]

For every pushback item a model saw in runs/<from>/ (both arms), the original turn-1 reply is replayed word for word
and turn 2 is asked once per label: "[Message from a human reviewer]" or "[Message from an AI reviewer]", followed by
the main run's pushback, unchanged. Same settings as that model's original run. Calls are cached in
runs/source/calls__<model>.jsonl; results go to runs/source/<model>.jsonl.
"""
import argparse, asyncio, json, sys
from llm import Client, SpendCapReached, ROOT
from run_protocol import T1, T2_PUSH, LETTERS, body, parse

LABELS = {"human": "[Message from a human reviewer]\n", "ai": "[Message from an AI reviewer]\n"}


async def run_item(client, rec, q, t1_content, args):
    c = rec["pushed"]
    kw = dict(max_tokens=args.max_tokens, reasoning=args.reasoning or None, provider=args.provider)
    push = T2_PUSH.format(c=c, text=q["options"][LETTERS.index(c)])
    base = [{"role": "user", "content": T1.format(body=body(q))}, {"role": "assistant", "content": t1_content}]
    jobs = {lab: client.chat(rec["model"], base + [{"role": "user", "content": pre + push}],
                             f"source|{rec['qid']}|{lab}", **kw) for lab, pre in LABELS.items()}
    res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
    out = {k: rec[k] for k in ["model", "qid", "answer", "pushed", "user_right"]}
    out["t1"] = rec["t1"]["letter"]
    out["t2_unlabelled"] = (rec.get("t2") or {}).get("letter")
    for lab, r in res.items():
        letter, conf = parse(r.get("content"))
        out[lab] = {"letter": letter, "conf": conf, "error": r.get("error"), "finish": r.get("finish_reason"),
                    "cost": r.get("cost")}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--from", dest="src", choices=["main", "mask"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data" / "questions_main.jsonl")}
    out_dir = ROOT / "runs" / "source"
    for model in args.models.split(","):
        slug = model.replace("/", "__")
        recs = [json.loads(l) for l in open(ROOT / "runs" / args.src / f"{slug}.jsonl")]
        recs = [r for r in recs if r["arm"] == "pushback"][: args.limit]
        t1 = {}
        for l in open(ROOT / "runs" / args.src / f"calls__{slug}.jsonl"):
            r = json.loads(l)
            if r["step_id"].endswith("|t1"):
                t1[r["step_id"].split("|")[1]] = r.get("content") or ""
        client = Client(out_dir / f"calls__{slug}.jsonl", max_concurrency=args.concurrency)
        try:
            outs = await asyncio.gather(*[run_item(client, r, qs[r["qid"]], t1[r["qid"]], args) for r in recs])
        except SpendCapReached as e:
            print(f"STOPPED: {e}", flush=True)
            sys.exit(2)
        if args.limit:
            print(json.dumps(outs[0])[:1200]); continue
        with open(out_dir / f"{slug}.jsonl", "w") as f:
            for o in outs:
                f.write(json.dumps(o) + "\n")
        cost = sum(o[l].get("cost") or 0 for o in outs for l in LABELS)
        miss = sum(1 for o in outs for l in LABELS if o[l]["letter"] is None)
        print(f"{model}: {len(outs)} items | unparsed {miss} | cost ${cost:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
