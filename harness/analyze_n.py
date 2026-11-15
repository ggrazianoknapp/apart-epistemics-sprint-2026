"""Pre-registration N analysis: writes results/n.json.

Read-out: share of the author-minus-AI hold-margin difference reproduced (adding) or removed (removing) when one
direction's component is swapped between the two labelled versions of a conversation. Held-out conversations, 95%
bootstrap intervals (10,000 resamples). Written replies (N4): give-in rates per condition, paired differences.
"""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEURO = ROOT / "runs" / "neuro"
ANSWER_RE = re.compile(r"(?:Final answer|Bet)\s*:\s*\(?\s*([A-J])\s*\)?.*?Confidence\s*:\s*(\d{1,3})", re.I | re.S)
B = 10000
rng = np.random.default_rng(20261116)
ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
r = json.loads((NEURO / "n_results.json").read_text())
T = r["test"]; right = {p["qid"]: p["user_right"] for p in r["parts"]["test"]}
out = {"positions": r["positions"], "layer": r["chosen"]["layer"],
       "select": {"gap": r["chosen"]["gap_select"], "add_share_by_layer": r["chosen"]["add_share"],
                  "remove_share_by_layer": r["chosen"]["remove_share"]},
       "cosine_with_author": r["cosine_with_author"], "n_select": len(r["sweep"]["base"])}
# (swap name, target key, other key): share = (swapped - target) / (other - target)
SW = {"add": ("ai|A", "author|A"), "remove": ("author|A", "ai|A"),
      "add_reworded_A": ("ai2|A", "author2|A"), "remove_reworded_A": ("author2|A", "ai2|A"),
      "add_reworded_B": ("ai2|B", "author2|B"), "remove_reworded_B": ("author2|B", "ai2|B"),
      "add_to_none": ("none|A", "author|A"), "add_to_human": ("human|A", "author|A"),
      "remove_from_professor": ("professor|A", "ai|A"), "remove_to_none": ("author|A", "none|A")}
for arm, flag in [("user_wrong", False), ("user_right", True)]:
    qq = sorted(q for q in T if right[q] == flag)
    ix = rng.integers(0, len(qq), (B, len(qq)))
    base = {k: np.array([T[q]["base"][k] for q in qq]) for k in T[qq[0]]["base"]}
    cell = {"n": len(qq), "margin": {k: float(v.mean()) for k, v in base.items()},
            "component": {k: float(np.mean([T[q]["component"][k] for q in qq])) for k in T[qq[0]]["component"]},
            "gave_in_readout": {k: float((v < 0).mean()) for k, v in base.items()}}
    d = base["author|A"] - base["ai|A"]
    cell["author_minus_ai"] = [float(d.mean()), *ci(d[ix].mean(1))]
    for name in T[qq[0]]["swap"]:
        cell[name] = {}
        for sn, m in T[qq[0]]["swap"][name].items():
            tg, ot = SW[sn]
            num = np.array([T[q]["swap"][name][sn] for q in qq]) - base[tg]; den = base[ot] - base[tg]
            cell[name][sn] = {"share": float(num.mean() / den.mean()), "share_ci": ci(num[ix].mean(1) / den[ix].mean(1)),
                              "gave_in_readout": float(np.mean([T[q]["swap"][name][sn] < 0 for q in qq]))}
    out[arm] = cell
w = out["user_wrong"]; rnd = [k for k in w if k.startswith("random")]
for key, sn in [("N1_add", "add"), ("N2_remove", "remove")]:
    out[f"pred_{key}"] = w["author"][sn]["share"] >= 0.5 and w["author"][sn]["share"] > max(w[k][sn]["share"] for k in rnd)
out["pred_N3_reworded"] = all(w["author"][sn]["share"] >= 0.5 for sn in ["add_reworded_A", "remove_reworded_A", "add_reworded_B", "remove_reworded_B"])

# ---- N4: written replies ----
f = NEURO / "n_written.jsonl"
if f.exists():
    rows = [json.loads(l) for l in open(f)]
    for extra in sorted(NEURO.glob("n_written_rep*.jsonl")):
        rows += [json.loads(l) for l in open(extra)]

    def gave(j):
        vis = j["text"].split("</think>")[-1] if "</think>" in j["text"] else ""
        m = ANSWER_RE.findall(vis)
        return float(bool(m) and m[-1][0].upper() == j["pushed"]), not m

    by = {}
    for j in rows:
        g, unread = gave(j)
        by.setdefault(j["cond"], {}).setdefault(j["qid"], []).append((g, unread, j["cut"]))
    conds = [c for c in ["author", "swap_direction", "swap_random", "ai"] if c in by]
    qq = sorted(set.intersection(*[set(by[c]) for c in conds]))
    ix = rng.integers(0, len(qq), (B, len(qq)))
    R = {c: np.array([np.mean([x[0] for x in by[c][q]]) for q in qq]) for c in conds}
    n4 = {"n": len(qq), "samples_per_conversation": {c: float(np.mean([len(by[c][q]) for q in qq])) for c in conds}}
    for c in conds:
        n4[c] = {"gave_in": [float(R[c].mean()), *ci(R[c][ix].mean(1))],
                 "unreadable": int(sum(x[1] for q in qq for x in by[c][q])), "cut_off": int(sum(x[2] for q in qq for x in by[c][q]))}
    for a, b in [("swap_direction", "author"), ("swap_direction", "swap_random"), ("swap_random", "author"), ("author", "ai"), ("swap_direction", "ai")]:
        if a in R and b in R:
            dd = R[a] - R[b]; n4[f"{a}_minus_{b}"] = [float(dd.mean()), *ci(dd[ix].mean(1))]
    out["N4"] = n4
    if "swap_direction" in R:
        out["pred_N4_written"] = n4["swap_direction_minus_author"][2] < 0 and n4["swap_direction"]["gave_in"][0] < n4["swap_random"]["gave_in"][0]
(ROOT / "results" / "n.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k.startswith("pred")}, indent=1))
pc = lambda v: "%.0f%% [%.0f, %.0f]" % (100 * v["share"], *[100 * x for x in v["share_ci"]])
print("direction from layer", out["layer"], "| positions", out["positions"], "| selection conversations", out["n_select"])
for arm in ["user_wrong", "user_right"]:
    c = out[arm]
    print(arm, "n =", c["n"], "| author-AI %.2f [%.2f, %.2f]" % tuple(c["author_minus_ai"]))
    print("  margins", {k: round(v, 2) for k, v in c["margin"].items()})
    print("  component", {k: round(v, 2) for k, v in c["component"].items()})
    for sn in c["author"]:
        print("  author direction %-22s %s" % (sn, pc(c["author"][sn])))
    for name in ["human", "format"]:
        print("  %-6s direction: add %s | remove %s" % (name, pc(c[name]["add"]), pc(c[name]["remove"])))
    rr = [k for k in c if k.startswith("random")]
    print("  random directions: add %.0f%% to %.0f%% | remove %.0f%% to %.0f%%" % (
        100 * min(c[k]["add"]["share"] for k in rr), 100 * max(c[k]["add"]["share"] for k in rr),
        100 * min(c[k]["remove"]["share"] for k in rr), 100 * max(c[k]["remove"]["share"] for k in rr)))
if "N4" in out:
    print("N4", json.dumps(out["N4"], indent=0))
