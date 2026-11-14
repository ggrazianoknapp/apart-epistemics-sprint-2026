"""Pre-registration L analysis: writes results/l.json (or l_analysis_test.json next to a test file).

Share of the author-minus-AI hold-margin difference removed = 1 - edited difference / unedited difference, with 95%
bootstrap intervals over questions (user-wrong arm unless stated).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None
r = json.loads(((TESTDIR / "l_test.json") if TESTDIR else ROOT / "runs" / "neuro" / "l_results.json").read_text())
B = 10000
rng = np.random.default_rng(20261117)
ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
out = {}


def share(base, edit, keys):
    """base/edit: dict key -> (ai margin, author margin). Returns share removed with interval, and the effects."""
    b = np.array([base[k][1] - base[k][0] for k in keys]); e = np.array([edit[k][1] - edit[k][0] for k in keys])
    ix = rng.integers(0, len(keys), (B, len(keys)))
    return {"share": float(1 - e.mean() / b.mean()), "share_ci": ci(1 - e[ix].mean(1) / b[ix].mean(1)),
            "effect_base": float(b.mean()), "effect_edit": float(e.mean())}


# ---- L1 / L2 ----
L12 = r["L12"]; keys = sorted(L12["baseline"])
pair = lambda name: {k: (L12[name][k]["ai"], L12[name][k]["author"]) for k in keys}
base = pair("baseline")
l12 = {name: share(base, pair(name), keys) for name in L12 if name != "baseline"}
out["L1"] = {"attr25": l12["attr25"], "attr100": l12["attr100"],
             "rand25": sorted(v["share"] for k, v in l12.items() if k.startswith("rand25_")),
             "rand100": sorted(v["share"] for k, v in l12.items() if k.startswith("rand100_"))}
out["pred_L1_attr25_beyond_all_20"] = l12["attr25"]["share"] > max(out["L1"]["rand25"])
out["pred_L1_attr100_beyond_all_20"] = l12["attr100"]["share"] > max(out["L1"]["rand100"])
out["L2"] = {"short25": [l12[f"short25_{j}"]["share"] for j in range(5) if f"short25_{j}" in l12],
             "short100": [l12[f"short100_{j}"]["share"] for j in range(5) if f"short100_{j}" in l12]}

# ---- L3 ----
L3 = r["L3"]; out["L3"] = {}
for arm in ["user_wrong", "user_right"]:
    ks = sorted({k.rsplit("|", 1)[0] for k in L3["baseline"] if f"|{arm}|" in k})
    get = lambda d: {k: (d[f"{k}|ai"], d[f"{k}|author"]) for k in ks}
    b3 = get(L3["baseline"])
    out["L3"][arm] = {g: share(b3, get(L3[g]), ks) for g in ["attr25_free", "attr25_frozen", "attr100_free", "attr100_frozen"]}
out["L3"]["replay_noedit_max_abs_diff"] = float(max(abs(L3["replay_noedit"][k] - L3["baseline"][k]) for k in L3["baseline"]))
w = out["L3"]["user_wrong"]
out["pred_L3_attr25_frozen_keeps_half"] = w["attr25_frozen"]["share"] >= 0.5 * w["attr25_free"]["share"]
out["pred_L3_attr100_frozen_keeps_half"] = w["attr100_frozen"]["share"] >= 0.5 * w["attr100_free"]["share"]

# ---- L4 ----
L4 = r["L4"]; meta_known = set(keys)
qk = sorted(q for q in L4["baseline"] if q in meta_known); qa = sorted(L4["baseline"])
out["L4"] = {}
for name in L4:
    if name == "baseline":
        continue
    dk = np.array([L4[name][q] - L4["baseline"][q] for q in qk]); da = np.array([L4[name][q] - L4["baseline"][q] for q in qa])
    out["L4"][name] = {"change_known": [float(dk.mean()), *ci(dk[rng.integers(0, len(qk), (B, len(qk)))].mean(1))],
                       "change_all": [float(da.mean()), *ci(da[rng.integers(0, len(qa), (B, len(qa)))].mean(1))]}
out["pred_L4_attr25_less_than_half_of_0.32"] = abs(out["L4"]["attr25"]["change_known"][0]) < 0.16

# ---- L5 ----
L5 = r["L5"]; meta = {m["qid"]: m for m in r["L5_meta"]}
uw = sorted(q for q in L5["baseline"] if not meta[q]["user_right"]); ur = sorted(q for q in L5["baseline"] if meta[q]["user_right"])
out["L5"] = {"n_user_wrong": len(uw), "n_user_right": len(ur)}
for arm, qq in [("user_wrong", uw), ("user_right", ur)]:
    if not qq:
        continue
    ix = rng.integers(0, len(qq), (B, len(qq)))
    au = np.array([L5["baseline"][q]["author"] - L5["baseline"][q]["ai"] for q in qq])
    na = np.array([L5["baseline"][q]["none"] - L5["baseline"][q]["ai"] for q in qq])
    cell = {"author_minus_ai": [float(au.mean()), *ci(au[ix].mean(1))], "none_minus_ai": [float(na.mean()), *ci(na[ix].mean(1))],
            "gave_in": {lab: float(np.mean([L5["baseline"][q][lab] < 0 for q in qq])) for lab in ["none", "ai", "author"]}}
    pb = {q: (L5["baseline"][q]["ai"], L5["baseline"][q]["author"]) for q in qq}
    for name in L5:
        if name != "baseline":
            cell[name] = share(pb, {q: (L5[name][q]["ai"], L5[name][q]["author"]) for q in qq}, qq)
    out["L5"][arm] = cell
c = out["L5"].get("user_wrong", {})
out["pred_L5a_author_lowers_margin"] = bool(c) and c["author_minus_ai"][2] < 0
out["pred_L5b_attr25_removes_half_beyond_random"] = bool(out["pred_L5a_author_lowers_margin"]) and c["attr25"]["share"] >= 0.5 and \
    c["attr25"]["share"] > max(c[f"rand25_{j}"]["share"] for j in range(4) if f"rand25_{j}" in c)
out["headline_rule_met"] = all([out["pred_L1_attr25_beyond_all_20"], out["pred_L1_attr100_beyond_all_20"],
                                out["pred_L3_attr25_frozen_keeps_half"], out["pred_L3_attr100_frozen_keeps_half"],
                                out["pred_L5a_author_lowers_margin"], out["pred_L5b_attr25_removes_half_beyond_random"]])
dest = (TESTDIR / "l_analysis_test.json") if TESTDIR else ROOT / "results" / "l.json"
dest.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k.startswith("pred") or k == "headline_rule_met"}, indent=1))
print("L1 attr25 %.2f attr100 %.2f | rand25 max %.2f rand100 max %.2f" % (out["L1"]["attr25"]["share"], out["L1"]["attr100"]["share"],
      max(out["L1"]["rand25"]), max(out["L1"]["rand100"])))
print("L2", {k: [round(x, 2) for x in v] for k, v in out["L2"].items()})
print("L3", {g: round(v["share"], 2) for g, v in w.items()})
print("L4", {k: [round(x, 3) for x in v["change_known"]] for k, v in out["L4"].items()})
if c:
    print("L5 uw", {k: (v if not isinstance(v, dict) else round(v.get("share", 0), 2)) for k, v in c.items()})
