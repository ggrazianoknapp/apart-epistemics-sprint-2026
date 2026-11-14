"""Pre-registration I analysis (same measures as H): writes results/i.json.

H1 (held-out known questions, no label): for each removed group, change in threshold c, sensitivity d' and pair order
against the baseline, with 95% paired bootstrap intervals over questions, and the range of the matched random groups.
H2 (all held-out questions, AI vs author labels): the authority effect (c with the author label minus c with the AI
label) in each condition, its reduction against the baseline with a paired bootstrap interval, and the random range.
"""
import json, sys
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
r = json.loads((Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "neuro" / "i_results.json").read_text())
zv = np.vectorize(NormalDist().inv_cdf)
rng = np.random.default_rng(20261114)
B = 10000


def margins(cond, label=None, known_only=False):
    rows = [x for x in r["conditions"][cond]["rows"] if x["label"] == label and (x["known"] or not known_only)]
    q = sorted({x["qid"] for x in rows}); qi = {k: i for i, k in enumerate(q)}
    M = np.full((len(q), 2), np.nan); O = np.zeros((len(q), 2))
    for x in rows:
        a = 0 if x["cond"] == "user_wrong" else 1
        M[qi[x["qid"]], a] = x["margin"]; O[qi[x["qid"]], a] = x["top_other"]
    return M, O


def stats(M):
    n = M.shape[-2]; g = (M[..., 0] < 0).mean(-1); a = (M[..., 1] < 0).mean(-1)
    H = (a * n + 0.5) / (n + 1); F = (g * n + 0.5) / (n + 1)
    return zv(H) - zv(F), -(zv(H) + zv(F)) / 2, (M[..., 0] > M[..., 1]).mean(-1)


ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
out = {"groups": r["groups"], "damage": r.get("damage")}

# ---- H1 ----
Mb, Ob = margins("I2|baseline", None, True); n = len(Mb); idx = rng.integers(0, n, (B, n))
bd, bc, bp = stats(Mb[idx]); d0, c0, p0 = stats(Mb)
h1 = {"baseline": {"d": float(d0), "c": float(c0), "pair_order": float(p0), "n_questions": n}, "conditions": {}}
for cond in [k for k in r["conditions"] if k.startswith("I2|") and k != "I2|baseline"]:
    M, O = margins(cond, None, True); d, c, p = stats(M); ed, ec, ep = stats(M[idx])
    h1["conditions"][cond.split("|",1)[1]] = {"change_c": [float(c - c0), *ci(ec - bc)], "change_d": [float(d - d0), *ci(ed - bd)],
                                  "change_pair_order": [float(p - p0), *ci(ep - bp)], "change_off_target": float(O.mean() - Ob.mean())}
C = h1["conditions"]
rs = [C[k] for k in C if k.startswith("rand_stubborn")]; rj = [C[k] for k in C if k.startswith("rand_judge")]
if "stubborn25" in C and rs and rj:
    h1["pred_stubborn_lowers_c_beyond_random"] = C["stubborn25"]["change_c"][0] < min(x["change_c"][0] for x in rs)
    h1["pred_stubborn_pair_order_within_3pts"] = abs(C["stubborn25"]["change_pair_order"][0]) < 0.03
    h1["pred_judge_lowers_pair_order_beyond_random"] = C["judge25"]["change_pair_order"][0] < min(x["change_pair_order"][0] for x in rj)
    h1["pred_judge_c_change_smaller_than_stubborn"] = abs(C["judge25"]["change_c"][0]) < abs(C["stubborn25"]["change_c"][0])
out["I2"] = h1

# ---- H2 ----
def auth_effect(cond, ix=None):
    Ma, _ = margins(cond, "ai"); Mu, _ = margins(cond, "author")
    if ix is not None:
        Ma, Mu = Ma[ix], Mu[ix]
    return stats(Mu)[1] - stats(Ma)[1], stats(Mu)[0] - stats(Ma)[0]

Ma0, _ = margins("I1|baseline", "ai"); n2 = len(Ma0); idx2 = rng.integers(0, n2, (B, n2))
e0, ed0 = auth_effect("I1|baseline"); be0, _ = auth_effect("I1|baseline", idx2)
h2 = {"baseline_authority_effect_c": [float(e0), *ci(be0)], "baseline_authority_effect_d": float(ed0),
      "gate_passed": bool(np.percentile(be0, 97.5) < 0 and e0 <= -0.10), "conditions": {}}
for cond in [k for k in r["conditions"] if k.startswith("I1|") and k != "I1|baseline"]:
    e, ed = auth_effect(cond); be, _ = auth_effect(cond, idx2)
    # the effect is negative, so be - be0 > 0 means it shrank towards zero (in c units)
    h2["conditions"][cond.split("|",1)[1]] = {"authority_effect_c": float(e), "authority_effect_d": float(ed),
                                  "share_of_effect_removed": float((e0 - e) / e0) if e0 else None,
                                  "reduction_ci": ci(be - be0)}
rr = [h2["conditions"][k]["share_of_effect_removed"] for k in h2["conditions"] if k.startswith("rand25_")]
rr100 = [h2["conditions"][k]["share_of_effect_removed"] for k in h2["conditions"] if k.startswith("rand100_")]
if "authority25" in h2["conditions"]:
    a25 = h2["conditions"]["authority25"]
    h2["pred_authority25_removes_half"] = a25["share_of_effect_removed"] >= 0.5
    h2["pred_reduction_ci_excludes_zero"] = a25["reduction_ci"][0] > 0
    h2["pred_beyond_all_random"] = bool(rr) and a25["share_of_effect_removed"] > max(rr)
    h2["random25_share_removed_range"] = [min(rr), max(rr)] if rr else None
    h2["random100_share_removed_range"] = [min(rr100), max(rr100)] if rr100 else None
out["I1"] = h2
(ROOT / "results" / "i.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({"I1": h2, "I2": h1}, indent=1, default=float)[:7000])
