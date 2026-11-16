"""Robustness check for the 11-model label study (Studies D, G3, U, V): the same analysis without the questions whose
answer keys two checker models flagged as doubtful or ambiguous (results/keys.json, Study G2). Writes
results/u_keys.json. Added on 16 November after the other analyses, at a reviewer's request: exploratory.
"""
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
zv = np.vectorize(NormalDist().inv_cdf)
rng = np.random.default_rng(7)
B = 10000
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
drop = {q for q, v in json.load(open(ROOT / "results" / "keys.json"))["questions"].items() if v["verdict"] in ("doubtful", "ambiguous")}
LABS = ["internet", "student", "ai", "ai_key", "human", "professor", "grader", "author", "developer", "creator", "maker", "rival", "father", "mother"]


def sdt(g, a, nu, nr):
    H = (a + 0.5) / (nr + 1); F = (g + 0.5) / (nu + 1)
    return zv(H) - zv(F), -(zv(H) + zv(F)) / 2


out = {"n_flagged": len(drop)}
for tag, dropset in [("all_questions", set()), ("without_flagged_keys", drop)]:
    per, boots, share = {}, {}, []
    for f in sorted((ROOT / "runs" / "source3").glob("*.jsonl")):
        if f.name.startswith("calls__") or f.name.startswith("qwen"):
            continue
        d = [{r["qid"]: r for r in map(json.loads, (ROOT / "runs" / s / f.name).open())} for s in ["source", "source2", "source3", "source4"]]
        items = [dict(d[0][q], **{k: d[1][q][k] for k in ["professor", "author"]},
                      **{k: d[2][q][k] for k in ["internet", "student", "grader", "ai_key", "developer", "creator"]},
                      **{k: d[3][q][k] for k in ["maker", "rival", "father", "mother"]}) for q in d[2] if all(q in x for x in d)]
        ur_all = [it for it in items if it["user_right"]]
        share.append(sum(it["qid"] in drop for it in ur_all) / len(ur_all))
        items = [it for it in items if it["qid"] not in dropset]
        uw = [it for it in items if not it["user_right"] and it["t1"] == it["answer"]]; ur = [it for it in items if it["user_right"]]
        nu, nr = len(uw), len(ur); iu = rng.integers(0, nu, (B, nu)); ir = rng.integers(0, nr, (B, nr)); m = items[0]["model"]
        per[m], boots[m] = {"n_user_right": nr}, {}
        for l in LABS:
            G = np.array([it[l]["letter"] == it["pushed"] for it in uw], float); A = np.array([it[l]["letter"] == it["answer"] for it in ur], float)
            OK = np.array([it[l]["letter"] == it["answer"] for it in uw], float)
            dd, cc = sdt(G.sum(), A.sum(), nu, nr); bd, bc = sdt(G[iu].sum(1), A[ir].sum(1), nu, nr)
            per[m][l] = {"accepted": float(A.mean()), "gave_in": float(G.mean()), "d": float(dd), "c": float(cc), "ends_correct": float((OK.mean() + A.mean()) / 2)}
            boots[m][l] = {"d": bd, "c": bc, "accepted": A[ir].mean(1)}
    M = sorted(per)
    pool = lambda l, k: np.mean([boots[m][l][k] for m in M], axis=0); est = lambda l, k: float(np.mean([per[m][l][k] for m in M]))
    diff = lambda k: [est("internet", k) - est("author", k), *ci(pool("internet", k) - pool("author", k))]
    out[tag] = {"accepted_internet": est("internet", "accepted"), "accepted_author": est("author", "accepted"),
                "accepted_internet_minus_author": diff("accepted"), "stubbornness_internet_minus_author": diff("c"),
                "judgement_internet_minus_author": diff("d"),
                "models_accepting_less_under_internet": int(sum(per[m]["internet"]["accepted"] < per[m]["author"]["accepted"] for m in M)),
                "ends_correct_range": [min(est(l, "ends_correct") for l in LABS), max(est(l, "ends_correct") for l in LABS)],
                "c_vs_ai": {l: est(l, "c") - est("ai", "c") for l in LABS if l != "ai"},
                "right_user_cases_per_model": [per[m]["n_user_right"] for m in M]}
    if tag == "all_questions":
        out["share_of_right_user_cases_flagged_per_model"] = [min(share), max(share)]
(ROOT / "results" / "u_keys.json").write_text(json.dumps(out, indent=1))
for tag in ["all_questions", "without_flagged_keys"]:
    o = out[tag]
    print(tag, "| accepted %.1f%% vs %.1f%%, diff %.1f [%.1f, %.1f] | c %.2f [%.2f, %.2f] | d' %.2f [%.2f, %.2f] | %d of 11 | ends correct %.0f%%-%.0f%%" % (
        100 * o["accepted_internet"], 100 * o["accepted_author"], *[100 * x for x in o["accepted_internet_minus_author"]],
        *o["stubbornness_internet_minus_author"], *o["judgement_internet_minus_author"], o["models_accepting_less_under_internet"],
        *[100 * x for x in o["ends_correct_range"]]))
