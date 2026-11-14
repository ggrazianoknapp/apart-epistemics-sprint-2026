"""Pre-registration K analysis: writes results/k.json (or j_analysis-style test output when given a folder).

K1  per label and removed group: change in S (threshold part) and J (belief part) against the unedited model, and the
    change for each label minus the change for the author label, with 95% paired bootstrap intervals over questions.
K2  summed activation of the attr25 neurons on each token group, per label (unedited model).
K3  no-pushback accuracy with attr25 removed.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "neuro"
r = json.loads((RUN / ("k_test.json" if len(sys.argv) > 1 else "k_results.json")).read_text())
LABS = ["none", "ai", "human", "professor", "author", "format", "author_nokey"]
B = 10000
rng = np.random.default_rng(20261116)
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def SJ(name):
    R = r["conditions"][name]["rows"]
    q = sorted({x["qid"] for x in R}); qi = {k: i for i, k in enumerate(q)}
    M = np.full((len(LABS), len(q), 2), np.nan)
    for x in R:
        M[LABS.index(x["label"]), qi[x["qid"]], 0 if x["cond"] == "user_wrong" else 1] = x["margin"]
    return q, M.mean(-1), (M[..., 0] - M[..., 1]) / 2, M


q0, S0, J0, M0 = SJ("baseline"); n = len(q0); ix = rng.integers(0, n, (B, n))
out = {"n_questions": n, "baseline": {lab: {"S": float(S0[i].mean()), "J": float(J0[i].mean())} for i, lab in enumerate(LABS)},
       "K1": {}}
au = LABS.index("author")
for name in [k for k in r["conditions"] if k != "baseline"]:
    q1, S1, J1, M1 = SJ(name); assert q1 == q0
    dS, dJ = S1 - S0, J1 - J0
    cell = {}
    for i, lab in enumerate(LABS):
        rel = dS[i] - dS[au]
        cell[lab] = {"dS": [float(dS[i].mean()), *ci(dS[i][ix].mean(1))], "dJ": [float(dJ[i].mean()), *ci(dJ[i][ix].mean(1))],
                     "dS_minus_author": [float(rel.mean()), *ci(rel[ix].mean(1))],
                     "dWrongArm": float((M1[i, :, 0] - M0[i, :, 0]).mean()), "dRightArm": float((M1[i, :, 1] - M0[i, :, 1]).mean())}
    out["K1"][name] = cell
others = [l for l in LABS if l != "author"]
for name in ["attr25", "verified100"]:
    if name in out["K1"]:
        out[f"pred_K1_{name}"] = all(out["K1"][name][l]["dS_minus_author"][2] < 0 for l in others)
out["pred_K1_random_within_0.10"] = all(abs(out["K1"][g][l]["dS"][0]) <= 0.10 for g in out["K1"] if g.startswith("rand") for l in LABS)

# K2: summed attr25 activation by token group and label
groups = ["before", "label", "after", "last"]
A25 = {f"{L}_{E}_{N}" for L, E, N in json.loads((RUN / ("j3_results.json")).read_text())["groups"]["attr25"]} \
    if (RUN / "j3_results.json").exists() else None
k2 = {}
for lab in LABS:
    rows = [x for x in r.get("activations", []) if x["label"] == lab]
    k2[lab] = {}
    for g in groups:
        vals = [sum(v[g] for k, v in x["acts"].items() if (A25 is None or k in A25) and v[g] is not None) for x in rows
                if any(v[g] is not None for v in x["acts"].values())]
        k2[lab][g] = float(np.mean(vals)) if vals else None
out["K2_attr25_summed_activation"] = k2
if "damage" in r:
    out["K3_damage"] = r["damage"]; out["pred_K3_within_2pts"] = r["damage"]["baseline"] - r["damage"]["attr25"] <= 0.02
(RUN / "k_analysis_test.json" if len(sys.argv) > 1 else ROOT / "results" / "k.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k.startswith("pred")}, indent=1))
for name, cell in out["K1"].items():
    print(name, {l: [round(x, 3) for x in cell[l]["dS"]] for l in LABS})
print("K2", {l: {g: (round(v, 3) if v is not None else None) for g, v in d.items()} for l, d in k2.items()})
