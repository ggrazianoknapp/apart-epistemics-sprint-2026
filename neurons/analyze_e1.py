"""Analyse the E1 readout (pre-registration A, experiment E1) from runs/neuro/e1_readout.npz and e1_meta.jsonl.

Label: the user's claim is false (user_wrong) = 1, true (user_right) = 0. Feature for a neuron: max activation on the
user's claim minus max activation on the same option text in the option list (positive pole), or the min-difference
for the negative pole. Everything is evaluated on the 400 held-out questions; "known" means Qwen answers the question
correctly without pushback and with thinking closed.
Usage: uv run --with numpy --with scikit-learn python neurons/analyze_e1.py
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

D = Path(__file__).resolve().parent.parent / "runs" / "neuro"
z = np.load(D / "e1_readout.npz")
meta = [json.loads(l) for l in open(D / "e1_meta.jsonl")]
layers = list(z["layers"])
rand = z["rand"]          # [conv, layer, (diff, diff_neg, routed), neuron]
resid = z["resid24"]
y = np.array([m["cond"] == "user_wrong" for m in meta], int)
held = np.array([m["split"] == "heldout" for m in meta])
known = np.array([m["known"] for m in meta])
gave = np.array([m["gave_in"] for m in meta])
rng = np.random.default_rng(1)
out = {}


def auc(mask, x):
    return float(roc_auc_score(y[mask], x[mask])) if len(set(y[mask])) == 2 else None


def boot_auc(mask, x, b=2000):
    idx = np.where(mask)[0]
    qids = np.array([meta[i]["qid"] for i in idx])
    uq = np.unique(qids)
    vals = []
    for _ in range(b):
        pick = rng.choice(uq, len(uq))
        sel = np.concatenate([idx[qids == q] for q in pick])
        if len(set(y[sel])) == 2:
            vals.append(roc_auc_score(y[sel], x[sel]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


lab_names = list(meta[0]["labelled"].keys())
for name in lab_names:
    pos = np.array([m["labelled"][name][0] for m in meta])
    neg = -np.array([m["labelled"][name][1] for m in meta])
    routed = np.array([m["labelled"][name][2] for m in meta])
    out[name] = {}
    for pole, x in [("positive", pos), ("negative", neg)]:
        out[name][pole] = {s: {"auroc": auc(held & k, x), "ci": boot_auc(held & k, x)}
                           for s, k in [("known", known), ("unknown", ~known), ("all", np.ones_like(known))]}
    out[name]["routed_on_claim"] = float(routed[held].mean())

# primary neuron against random neurons in its layer (and against routing-matched random neurons)
prim = "falsehood L24_E65_N421"
li = layers.index(24)
pos_p = np.array([m["labelled"][prim][0] for m in meta])
routed_p = np.array([m["labelled"][prim][2] for m in meta])[held].mean()
r_auc = np.array([auc(held & known, rand[:, li, 0, j]) or 0.5 for j in range(rand.shape[3])])
r_routed = rand[held][:, li, 2, :].mean(0)
matched = np.abs(r_routed - routed_p) <= max(0.5 * routed_p, 0.02)
p_auc = out[prim]["positive"]["known"]["auroc"]
out["primary_vs_random"] = {
    "primary_auroc_known": p_auc, "random_auroc_95th": float(np.percentile(r_auc, 95)),
    "primary_percentile_among_random": float((r_auc < p_auc).mean() * 100),
    "n_random": int(len(r_auc)), "n_routing_matched": int(matched.sum()),
    "primary_percentile_among_matched": float((r_auc[matched] < p_auc).mean() * 100) if matched.any() else None}

# flagged but gave in: user-wrong conversations where the neuron fired above the 90th percentile of true-claim cases
thr = np.percentile(pos_p[held & (y == 0)], 90)
flag = held & (y == 1) & (pos_p > thr)
out["flagged_but_gave_in"] = {"threshold": float(thr), "n_flagged": int(flag.sum()),
                              "gave_in_share": float(gave[flag].mean()) if flag.any() else None,
                              "gave_in_share_unflagged": float(gave[held & (y == 1) & ~flag].mean())}

# probe ladder: train on the selection split, test on held-out
train = ~held


def probe(X):
    sc = StandardScaler().fit(X[train])
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[train]), y[train])
    s = clf.decision_function(sc.transform(X))
    return {"known": auc(held & known, s), "all": auc(held, s)}


lab_X = np.array([[v for name in lab_names for v in m["labelled"][name][:2]] for m in meta])
out["probe_labelled_set"] = probe(lab_X)
out["probe_layer24_residual"] = probe(resid)
out["probe_layer24_random_neurons"] = probe(rand[:, li, 0, :])
out["gave_in_rates"] = {"user_wrong": float(gave[held & (y == 1)].mean()), "user_right": float(gave[held & (y == 0)].mean()),
                        "user_wrong_known": float(gave[held & (y == 1) & known].mean())}
out["n"] = {"conversations_heldout": int(held.sum()), "known_questions_heldout": int((held & known).sum() // 2)}
(D / "e1_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ["primary_vs_random", "flagged_but_gave_in", "probe_labelled_set",
                                      "probe_layer24_residual", "probe_layer24_random_neurons", "gave_in_rates", "n"]}, indent=1))
print(prim, out[prim])
