"""Pre-registration J analysis: writes results/j.json.

J1  per label: S (threshold part) and J (belief part) of the hold margins, binary rates, c and d', margin AUROC; changes
    against the AI label with paired bootstrap intervals over questions; the census of labelled-neuron poles.
J2  share of the author-minus-AI margin difference removed by copying the AI run into the author run, per layer.
J3  attribution gate; share of the author-minus-AI difference removed by each neuron group; damage.
J4  the authority direction: reliability, projections of held-out labels, one-direction copy against random ones.
"""
import json, sys
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "neuro"   # a folder with test files for smoke checks
zv = np.vectorize(NormalDist().inv_cdf)
B = 10000
rng = np.random.default_rng(20261115)
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
out = {}


def boot_idx(n):
    return rng.integers(0, n, (B, n))


# ---------------- J1 ----------------
r = json.loads((RUN / "j_results.json").read_text())
J1 = [x for x in r["J1"] if "margin" in x]
LABS = ["none", "ai", "human", "professor", "author", "format", "author_nokey"]
qids = sorted({x["qid"] for x in J1}); qi = {q: i for i, q in enumerate(qids)}
known = np.zeros(len(qids), bool)
M = np.full((len(LABS), len(qids), 2), np.nan)          # label x question x arm (0 wrong, 1 right): hold margin
OT = np.zeros_like(M)
for x in J1:
    a = 0 if x["cond"] == "user_wrong" else 1
    M[LABS.index(x["label"]), qi[x["qid"]], a] = x["margin"]; OT[LABS.index(x["label"]), qi[x["qid"]], a] = x["top_other"]
    known[qi[x["qid"]]] = x["known"]
S = M.mean(-1); Jb = (M[..., 0] - M[..., 1]) / 2


def rates(m):  # m: [..., n, 2] -> gave in, accepted, d', c
    n = m.shape[-2]; g = (m[..., 0] < 0).mean(-1); a = (m[..., 1] < 0).mean(-1)
    H = (a * n + 0.5) / (n + 1); F = (g * n + 0.5) / (n + 1)
    return g, a, zv(H) - zv(F), -(zv(H) + zv(F)) / 2


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = allv.argsort().argsort() + 1.0
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


iall = boot_idx(len(qids)); kq = np.where(known)[0]; ikn = boot_idx(len(kq))
j1 = {"n_questions": len(qids), "n_known": int(known.sum()), "labels": {}}
for li, lab in enumerate(LABS):
    g, a, d, c = rates(M[li]); gk, ak, dk, ck = rates(M[li][kq])
    j1["labels"][lab] = {"S_all": float(S[li].mean()), "S_known": float(S[li][kq].mean()), "J_known": float(Jb[li][kq].mean()),
                         "J_all": float(Jb[li].mean()), "gave_in_all": float(g), "accepted_all": float(a), "d_all": float(d),
                         "c_all": float(c), "gave_in_known": float(gk), "accepted_known": float(ak), "d_known": float(dk),
                         "c_known": float(ck), "auroc_known": auroc(M[li][kq, 0], M[li][kq, 1]),
                         "off_target_share": float(OT[li].mean())}
ai = LABS.index("ai")
for li, lab in enumerate(LABS):
    if lab == "ai":
        continue
    dS = S[li] - S[ai]; dJ = Jb[li] - Jb[ai]
    j1["labels"][lab]["vs_ai"] = {
        "dS_all": [float(dS.mean()), *ci(dS[iall].mean(1))],
        "dS_known": [float(dS[kq].mean()), *ci(dS[kq][ikn].mean(1))],
        "dJ_known": [float(dJ[kq].mean()), *ci(dJ[kq][ikn].mean(1))],
        "dJ_all": [float(dJ.mean()), *ci(dJ[iall].mean(1))],
        "dc_all": [float(rates(M[li])[3] - rates(M[ai])[3]), *ci(rates(M[li][iall])[3] - rates(M[ai][iall])[3])],
        "dd_all": [float(rates(M[li])[2] - rates(M[ai])[2]), *ci(rates(M[li][iall])[2] - rates(M[ai][iall])[2])]}
    # margin AUROC change on known questions
    bs = [auroc(M[li][kq[ix], 0], M[li][kq[ix], 1]) - auroc(M[ai][kq[ix], 0], M[ai][kq[ix], 1]) for ix in ikn[:2000]]
    j1["labels"][lab]["vs_ai"]["dAUROC_known"] = [j1["labels"][lab]["auroc_known"] - j1["labels"]["ai"]["auroc_known"], *ci(bs)]
L = j1["labels"]
j1["pred_J1a_ladder_order"] = L["ai"]["S_all"] > L["human"]["S_all"] > L["professor"]["S_all"] > L["author"]["S_all"]
dfa = S[LABS.index("author")] - S[LABS.index("format")]
j1["author_minus_format_S_all"] = [float(dfa.mean()), *ci(dfa[iall].mean(1))]
j1["pred_J1b_format_less_than_half"] = (L["format"]["vs_ai"]["dS_all"][0] > 0.5 * L["author"]["vs_ai"]["dS_all"][0]
                                        and j1["author_minus_format_S_all"][2] < 0)
j1["pred_J1c_professor_lowers_J"] = L["professor"]["vs_ai"]["dJ_known"][2] < 0
j1["pred_J1c_format_J_less_than_half"] = L["format"]["vs_ai"]["dJ_known"][0] > 0.5 * L["author"]["vs_ai"]["dJ_known"][0]

# census
cz = np.load(RUN / "j_results_census.npz")
acts = cz["acts"].astype(np.float32); cmeta = cz["meta"]; neur = cz["neurons"]
key = {(q, c, l): i for i, (q, c, l) in enumerate(cmeta.tolist())}
pairs = [(key[(q, c, "author")], key[(q, c, "ai")]) for (q, c, l) in cmeta.tolist() if l == "author" and (q, c, "ai") in key]
Au = acts[[p[0] for p in pairs]]; Ai = acts[[p[1] for p in pairs]]          # [n, 2 poles, 3244]
sd = np.sqrt((Au.var(0) + Ai.var(0)) / 2)
z = np.where(sd > 0, (Au - Ai).mean(0) / np.where(sd > 0, sd, 1), 0.0)      # [2, 3244]
flat = z.flatten(); rank = (-flat).argsort().argsort() + 1                    # 1 = largest rise
nidx = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(neur.tolist())}
NAMED = {"L30_E25_N342-": 0.724, "L19_E7_N137-": 0.752, "L24_E18_N122-": 0.924, "L36_E167_N252-": 0.969,
         "L36_E167_N252+": 0.969, "L19_E16_N33+": 0.543}
atlas = json.loads((ROOT / "research/deliberation2/scratch_02/atlas/index.json").read_text())
DESC = {(n["L"], n["E"], n["N"]): {p: v.get("desc", "") for p, v in n["poles"].items()} for n in atlas["neurons"]} if "neurons" in atlas else {}
cen = {"n_poles": int(flat.size), "n_pairs": len(pairs), "top5pct_rank": int(np.ceil(0.05 * flat.size)), "named": {}}
for nm, theta in NAMED.items():
    Lx, Ex, Nx = [int(t[1:]) for t in nm[:-1].split("_")]; pole = 0 if nm[-1] == "+" else 1
    i = nidx[(Lx, Ex, Nx)]
    per = {}
    for lab in ["ai", "human", "professor", "author", "format", "author_nokey"]:
        rows = [k for (q, c, l), k in key.items() if l == lab]
        v = acts[rows, pole, i]
        per[lab] = {"mean": float(v.mean()), "share_half_theta": float((v >= 0.5 * theta).mean())}
    cen["named"][nm] = {"z": float(z[pole, i]), "rank": int(rank[pole * len(neur) + i]), "per_label": per}
cen["pred_J1d_named_in_top5pct"] = any(v["rank"] <= cen["top5pct_rank"] for v in cen["named"].values())
top = np.argsort(-flat)[:15]
cen["top15"] = [{"neuron": neur[t % len(neur)].tolist(), "pole": "+" if t < len(neur) else "-", "z": float(flat[t]),
                 "desc": DESC.get(tuple(neur[t % len(neur)].tolist()), {}).get("+" if t < len(neur) else "-", "")} for t in top]
j1["census"] = cen
out["J1"] = j1

# ---------------- J2 and J4b ----------------
rows = r["J2"]
cols = [k for k in rows[0] if k not in ("qid", "cond", "ai", "author")]
j2 = {}
for arm in ["user_wrong", "user_right"]:
    R = [x for x in rows if x["cond"] == arm]
    A = np.array([x["ai"] for x in R]); U = np.array([x["author"] for x in R]); ix = boot_idx(len(R))
    den = A - U
    cell = {"n": len(R), "author_minus_ai": [float((U - A).mean()), *ci((U - A)[ix].mean(1))], "share_removed": {}}
    for k in cols:
        V = np.array([x[k] for x in R]); num = V - U
        cell["share_removed"][k] = [float(num.mean() / den.mean()), *ci(num[ix].mean(1) / den[ix].mean(1))]
    j2[arm] = cell
P = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]
sw = j2["user_wrong"]["share_removed"]
j2["pred_J2a_layer39_full"] = all(abs(sw[f"{m}39"][0] - 1) <= 0.01 for m in ["post", "last"])
j2["pred_J2b_post_80_by_28"] = any(sw[f"post{l}"][0] >= 0.80 for l in P if l <= 28)
j2["pred_J2c_last_below_25_to_24"] = all(sw[f"last{l}"][0] < 0.25 for l in P if l <= 24)
out["J2"] = j2

# ---------------- J4 ----------------
d = r["direction"]
pr = [x for x in r["J1"] if "proj_vs_ai" in x]
j4 = {"split_half_cosine": d["split_half_cosine"], "cosine_with_c2_judgement": d.get("cosine_with_c2_judgement"), "projection": {}}
for Lx in ["16", "24", "32"]:
    tab = {}
    for lab in ["none", "human", "professor", "author", "format", "author_nokey"]:
        v = {(x["qid"], x["cond"]): x["proj_vs_ai"][Lx] for x in pr if x["label"] == lab}
        tab[lab] = v
    ks = sorted(tab["author"]); auth = np.array([tab["author"][k] for k in ks]); ix = boot_idx(len(ks))
    j4["projection"][Lx] = {lab: [float(np.mean([tab[lab][k] for k in ks]) / auth.mean()),
                                  *ci(np.array([tab[lab][k] for k in ks])[ix].mean(1) / auth[ix].mean(1))]
                            for lab in tab}
p24 = j4["projection"]["24"]
j4["pred_J4a_order"] = 0 < p24["human"][0] < p24["professor"][0] < 1 and p24["format"][0] < 0.5
j4["one_direction_share"] = {Lx: {"direction": sw[f"dir{Lx}"], "random": [sw[f"rand{Lx}_{j}"][0] for j in range(5)],
                                  "full_post_copy": sw[f"post{Lx}"]} for Lx in ["16", "24", "32"]}
j4["pred_J4b_one_direction_half"] = any(v["direction"][0] >= 0.5 and v["direction"][0] > max(v["random"])
                                        for v in j4["one_direction_share"].values())
out["J4"] = j4

# ---------------- J3 ----------------
p3 = RUN / "j3_results.json"
if p3.exists():
    r3 = json.loads(p3.read_text())
    j3 = {"gate": {k: r3["gate"].get(k) for k in ["pearson", "spearman", "passed", "n_neurons", "pearson_layers_30_39", "n_layers_30_39"]},
          "conditions": {}}
    def eff(name, arm):
        R = r3["conditions"][name]["rows"]
        d_ = {(x["qid"], x["label"]): x["margin"] for x in R if x["cond"] == arm}
        qs_ = sorted({q for q, _ in d_})
        return qs_, np.array([d_[(q, "author")] - d_[(q, "ai")] for q in qs_]), np.mean([x["top_other"] for x in R])
    for arm in ["user_wrong", "user_right"]:
        qs0, e0, _ = eff("baseline", arm); ix = boot_idx(len(qs0)); cell = {"baseline_effect": [float(e0.mean()), *ci(e0[ix].mean(1))]}
        for name in r3["conditions"]:
            if name == "baseline":
                continue
            qs1, e1, ot = eff(name, arm); assert qs1 == qs0
            red = e1 - e0
            cell[name] = {"effect": float(e1.mean()), "share_removed": float(1 - e1.mean() / e0.mean()),
                          "share_ci": ci(1 - e1[ix].mean(1) / e0[ix].mean(1)), "reduction": [float(red.mean()), *ci(red[ix].mean(1))],
                          "off_target_share": float(ot)}
        j3["conditions"][arm] = cell
    c = j3["conditions"]["user_wrong"]
    rr = [c[k]["share_removed"] for k in c if k.startswith("rand100_")]
    if "attr100" in c:
        sign = np.sign(c["baseline_effect"][0])
        j3["pred_attr100_removes_half"] = c["attr100"]["share_removed"] >= 0.5
        j3["pred_reduction_ci_excludes_zero"] = (c["attr100"]["reduction"][1] > 0) if sign < 0 else (c["attr100"]["reduction"][2] < 0)
        j3["pred_beyond_all_random100"] = bool(rr) and c["attr100"]["share_removed"] > max(rr)
        j3["random100_share_range"] = [min(rr), max(rr)] if rr else None
    rv = [c[k]["share_removed"] for k in c if k.startswith("rand_v100_")]
    if "verified100" in c:
        sign = np.sign(c["baseline_effect"][0])
        j3["pred_verified100_removes_half"] = c["verified100"]["share_removed"] >= 0.5
        j3["pred_verified100_ci_excludes_zero"] = (c["verified100"]["reduction"][1] > 0) if sign < 0 else (c["verified100"]["reduction"][2] < 0)
        j3["pred_verified100_beyond_random"] = bool(rv) and c["verified100"]["share_removed"] > max(rv)
        j3["random_v100_share_range"] = [min(rv), max(rv)] if rv else None
    if "damage" in r3:
        j3["damage"] = r3["damage"]
        j3["damage_within_2pts"] = {k: r3["damage"]["baseline"] - r3["damage"][k] <= 0.02 for k in r3["damage"] if k != "baseline"}
    out["J3"] = j3

(RUN / "j_analysis_test.json" if len(sys.argv) > 1 else ROOT / "results" / "j.json").write_text(json.dumps(out, indent=1, default=float))
summary = {"J1": {k: v for k, v in j1.items() if k.startswith("pred")}, "census_pred": cen["pred_J1d_named_in_top5pct"],
           "J2": {k: v for k, v in j2.items() if k.startswith("pred")}, "J4": {k: v for k, v in j4.items() if k.startswith("pred")},
           "J3": {k: v for k, v in out.get("J3", {}).items() if k.startswith("pred") or k == "gate"}}
print(json.dumps(summary, indent=1, default=float))
