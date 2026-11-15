"""Pre-registration O analysis: writes results/o.json.

Further tests of the authority direction of Study N on its held-out conversations. Shares as in Study N:
adding = (swapped - AI) / (author - AI); removing = (swapped - author) / (AI - author). 95% bootstrap intervals
over conversations (10,000 resamples).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "runs" / "neuro" / "o_results.json").read_text())
B = 10000
rng = np.random.default_rng(20261117)
ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
T, right = r["test"], r["user_right"]
out = {"layer": r["layer"], "positions": r["positions"], "reminder_positions": r["reminder_positions"]}


def share(sw, tg, ot, ix):
    num, den = sw - tg, ot - tg
    return {"share": float(num.mean() / den.mean()), "share_ci": ci(num[ix].mean(1) / den[ix].mean(1))}


for arm, flag in [("user_wrong", False), ("user_right", True)]:
    qq = sorted(q for q in T if right[q] == flag)
    ix = rng.integers(0, len(qq), (B, len(qq)))
    A = lambda f: np.array([f(T[q]) for q in qq])
    cell = {"n": len(qq)}
    # O1
    labs = list(T[qq[0]]["o1"])
    M = np.array([[T[q]["o1"][l]["margin"] for l in labs] for q in qq]); C = np.array([[T[q]["o1"][l]["component"] for l in labs] for q in qq])
    corr = lambda m, c: float(np.corrcoef(m.mean(0), c.mean(0))[0, 1])
    boot = [corr(M[i], C[i]) for i in ix[:2000]]
    rest = [j for j, l in enumerate(labs) if l not in ("ai", "author")]  # leaving out the two labels the direction was built from
    cell["o1"] = {"labels": {l: {"margin": float(M[:, j].mean()), "component": float(C[:, j].mean()),
                                 "gave_in_readout": float((M[:, j] < 0).mean())} for j, l in enumerate(labs)},
                  "correlation": [corr(M, C), *ci(boot)],
                  "correlation_without_ai_and_author": [corr(M[:, rest], C[:, rest]), *ci([corr(M[i][:, rest], C[i][:, rest]) for i in ix[:2000]])]}
    # O2
    a2, u2 = A(lambda t: t["o2"]["ai"]), A(lambda t: t["o2"]["author"])
    d = u2 - a2
    nr = len(T[qq[0]]["o2"]["random"])
    cell["o2"] = {"margin_ai": float(a2.mean()), "margin_author": float(u2.mean()), "author_minus_ai": [float(d.mean()), *ci(d[ix].mean(1))],
                  "component_ai": float(A(lambda t: t["o2"]["component_ai"]).mean()), "component_author": float(A(lambda t: t["o2"]["component_author"]).mean()),
                  "add": share(A(lambda t: t["o2"]["add"]), a2, u2, ix), "remove": share(A(lambda t: t["o2"]["remove"]), u2, a2, ix),
                  "random_add": [share(A(lambda t, j=j: t["o2"]["random"][j]["add"]), a2, u2, ix)["share"] for j in range(nr)],
                  "random_remove": [share(A(lambda t, j=j: t["o2"]["random"][j]["remove"]), u2, a2, ix)["share"] for j in range(nr)]}
    # O3
    ai, au = A(lambda t: t["o1"]["ai"]["margin"]), A(lambda t: t["o1"]["author"]["margin"])
    cell["o3"] = {"0": [float(ai.mean()), *ci(ai[ix].mean(1))]}
    for k in ["0.5", "1", "2", "3"]:
        x = A(lambda t, k=k: t["o3"][k]); cell["o3"][k] = [float(x.mean()), *ci(x[ix].mean(1))]
    cell["o3"]["author_label_unedited"] = float(au.mean())
    cell["o3"]["gave_in_readout"] = {k: float((A(lambda t, k=k: t["o3"][k]) < 0).mean()) for k in ["0.5", "1", "2", "3"]}
    # O4, O5
    for key in ["o4", "o5"]:
        cell[key] = {name: {"add": share(A(lambda t, n=name: t[key][n]["add"]), ai, au, ix),
                            "remove": share(A(lambda t, n=name: t[key][n]["remove"]), au, ai, ix)} for name in T[qq[0]][key]}
    # O6
    if "o6" in r:
        o6 = r["o6"]; a6, u6 = np.array([o6[q]["ai"] for q in qq]), np.array([o6[q]["author"] for q in qq])
        cell["o6"] = {"margin_ai": [float(ai.mean()), float(a6.mean())], "margin_author": [float(au.mean()), float(u6.mean())],
                      "component_ai": [float(A(lambda t: t["o1"]["ai"]["component"]).mean()), float(np.mean([o6[q]["component_ai"] for q in qq]))],
                      "component_author": [float(A(lambda t: t["o1"]["author"]["component"]).mean()), float(np.mean([o6[q]["component_author"] for q in qq]))],
                      "add": share(np.array([o6[q]["add"] for q in qq]), a6, u6, ix), "remove": share(np.array([o6[q]["remove"] for q in qq]), u6, a6, ix)}
    out[arm] = cell
w = out["user_wrong"]
out["pred_O1_correlation"] = w["o1"]["correlation"][0] <= -0.7 and w["o1"]["correlation"][2] < 0
out["pred_O2a_sentence_author_lowers_margin"] = w["o2"]["author_minus_ai"][2] < 0
out["pred_O2b_swap_third_beyond_random"] = bool(out["pred_O2a_sentence_author_lowers_margin"]) and \
    w["o2"]["add"]["share"] >= 1 / 3 and w["o2"]["remove"]["share"] >= 1 / 3 and \
    w["o2"]["add"]["share"] > max(w["o2"]["random_add"]) and w["o2"]["remove"]["share"] > max(w["o2"]["random_remove"])
m3 = [w["o3"][k][0] for k in ["0", "0.5", "1", "2", "3"]]
out["pred_O3_falls_at_every_step"] = all(b < a for a, b in zip(m3, m3[1:]))
(ROOT / "results" / "o.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k.startswith("pred")}, indent=1))
pc = lambda v: "%.0f%% [%.0f, %.0f]" % (100 * v["share"], *[100 * x for x in v["share_ci"]])
for arm in ["user_wrong", "user_right"]:
    c = out[arm]; print(arm, "n =", c["n"])
    print("  O1 correlation %.2f [%.2f, %.2f] | without AI and author %.2f [%.2f, %.2f]" % (*c["o1"]["correlation"], *c["o1"]["correlation_without_ai_and_author"]))
    for l, v in sorted(c["o1"]["labels"].items(), key=lambda kv: kv[1]["component"]):
        print("     %-13s component %+.3f | margin %.2f | gives in (read-out) %.0f%%" % (l, v["component"], v["margin"], 100 * v["gave_in_readout"]))
    o2 = c["o2"]
    print("  O2 sentence: AI %.2f, author %.2f, difference %.2f [%.2f, %.2f] | add %s | remove %s | random add max %.0f%%, remove max %.0f%%" % (
        o2["margin_ai"], o2["margin_author"], *o2["author_minus_ai"], pc(o2["add"]), pc(o2["remove"]), 100 * max(o2["random_add"]), 100 * max(o2["random_remove"])))
    print("  O3 dose:", {k: round(c["o3"][k][0], 2) for k in ["0", "0.5", "1", "2", "3"]}, "| author label", round(c["o3"]["author_label_unedited"], 2),
          "| gives in (read-out)", {k: round(v, 2) for k, v in c["o3"]["gave_in_readout"].items()})
    for key in ["o4", "o5"]:
        for name, v in c[key].items():
            print("  %s %-24s add %s | remove %s" % (key.upper(), name, pc(v["add"]), pc(v["remove"])))
    if "o6" in c:
        o6 = c["o6"]
        print("  O6 with the 25 removed: margin AI %.2f -> %.2f, author %.2f -> %.2f | component AI %+.3f -> %+.3f, author %+.3f -> %+.3f | add %s | remove %s" % (
            *o6["margin_ai"], *o6["margin_author"], *o6["component_ai"], *o6["component_author"], pc(o6["add"]), pc(o6["remove"])))
