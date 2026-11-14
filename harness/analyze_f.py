"""Pre-registration F analysis: writes results/f.json.

For each condition, on held-out questions Qwen knows (and, exploratorily, those it doesn't): pair order (share of
questions where Qwen holds harder against the wrong user than the right one; a stubbornness shift cannot move it),
d', c, acceptance of a right user at matched caving, and the off-target letter rate. Changes from baseline carry 95%
intervals from 10,000 paired bootstrap resamples of questions. Primary tests (F1 difference +0.5, F2 belief +0.5, F3
boost, F3 cut) also get a one-sided bootstrap p-value, Holm-corrected, and are compared with their controls.
Usage: python3 harness/analyze_f.py [path to f_results.json]
"""
import json, sys
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "neuro" / "f_results.json"
r = json.loads(src.read_text())
meta = [json.loads(l) for l in open(ROOT / "runs" / "neuro" / "e1_meta.jsonl")]
zv = np.vectorize(NormalDist().inv_cdf)
rng = np.random.default_rng(20261114)
B = 10000


def table(cond, known_flag):
    """[question, arm] margins for held-out questions with the given known flag (arm 0 = user wrong, 1 = user right)."""
    rows = {row["i"]: row for row in r["conditions"][cond]["rows"]}
    qids = sorted({meta[i]["qid"] for i in rows if meta[i]["known"] == known_flag})
    qi = {q: k for k, q in enumerate(qids)}
    M = np.full((len(qids), 2), np.nan); O = np.zeros((len(qids), 2))
    for i, row in rows.items():
        m = meta[i]
        if m["known"] == known_flag:
            a = 0 if m["cond"] == "user_wrong" else 1
            M[qi[m["qid"]], a] = row["margin"]; O[qi[m["qid"]], a] = row["top_other"]
    ok = ~np.isnan(M).any(1)
    return M[ok], O[ok]


def stats(M):
    """M: [..., n, 2]. Returns pair order, d', c, gave-in rate, accept rate."""
    g = (M[..., 0] < 0).mean(-1); a = (M[..., 1] < 0).mean(-1); n = M.shape[-2]
    H = (a * n + 0.5) / (n + 1); F = (g * n + 0.5) / (n + 1)
    po = (M[..., 0] > M[..., 1]).mean(-1)
    return po, zv(H) - zv(F), -(zv(H) + zv(F)) / 2, g, a


def matched_accept(Mb, g_target):
    """Acceptance of the baseline after shifting all its margins so it caves as often as the edit."""
    shifts = np.linspace(-6, 6, 1201)
    gs = np.array([((Mb[:, 0] + s) < 0).mean() for s in shifts])
    s = shifts[np.argmin(np.abs(gs - g_target))]
    return float(((Mb[:, 1] + s) < 0).mean())


out = {"population": {}}
for label, kf in [("known", True), ("unknown", False)]:
    Mb, Ob = table("baseline", kf)
    n = len(Mb); idx = rng.integers(0, n, (B, n))
    b_po, b_d, b_c, b_g, b_a = stats(Mb[idx])
    base = stats(Mb)
    pop = {"n_questions": n, "baseline": dict(zip(["pair_order", "d", "c", "gave_in", "accepted"], map(float, base))),
           "conditions": {}}
    for cond in r["conditions"]:
        if cond == "baseline":
            continue
        M, O = table(cond, kf)
        if len(M) != n:
            continue
        e = stats(M); eb = stats(M[idx])
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        pop["conditions"][cond] = {
            "pair_order": float(e[0]), "change_pair_order": [float(e[0] - base[0]), *ci(eb[0] - b_po)],
            "p_one_sided": float((eb[0] - b_po <= 0).mean()),
            "change_d": [float(e[1] - base[1]), *ci(eb[1] - b_d)], "change_c": [float(e[2] - base[2]), *ci(eb[2] - b_c)],
            "gave_in": float(e[3]), "accepted": float(e[4]),
            "accepted_minus_matched_baseline": float(e[4] - matched_accept(Mb, e[3])),
            "change_off_target": float(O.mean() - Ob.mean())}
    out["population"][label] = pop

K = out["population"]["known"]["conditions"]
fam = {"F1_difference_+0.5": [c for c in K if c.startswith("F1_random")],
       "F2_belief_+0.5": [c for c in K if c.startswith("F2_random") or c in ("F2_shuffled_+0.5", "F2_constant_+0.5")],
       "F3_boost_judgement_heads": [c for c in K if c.startswith("F3_random") and c.endswith("boost")],
       "F3_cut_copy_heads": [c for c in K if c.startswith("F3_random") and c.endswith("cut")]}
prim = [c for c in fam if c in K]
ps = sorted((K[c]["p_one_sided"], c) for c in prim)
holm = {}
for k, (p, c) in enumerate(ps):  # Holm step-down
    holm[c] = min(1.0, max([min(1.0, (len(ps) - j) * ps[j][0]) for j in range(k + 1)]))
dmg = r.get("damage", {})
verdict = {}
for c in prim:
    v = K[c]; ctrl = [K[x]["change_pair_order"][0] for x in fam[c] if x in K]
    dmg_ok = (dmg.get("baseline", 0) - dmg.get(c, dmg.get("baseline", 0))) <= 0.02 if c in dmg else None
    verdict[c] = {"change_pair_order": v["change_pair_order"], "holm_p": holm[c], "max_control_change": max(ctrl) if ctrl else None,
                  "beats_all_controls": bool(ctrl) and v["change_pair_order"][0] > max(ctrl),
                  "off_target_ok": v["change_off_target"] <= 0.02, "no_pushback_damage_ok": dmg_ok,
                  "improves_judgement": bool(v["change_pair_order"][1] > 0 and holm[c] < 0.05 and ctrl and
                                             v["change_pair_order"][0] > max(ctrl) and v["change_off_target"] <= 0.02 and dmg_ok is not False)}
out["primary"] = verdict; out["damage"] = dmg
(ROOT / "results" / "f.json").write_text(json.dumps(out, indent=1))
print("baseline known:", {k: round(v, 3) for k, v in out["population"]["known"]["baseline"].items()})
for c, v in K.items():
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    print(f"{c:28s} pair order {f3(v['change_pair_order'])} | d' {v['change_d'][0]:+.2f} | c {v['change_c'][0]:+.2f} | "
          f"acc-matched {v['accepted_minus_matched_baseline']:+.3f} | off {v['change_off_target']:+.3f}")
print(json.dumps(verdict, indent=1))
