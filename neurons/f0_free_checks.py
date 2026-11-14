"""F0 free checks (no forward passes) on the E4 capture, before pre-registration F.

1. Headroom: for every layer and token group (planted answer, claim, after the claim, last token), a difference-of-means
   reading (user wrong minus user right) fitted on known selection conversations, scored on known held-out ones.
   Compared with how well Qwen's own hold margin separates the arms (AUROC 0.975).
2. GG's split at every layer, last token: d_wrong = held minus gave in (user-wrong conversations), d_right = refused
   minus accepted (user-right conversations), both on the selection split. Shared = (d_wrong + d_right) / 2,
   difference = (d_wrong - d_right) / 2. Reports split-half stability (cosine between directions built on two random
   halves of the selection questions) and how well each part predicts a correct final decision on held-out
   conversations.
Writes /workspace/runs/f0_free_checks.json.
"""
import json
import numpy as np
import torch

CAP = torch.load("/workspace/runs/e4_capture.pt").float()  # [1200, layers, 4 groups, hidden]
meta = [json.loads(l) for l in open("/workspace/runs/e1_meta.jsonl")]
NL = CAP.shape[1]
uw = torch.tensor([m["cond"] == "user_wrong" for m in meta]); known = torch.tensor([m["known"] for m in meta])
sel = torch.tensor([m["split"] == "select" for m in meta]); held = ~sel
gave = torch.tensor([m["gave_in"] for m in meta])  # margin < 0: switched to the user's letter
correct = torch.where(uw, ~gave, gave)  # held against a wrong user, or accepted a right one
rng = np.random.default_rng(0)


def auroc(pos, neg):
    allv = torch.cat([pos, neg]); r = allv.argsort().argsort().double() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
out = {"reading_auroc": {}, "split": {}}
GROUPS = ["planted", "claim", "suffix", "last"]
for L in range(NL):
    for gi, g in enumerate(GROUPS):
        X = CAP[:, L, gi]
        d = X[sel & known & uw].mean(0) - X[sel & known & ~uw].mean(0)
        s = X @ d
        out["reading_auroc"][f"{L}|{g}"] = auroc(s[held & known & uw], s[held & known & ~uw])
    # GG's split at the last token
    X = CAP[:, L, 3]
    def parts(mask):
        dw = X[mask & uw & ~gave].mean(0) - X[mask & uw & gave].mean(0)
        dr = X[mask & ~uw & ~gave].mean(0) - X[mask & ~uw & gave].mean(0)
        return dw, dr, (dw + dr) / 2, (dw - dr) / 2
    dw, dr, sh, df = parts(sel)
    qs = sorted({m["qid"] for m, s_ in zip(meta, sel) if s_})
    halves = []
    for _ in range(20):
        pick = set(rng.choice(qs, len(qs) // 2, replace=False))
        hA = torch.tensor([m["qid"] in pick for m in meta]) & sel; hB = sel & ~hA
        a, b = parts(hA), parts(hB)
        halves.append([cos(a[k], b[k]) for k in range(4)])
    halves = np.array(halves).mean(0)
    # how well each part predicts a correct decision on held-out conversations (sign: projection toward "correct")
    res = {"cos_dwrong_dright": cos(dw, dr), "cos_shared_reading": cos(sh, X[sel & known & uw].mean(0) - X[sel & known & ~uw].mean(0)),
           "split_half_cos": dict(zip(["d_wrong", "d_right", "shared", "difference"], [float(h) for h in halves]))}
    for name, v in [("shared", sh), ("difference", df)]:
        p = X @ v
        res[f"{name}_predicts_correct_heldout"] = auroc(p[held & correct], p[held & ~correct])
        res[f"{name}_predicts_correct_heldout_known"] = auroc(p[held & known & correct], p[held & known & ~correct])
    out["split"][L] = res
best = max(out["reading_auroc"].items(), key=lambda kv: kv[1])
out["best_reading"] = {"cell": best[0], "auroc": best[1], "margin_auroc_bar": 0.975}
json.dump(out, open("/workspace/runs/f0_free_checks.json", "w"), indent=1)
print("best reading", best)
for L in range(16, NL, 2):
    r = out["split"][L]
    print(L, "split-half cos", {k: round(v, 2) for k, v in r["split_half_cos"].items()},
          "| diff->correct (known)", round(r["difference_predicts_correct_heldout_known"], 3),
          "| shared->correct (known)", round(r["shared_predicts_correct_heldout_known"], 3))
