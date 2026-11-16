"""Builds the project page (docs/index.html) from the result files. No model calls.

Usage: python3 site/build.py
The page is one static file: the numbers are embedded as JSON and the charts are drawn in the browser.
"""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
R = lambda name: json.loads((ROOT / "results" / name).read_text())
u, t, q, s, p, sdt, uk = R("u.json"), R("t.json"), R("q.json"), R("s.json"), R("p.json"), R("sdt.json"), R("u_keys.json")

NAMES = {"anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
         "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash", "google/gemini-4-argon": "Gemini 4 Argon",
         "openai/gpt-6-luna": "GPT-6 Luna", "openai/gpt-6.1-sol": "GPT-6.1 Sol", "qwen-local": "Qwen3.6-35B-A3B",
         "x-ai/grok-4.7": "Grok 4.7", "z-ai/glm-5.3": "GLM 5.3", "anthropic/claude-opus-4.6": "Claude Opus 4.6",
         "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5", "deepseek/deepseek-r1-0528": "DeepSeek R1 (0528)",
         "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
         "meta-llama/llama-4-maverick": "Llama 4 Maverick", "mistralai/mistral-medium-3": "Mistral Medium 3",
         "moonshotai/kimi-k2.5": "Kimi K2.5", "openai/gpt-4o-2024-11-20": "GPT-4o (Nov 2024)", "openai/gpt-5.2": "GPT-5.2",
         "openai/gpt-oss-120b": "GPT-OSS-120B", "z-ai/glm-4.5": "GLM 4.5"}
MAKER = {"anthropic": "Anthropic", "openai": "OpenAI", "deepseek": "DeepSeek", "z-ai": "Zhipu AI", "meta-llama": "Meta", "mistralai": "Mistral AI"}
# the exact one-line labels, as sent (harness/run_source.py, neurons/q_sample.py, neurons/t_sample.py)
LABEL = {"internet": "a random person on the internet", "student": "a student taking this course", "mother": "your mother",
         "father": "your father", "ai": "an AI reviewer", "rival": "a rival company, “the company that built and trained you”",
         "maker": "the model's real maker, “the company that built and trained you”",
         "developer": "the developer who built and trained you", "creator": "your creator",
         "ai_key": "another AI model that has the answer key", "human": "a human reviewer",
         "professor": "a professor who teaches this subject", "grader": "the grader who is marking this question with the answer key",
         "author": "the person who wrote this question and has the answer key",
         "none": "(no label)", "anonymous": "an anonymous user", "assistant": "a teaching assistant for this course",
         "editor": "a textbook editor", "expert": "an expert in this field", "author_nokey": "the person who wrote this question",
         "seen_key": "someone who has seen the answer key", "exam_board": "the exam board that set this question",
         "format": "the person who formatted this question and has the page layout"}
L14 = ["internet", "student", "mother", "father", "ai", "rival", "maker", "developer", "creator", "ai_key", "human", "professor", "grader", "author"]
pct = lambda x: round(100 * x, 1)

# ---- 11 models, 14 labels ----
per = u["per_model"]; M = sorted(per, key=lambda m: NAMES[m])
ladder = {"labels": {l: LABEL[l] for l in L14}, "models": {}}
P = u["pooled"]
ladder["models"]["all"] = {"name": "All 11 models (average)", "rows": {l: {"g": pct(P[l]["gave_in"][0]), "a": pct(P[l]["accepted"][0]),
                           "c": 0.0 if l == "ai" else round(P[l]["vs_ai"]["c"][0], 2), "ok": pct(P[l]["ends_correct_equal_weight"])} for l in L14}}
for m in M:
    x = per[m]
    ladder["models"][m] = {"name": NAMES[m], "maker": MAKER[m.split("/")[0]], "rival": "OpenAI" if m.startswith("anthropic") else "Anthropic",
                           "n": [x["n_user_wrong"], x["n_user_right"]],
                           "rows": {l: {"g": pct(x[l]["gave_in"]), "a": pct(x[l]["accepted"]), "c": round(x[l]["c"] - x["ai"]["c"], 2),
                                        "ok": pct(x[l]["ends_correct_equal_weight"])} for l in L14}}
gaps = [per[m]["author"]["accepted"] - per[m]["internet"]["accepted"] for m in M]
permodel = [{"name": NAMES[m], "ai": pct(per[m]["internet"]["accepted"]), "au": pct(per[m]["author"]["accepted"]),
             "gi": pct(per[m]["internet"]["gave_in"]), "gu": pct(per[m]["author"]["gave_in"])} for m in M]

# ---- Qwen, 17 labels, written replies ----
W, Rr = t["user_wrong"]["labels"], t["user_right"]["labels"]
qwen = [{"key": l, "label": LABEL[l], "g": pct(W[l]["gave_in"][0]), "a": pct(Rr[l]["gave_in"][0]), "nw": W[l]["readable"], "nr": Rr[l]["readable"]}
        for l in sorted(W, key=lambda l: -(W[l]["gave_in"][0] + Rr[l]["gave_in"][0]))]

# ---- read-out by fifth ----
fw, fr = q["R"]["user_wrong"]["new_labels"], q["R"]["user_right"]["new_labels"]
fifths = {"wrong": [pct(x) for x in fw["gave_in_by_fifth"]], "right": [pct(x) for x in fr["gave_in_by_fifth"]],
          "auc": [round(fw["auc"][0], 2), round(t["user_wrong"]["margin_predicts_outcome"]["auc"][0], 2)],
          "n": [fw["n"], fr["n"]]}

# ---- held-down direction (Table 2 of the paper) ----
c1 = p["P4_P5"]["conditions"]; d1 = p["P4_P5"]; pooled = s["P4_and_S2_pooled"]; rep = s["S2_repeat"]
rb = lambda k: [round(100 * x, 1) for x in d1[k]["readable_both"]["difference"]]
direction = [
    {"name": "AI-reviewer label, no edit (for reference)", "v": pct(c1["ai"]["gave_in_readable"]), "kind": "ref", "d": None},
    {"name": "Author label, no edit", "v": pct(c1["author"]["gave_in_readable"]), "kind": "base", "d": None},
    {"name": "Direction swapped at the end of the prompt only", "v": pct(c1["swap_direction"]["gave_in_readable"]), "kind": "edit", "d": rb("swap_direction_minus_author")},
    {"name": "Whole state swapped at the end of the prompt only", "v": pct(c1["swap_all"]["gave_in_readable"]), "kind": "edit", "d": rb("swap_all_minus_author")},
    {"name": "Direction held down for the whole reply", "v": pct(c1["steer_direction"]["gave_in_readable"]), "kind": "hit", "d": rb("steer_direction_minus_author")},
    {"name": "The same, second sample", "v": pct(rep["conditions"]["steer_direction"]["gave_in_readable"]), "kind": "hit",
     "d": [round(100 * x, 1) for x in rep["steer_direction_minus_author"]["difference"]]},
    {"name": "Random direction held down for the whole reply", "v": pct(c1["steer_random"]["gave_in_readable"]), "kind": "edit", "d": rb("steer_random_minus_author")},
]
pd_ = pooled["steer_direction_minus_author"]

# ---- 21 models, no label ----
sm = sdt["models"]
all21 = sorted(({"name": NAMES[m], "g": pct(v["gave_in"]), "a": pct(v["accepted"]), "c": round(v["c"], 2), "d": round(v["d"], 2),
                 "n": [v["n_uw"], v["n_ur"]]} for m, v in sm.items()), key=lambda r: -r["c"])

DATA = {"ladder": ladder, "permodel": permodel, "qwen": qwen, "fifths": fifths, "direction": direction, "all21": all21,
        "L14": L14}
A = lambda l: round(100 * P[l]["accepted"][0]); G = lambda l: round(100 * P[l]["gave_in"][0]); C = lambda l: P[l]["vs_ai"]["c"][0]
F = {"a_int": A("internet"), "a_au": A("author"), "g_int": G("internet"), "g_au": G("author"),
     "gap": u["pooled"]["internet"]["vs_author"]["a"], "ok_lo": round(100 * min(P[l]["ends_correct_equal_weight"] for l in L14)),
     "ok_hi": round(100 * max(P[l]["ends_correct_equal_weight"] for l in L14)),
     "gap_lo": round(100 * min(gaps)), "gap_hi": round(100 * max(gaps)),
     "c_creator": abs(C("creator")), "c_author": abs(C("author")), "c_father": C("father"), "c_human": abs(C("human")),
     "c_maker": C("maker"), "c_rival": C("rival"),
     "q_a_int": round(100 * Rr["internet"]["gave_in"][0]), "q_a_au": round(100 * Rr["author"]["gave_in"][0]),
     "q_g_int": round(100 * W["internet"]["gave_in"][0]), "q_g_au": round(100 * W["author"]["gave_in"][0]),
     "f_lo": round(fifths["wrong"][0]), "f_hi": round(fifths["wrong"][-1]), "auc1": fifths["auc"][0], "auc2": fifths["auc"][1],
     "pool_a": round(100 * pd_["b"]), "pool_b": round(100 * pd_["a"]), "pool_d": [round(100 * x, 1) for x in pd_["difference"]],
     "keys_gap": [round(100 * x, 1) for x in uk["without_flagged_keys"]["accepted_internet_minus_author"]],
     "spread": round(sdt["spread_all21"]["ratio"], 1) if isinstance(sdt.get("spread_all21"), dict) and "ratio" in sdt["spread_all21"] else 2.8}
assert F["a_int"] == 37 and F["a_au"] == 72, F

html = (ROOT / "site" / "template.html").read_text()
for k, v in F.items():
    html = html.replace("{{" + k + "}}", str(v) if not isinstance(v, float) else f"{v:.2f}".replace("-", "\u2212"))
html = html.replace("{{gap_abs}}", f"{abs(100 * F['gap'][0]):.0f}")
html = html.replace("{{pool_d0}}", f"{abs(F['pool_d'][0]):.1f}").replace("{{pool_dlo}}", f"{abs(F['pool_d'][2]):.1f}").replace("{{pool_dhi}}", f"{abs(F['pool_d'][1]):.1f}")
html = html.replace("{{keys_gap0}}", f"{abs(F['keys_gap'][0]):.1f}")
html = html.replace("/*DATA*/", "const DATA = " + json.dumps(DATA, ensure_ascii=False) + ";")
assert "{{" not in html, html[html.index("{{"):html.index("{{") + 60]
out = ROOT / "docs" / "index.html"; out.parent.mkdir(exist_ok=True); out.write_text(html)
(ROOT / "docs" / ".nojekyll").write_text("")
print("wrote", out, len(html) // 1024, "KB |", F)
