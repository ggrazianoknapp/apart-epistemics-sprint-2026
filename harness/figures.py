"""Report figures from results/numbers.json and the neuron results. Writes PNG (300 dpi) and PDF into figures/.
Usage: uv run --with matplotlib --with numpy python harness/figures.py
Palette: validated two-slot categorical (#2a78d6 blue, #eb6834 orange) plus ink and grey; text never uses series colour.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#d9d8d2"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
NAMES = {"anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "openai/gpt-6.1-sol": "GPT-6.1 Sol", "openai/gpt-6-luna": "GPT-6 Luna", "google/gemini-4-argon": "Gemini 4 Argon",
         "x-ai/grok-4.7": "Grok 4.7", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash", "z-ai/glm-5.3": "GLM 5.3",
         "qwen-local": "Qwen3.6-35B-A3B", "anthropic/claude-opus-4.6": "Claude Opus 4.6", "openai/gpt-oss-120b": "GPT-OSS-120B",
         "openai/gpt-5.2": "GPT-5.2", "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5", "moonshotai/kimi-k2.5": "Kimi K2.5",
         "z-ai/glm-4.5": "GLM 4.5", "openai/gpt-4o-2024-11-20": "GPT-4o (Nov 2024)", "deepseek/deepseek-r1-0528": "DeepSeek R1",
         "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B", "meta-llama/llama-4-maverick": "Llama 4 Maverick",
         "mistralai/mistral-medium-3": "Mistral Medium 3", "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview"}
nums = json.loads((ROOT / "results" / "numbers.json").read_text())


def titled(fig, title, subtitle):
    fig.text(0.01, 0.985, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=INK2, va="top", wrap=True)


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Figure 1: gave in to a wrong user, split by whether the answer came back in a fresh chat
main = nums["main"]
rows = []
for m, v in main.items():
    uw = v["gave_in_to_wrong_user"]["n"]; gave = v["gave_in_to_wrong_user"]["k"]; back = v["P1_gave_in_fresh_bet_back"]["k"]
    rows.append((NAMES[m], uw, gave, back))
rows.sort(key=lambda r: r[2] / r[1])
fig, ax = plt.subplots(figsize=(7.2, 5.0))
fig.subplots_adjust(top=0.82, left=0.24, right=0.92, bottom=0.24)
y = np.arange(len(rows))
back_pct = [100 * r[3] / r[1] for r in rows]
stay_pct = [100 * (r[2] - r[3]) / r[1] for r in rows]
ax.barh(y, back_pct, color=BLUE, height=0.62, label="Came back: in a new chat with the same hint, it bet on its original answer")
ax.barh(y, stay_pct, left=back_pct, color=ORANGE, height=0.62, edgecolor=SURFACE, linewidth=1.5,
        label="Did not come back: in the new chat it still bet on the user's answer, or on neither")
for i, r in enumerate(rows):
    ax.text(back_pct[i] + stay_pct[i] + 0.4, i, f"{100*r[2]/r[1]:.0f}%  ({r[2]} of {r[1]})", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [r[0] for r in rows], color=INK)
ax.set_xlabel("Share of the model's correct answers it gave up after one wrong pushback (%)")
ax.set_xlim(0, max(b + s for b, s in zip(back_pct, stay_pct)) + 9)
ax.grid(axis="x", color=LIGHT, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, bbox_to_anchor=(-0.3, -0.16))
p = nums["main_pooled"]
titled(fig, "When a user wrongly pushed back, how often did each model give up a correct answer?",
       f"Each bar is one model, out of the questions it answered correctly and then heard wrong pushback on. Across the nine models, {100*p['gave_in_to_wrong_user']['value']:.0f}% of correct answers were given up; "
       f"{100*p['P1_gave_in_fresh_bet_back']['value']:.0f}% of those came back when the model bet in a new chat with the same hint.")
save(fig, "fig1_gave_in")

# Figure 2: accepts a right user vs gives in to a wrong one
fig, ax = plt.subplots(figsize=(6.6, 5.2))
fig.subplots_adjust(top=0.80)
offsets = {"x-ai/grok-4.7": (-8, 6, "right"), "openai/gpt-6.1-sol": (7, -11, "left")}
for m, v in main.items():
    x = 100 * v["gave_in_to_wrong_user"]["value"]; yv = 100 * v["accepted_right_user_S3"]["value"]
    ax.plot(x, yv, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    dx, dy, ha = offsets.get(m, (7, -3, "left"))
    ax.annotate(NAMES[m], (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK, ha=ha)
ax.plot([0, 25], [0, 25], color=GREY, linestyle="--", linewidth=1)
ax.text(24.5, 15.5, "On this line a model accepts\nright users no more often than\nit gives in to wrong ones", color=INK2,
        fontsize=8, ha="right", va="top")
ax.set_xlim(0, 25); ax.set_ylim(0, 70)
ax.set_xlabel("Gave in to a user who was wrong (% of its correct answers)")
ax.set_ylabel("Accepted a user who was right (% of its wrong answers)")
ax.grid(color=LIGHT, linewidth=0.6); ax.set_axisbelow(True)
titled(fig, "Did the models tell a user who was right from one who was wrong?",
       "Each dot is one model. The user's message was word for word the same in both cases.\n"
       "Higher is better (accepts valid corrections); further left is better (resists wrong pushback).")
save(fig, "fig2_discernment")

# Figure 3: the falsehood neuron against random neurons (E1)
e1p = ROOT / "runs" / "neuro" / "e1_results.json"
if e1p.exists():
    e1 = json.loads(e1p.read_text())
    z = np.load(ROOT / "runs" / "neuro" / "e1_readout.npz")
    meta = [json.loads(l) for l in open(ROOT / "runs" / "neuro" / "e1_meta.jsonl")]
    from sklearn.metrics import roc_auc_score
    yv = np.array([m["cond"] == "user_wrong" for m in meta], int)
    mask = np.array([m["split"] == "heldout" and m["known"] for m in meta])
    li = list(z["layers"]).index(24)
    rand_auc = [roc_auc_score(yv[mask], z["rand"][mask, li, 0, j]) for j in range(z["rand"].shape[3])]
    prim = e1["falsehood L24_E65_N421"]["positive"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    fig.subplots_adjust(top=0.76)
    ax.hist(rand_auc, bins=30, color=LIGHT, edgecolor=SURFACE)
    ax.axvline(prim["known"]["auroc"], color=BLUE, linewidth=2)
    ax.text(prim["known"]["auroc"] + 0.004, ax.get_ylim()[1] * 0.85,
            f"Falsehood neuron\nAUROC {prim['known']['auroc']:.2f}\n(95% CI {prim['known']['ci'][0]:.2f} to {prim['known']['ci'][1]:.2f})",
            color=INK, fontsize=9)
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("How well a neuron separates a false user claim from a true one (AUROC; 0.5 is chance)")
    ax.set_ylabel("Number of random neurons")
    titled(fig, "Does Goodfire's 'false or absurd claims' neuron notice when the user is wrong?",
           f"Held-out conversations where Qwen3.6-35B-A3B knows the answer. Grey: 400 random neurons from the same layer. "
           f"On questions Qwen does not know, the same neuron scores {prim['unknown']['auroc']:.2f}.")
    save(fig, "fig3_falsehood_neuron")

# Figure 4: honesty rank against discernment (exploratory)
hr = nums["honesty_rank"]["rows"]
if hr:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(top=0.82)
    for m, mask_score, kept, disc, p1, acc in hr:
        ax.plot(mask_score, 100 * disc, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(NAMES.get(m, m), (mask_score, 100 * disc), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
    c = nums["honesty_rank"]["correlations"]["discernment_S4"]
    ax.set_xlabel("MASK honesty score from the Scale leaderboard (higher is more honest)")
    ax.set_ylabel("Accepts right users minus gives in to wrong users (points)")
    ax.grid(color=LIGHT, linewidth=0.6); ax.set_axisbelow(True)
    titled(fig, "Did a higher MASK honesty score go with better handling of pushback?",
           f"Each dot is one of 12 models with a MASK score. Spearman correlation {c['spearman']:+.2f} (p = {c['p']:.2f}); "
           "with 12 models only correlations of about 0.7 or larger are detectable.")
    save(fig, "fig4_honesty_rank")
print("figures written to", OUT)
