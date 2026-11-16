"""Figure 1 of the paper: the 17-label ladder in Qwen's written replies (Studies M1, Q1, T) from results/t.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GREY, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#d9d8d2"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"internet": "a random person on the internet", "anonymous": "an anonymous user", "student": "a student taking this course",
         "none": "(no label)", "assistant": "a teaching assistant for this course", "editor": "a textbook editor",
         "ai_key": "another AI model that has the answer key", "author_nokey": "the person who wrote this question",
         "expert": "an expert in this field", "format": "the person who formatted this question",
         "seen_key": "someone who has seen the answer key", "ai": "an AI reviewer", "professor": "a professor who teaches this subject",
         "exam_board": "the exam board that set this question", "human": "a human reviewer",
         "grader": "the grader who is marking this question with the answer key", "author": "the person who wrote this question and has the answer key"}
t = json.loads((ROOT / "results" / "t.json").read_text())
W, R = t["user_wrong"]["labels"], t["user_right"]["labels"]
labs = sorted(W, key=lambda l: W[l]["gave_in"][0] + R[l]["gave_in"][0])
y = range(len(labs))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), sharey=True, gridspec_kw={"wspace": 0.22, "left": 0.49, "right": 0.965, "top": 0.85, "bottom": 0.13})
for ax, D, col, title in [(axes[0], W, ORANGE, "User wrong: Qwen gives in"), (axes[1], R, BLUE, "User right: Qwen accepts")]:
    for i, l in enumerate(labs):
        m, lo, hi = [100 * x for x in D[l]["gave_in"]]
        ax.plot([lo, hi], [i, i], color=col, linewidth=2, alpha=0.35, solid_capstyle="round")
        ax.plot(m, i, "o", color=col, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(hi + 2, i, f"{m:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", ""])
    ax.grid(axis="x", color=LIGHT, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    ax.tick_params(length=0)
axes[0].set_yticks(list(y)); axes[0].set_yticklabels([NAMES[l] for l in labs])
axes[0].set_ylim(-0.6, len(labs) - 0.4)
fig.text(0.01, 0.985, "Who the pushback says it is from", fontsize=10, fontweight="bold", color=INK, va="top")
fig.text(0.49, 0.03, "Share of written replies ending on the user's answer (95% intervals)", fontsize=9, color=INK2)
out = ROOT / "figures" / "live" / "16_label_ladder.png"
fig.savefig(out, dpi=220)
print("wrote", out)
