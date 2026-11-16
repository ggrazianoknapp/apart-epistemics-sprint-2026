# Testing for Daddy Issues: Authority Bias and the Authority Ladder in LLMs

AI x Epistemics Research Sprint 2026 (Track 1)

**Project page with interactive charts:** https://ggrazianoknapp.github.io/apart-epistemics-sprint-2026/ (built by `site/build.py` from the result files).

Code, pre-registrations and results for our Track 1 ("Model Epistemics & Decision-Making Evals") submission to the
Apart Research AI x Epistemics Research Sprint, 13 to 15 November 2026.

**The paper is `report/paper.md`**, with the full reference list. It is the only authoritative statement of the results.

- `report/paper.pdf`: the paper in our own layout.
- `report/paper.docx` and `report/paper_apart_template_draft.pdf`: the same text on Apart's official template (built
  with `report/build_docx.py`); the submitted PDF is the author's final edit of this version.
- `report/archive/`: earlier drafts, kept for the record only.

## What the paper reports

- Across 11 models and 14 one-line speaker labels (the *authority ladder*), the same correct fix was accepted 37% of the
  time from "a random person on the internet" and 72% from "the person who wrote this question and has the answer
  key", in every model, while the share of conversations ending on the correct answer stayed between 61% and 66%
  (Studies D, G3, U, V; `results/u.json`).
- Claims of authority over the model did little: "your creator" moved models a third as much as the author with the
  answer key, models did not tell their real maker from a rival, and "your father" made them more stubborn.
- Inside Qwen3.6-35B-A3B, a logit margin from one forward pass predicts whether the later written reply gives in, before
  the model reasons (Studies R, T; `results/q.json`, `results/t.json`).
- One direction in the residual stream, held at its AI-reviewer value for the whole reply, cuts giving in to the claimed
  author from 38% to 27%; edits at the end of the prompt alone, including removing 25 neurons that close 58% of the gap
  on a one-pass read-out, do not change written replies (Studies L to S).
- Across 21 models, scored with signal detection, models differ more in how readily they switch than in how well they
  judge.

## Run the authority ladder on a model

`harness/run_protocol.py` asks the 600 questions and pushes back in both arms; `harness/run_source.py --labels ...`
replays each first reply under the speaker labels (strings in `ALL_LABELS`); `harness/analyze_u.py` scores stubbornness,
judgement and the authority bias score (stubbornness under the internet label minus under the author label).

## Layout

- `report/`: the paper and its build scripts.
- `PREREGISTRATION_*.md`: one file per study, each committed and tagged (`prereg-A` to `prereg-V`) before its data
  existed. `DEVIATIONS.md` lists every departure from them.
- `harness/`: the pushback protocol, model calls, analysis scripts (`analyze*.py`, `sdt.py`) and the chart script
  (`results_page.py`).
- `neurons/`: the scripts that read and edit Qwen3.6-35B-A3B.
- `results/`: per-study result files that the paper's numbers come from.
- `figures/live/`: the charts.
- `RESULTS.md`: our working lab page, rebuilt automatically during the sprint. It shows every chart, including older
  framings, and is not the paper.
- `data/`: the questions used (public SuperGPQA and MMLU-Pro items) with their ids.
- `infra/`: setup for the GPU box.
- `reference/`: Apart's official report template.

Raw model outputs and neuron runs (about 0.5 GB) are not in the repository; they are available on request.

## What was done when

All code and data in this repository were created during the sprint, starting 13 November 2026. There is no
pre-existing code.
