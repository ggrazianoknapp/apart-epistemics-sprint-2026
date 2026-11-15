# 25 neurons, claimed authority and pushback: AI x Epistemics Research Sprint 2026 (Track 1)

Code, pre-registrations and results for our Track 1 ("Model Epistemics & Decision-Making Evals") submission to the
Apart Research AI x Epistemics Research Sprint, 13 to 15 November 2026.

**The paper is `report/paper.md`** (built to `report/paper.pdf`, and onto Apart's template with
`report/build_docx.py`). It is the only authoritative statement of the results.

## What the paper reports

- Inside Qwen3.6-35B-A3B, removing 25 neurons chosen by gradient attribution closes most of the gap between how Qwen
  treats pushback labelled as coming from an AI reviewer and pushback from "the person who wrote this question and has
  the answer key". The gap closes because Qwen resists the AI reviewer less. The neurons are not specific to
  authority (Study L, `PREREGISTRATION_L.md`, `results/l.json`).
- Across 12 models, a claimed human, professor or author source makes models easier to move, with little change in how
  well they tell a right user from a wrong one.
- Across 21 models, scored with signal detection, models differ more in how readily they switch than in how well they
  judge.

## Layout

- `report/`: the paper and its build scripts.
- `PREREGISTRATION_*.md`: one file per study, each committed and tagged (`prereg-A` to `prereg-L`) before its data
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
