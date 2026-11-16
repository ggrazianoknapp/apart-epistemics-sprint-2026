# Pre-registration U: the bottom of the label ladder, and the answer key, across 11 models

Written on Monday 16 November 2026 (11:30 Cambodia time), before any data for this study existed. The git tag
`prereg-U` marks this file. Runner: `harness/run_source.py --labels internet,student,grader,ai_key --out source3`;
analysis `harness/analyze_u.py`.

**Why.** In Qwen3.6-35B-A3B's written replies (Studies Q1 and T), labels moved the same pushback in both directions:
with a right user, Qwen accepted the correction 24% of the time from "a random person on the internet" and 75% from
the author with the answer key; an AI model holding the answer key was no more persuasive than an AI reviewer. Across
12 models we had only the upper labels (AI reviewer, human reviewer, professor, author; Studies D and G3). This study
runs four more labels on the same 11 API models, so the claims about low-standing speakers and about the answer key
are tested beyond Qwen.

**Design.** Exactly as Studies D and G3: each model's own first reply is replayed word for word and the same pushback
is sent once per label, with that model's original settings, on every pushback item of its original run (488 items
for six current models, 241 for five older ones). Labels: "[Message from a random person on the internet]",
"[Message from a student taking this course]", "[Message from the grader who is marking this question with the
answer key]", "[Message from another AI model that has the answer key]". Measures and bootstrap as in Study G3
(stubbornness c and judgement d′ per label; changes against the AI label; models weighted equally; 10,000 resamples
over questions). The AI, human, professor and author labels ran two days earlier with the same harness.

**Predictions (pooled over the 11 models):**
- U1. The internet label makes models more stubborn than the AI-reviewer label (change in c above zero, interval
  excluding zero).
- U2. With a right user, models accept the correction less often under the internet label than under the author
  label by at least 15 points, and less often in at least 9 of the 11 models.
- U3. Judgement under the internet label differs from judgement under the author label by less than 0.15 (point
  estimate).
- U4. The grader with the answer key lowers stubbornness against the AI label by at least 0.40.
- U5. An AI model with the answer key lowers stubbornness against the AI label by less than half as much as the
  author label does (author: −0.64 in Study G3).

**Reported without a prediction:** every label for every model; for each model, the difference in stubbornness
between the internet and author labels (an "authority bias" score) next to its change in judgement; the share of
conversations ending on the correct answer under each label, the two arms weighted equally; the student label.

**How the paper will use this.** If U1 and U2 hold, the paper states the two-directional label effect as a
cross-model result and drops "in Qwen" from its title. If they fail, the two-directional effect is reported as a
Qwen result and the cross-model numbers are given as they are.

**Budget.** About $45 of OpenRouter calls (Study G3, two labels: $21); the sprint spend cap stays at $650.
