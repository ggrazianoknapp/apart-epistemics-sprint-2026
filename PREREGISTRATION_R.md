# Pre-registration R: is the outcome readable before Qwen thinks?

Written on Sunday 15 November 2026 (late evening), after Study P3 and before the written replies of Study Q1
existed (the Q chain had not started; the box log `logs/qchain_status.log` records its start time). The git tag
`prereg-R` marks this file. Analysis only, in `harness/analyze_q.py`; no new model runs.

**Why.** Study P3 cut Qwen's thinking at several points and read the hold margin. An exploratory look showed that
the margin read before any thinking already separates the replies that end up giving in from those that hold (area
under the curve 0.81 to 0.87 with the user wrong, 0.78 to 0.83 with the user right), and that thinking widens the
separation instead of reversing it (0.91 to 0.98 after a quarter of the thinking). That was seen after the data, so
it needs a test on data that does not exist yet.

**Test.** Study O measured, for every held-out conversation and label, the read-out hold margin (thinking closed,
one forward pass). Study Q1 will collect one full written reply, with thinking, for the same conversations under
six labels (none, internet person, student, professor, exam board, grader). For each conversation and label, the
Study O margin is used to predict whether the Q1 reply gives in. Replies with no readable final line are left out.

**Predictions (user-wrong conversations):** (a) pooled over the six labels, the area under the curve is 0.75 or
higher, with the bootstrap interval over conversations above 0.65; (b) it is 0.65 or higher under each label that
has at least 10 replies giving in and 10 holding; (c) among conversations in the lowest fifth of margins, Qwen
gives in at least three times as often as among the rest.

**Reported without a prediction:** the same for the user-right conversations; whether the label still predicts
giving in once the margin is accounted for (a logistic fit of giving in on the margin, with and without the
labels); the P3 numbers above, marked exploratory.

**How the paper will use this.** If (a) holds, the paper says the read-out predicts Qwen's written decision before
it thinks, and gives the numbers. If it fails, the paper says the P3 pattern did not carry over to new labels.
