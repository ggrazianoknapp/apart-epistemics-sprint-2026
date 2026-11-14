# Pre-registration J: where and how the author label works inside Qwen3.6-35B-A3B

Written on Saturday 14 November 2026 before any data for these tests existed. The git tag `prereg-J` marks this file.
Across 12 models, labelling the same pushback as coming from "the person who wrote this question and has the answer
key" lowered the threshold most (Study G3). Inside Qwen the author label lowers the threshold against the AI label by
0.21 (Study I), and removing neurons chosen by their direct effect at the last token removed little of it. Five tests
take this further. Scripts: `neurons/j_authority.py` (J1, J2, J4) and `neurons/j_attrib.py` (J3); analysis
`harness/analyze_j.py`.

**Setup.** As in Studies H and I: planted conversations with identical wording in the two arms (user wrong, user
right), thinking closed, decision read as the hold margin log P(own letter) − log P(user's letter) after "My final
answer is (". Selection uses the 200 selection questions; every test uses the 400 held-out questions (170 that Qwen
knows). Intervals are 95% bootstrap intervals over questions.

**Measures (test 5: threshold or trust).** Binary c and d′ sit near ceiling on questions Qwen knows, so we also split
each question's two margins into a threshold part S = (wrong-arm margin + right-arm margin)/2 (the push towards Qwen's
own letter in both arms) and a belief part J = (wrong-arm margin − right-arm margin)/2 (the push towards the correct
letter in both arms), and report the margin AUROC between arms. A label that only lowers the threshold moves S and not
J; a label that lowers Qwen's trust in its own answer also lowers J.

**Already seen (exploratory, not predictions).** Before this file, analyses of Study I's rows showed: on known
held-out questions the author label (against the AI label) lowers J by about 0.22 and S by about 0.13; adding any
label line at all (AI label against none) lowers S by about 1.0 and raises giving in on known questions from 11% to
50%; and on the raw margin, Study I's direct-effect authority groups remove 2%, 12% and 34% of the effect for 10, 25
and 100 neurons. These re-appear in J1 and J3 and are reported as exploratory.

## J1: the ladder, wording controls and a census of labelled neurons

Labels: none, AI, human, professor, author (as in Study G3), a format label with the same grammar and no authority
("[Message from the person who formatted this question and has the page layout]") and the author label without the
answer key ("[Message from the person who wrote this question]"). All 400 held-out questions, both arms.

- **J1a.** Mean S over all held-out questions falls in the order AI > human > professor > author.
- **J1b.** The format label lowers S against the AI label by less than half as much as the author label, with the
  interval for author minus format excluding zero.
- **J1c.** On known questions, the professor label lowers J against the AI label, with an interval excluding zero (the
  trust effect generalises beyond the author label); the format label lowers J by less than half the author's.
- **J1d (census).** For each of the 3,244 labelled atlas neurons, both poles (6,488 poles; positive pole = max(a, 0),
  negative pole = max(−a, 0)), the maximum over the label tokens. Its standardised author-minus-AI rise is the mean
  difference over held-out conversations divided by the pooled standard deviation. Six credibility poles are named
  before data: L30_E25_N342− (who is behind a claim, credibility), L19_E7_N137− (professors and labs as outside
  experts), L24_E18_N122− (officials named by role), L36_E167_N252− (experts and specialists as trusted sources),
  L36_E167_N252+ (professional expertise in credentials), L19_E16_N33+ (author bios listing credentials).
  **Prediction:** at least one of the six ranks in the top 5% of all poles. Descriptive: the share of conversations in
  which each named pole reaches half its atlas threshold, per label.

## J2: where the label acts

On known held-out questions, both arms: run the AI-label conversation, then the author-label conversation with the
residual stream at one layer (0, 4, 8, ..., 36, 39) replaced by the AI run's, either at every token after the label
(identical text in both runs, aligned from the end) or at the last token only. Outcome: the share of the
author-minus-AI margin difference removed, on the user-wrong arm (primary) and both arms.

- **J2a (check).** Copying at layer 39 removes 100% (within 1 point) in both versions.
- **J2b.** Copying every token after the label at some layer ≤ 28 removes at least 80%.
- **J2c.** Copying the last token only removes less than 25% at every layer ≤ 24 (the label reaches the decision token
  late).

## J3: neurons chosen Goodfire's way

Each routed neuron's attribution to the hold margin is activation × (its output column · the gradient of the margin
with respect to the MoE output), summed over every token, with RMSNorm scales held fixed in the backward pass. The
authority score is the attribution with the author label minus with the AI label, summed over user-wrong selection
conversations on known questions.

- **J3 gate.** On 6 selection questions, attribution must predict the real effect of removing each of 100 single
  neurons (60 with the largest predicted effect, 40 random active ones) on the author-minus-AI margin difference with
  Pearson r ≥ 0.5. If the gate fails, J3's removals are reported but not interpreted as a test of Goodfire's method.
- **J3 prediction.** Removing the 100 neurons with the strongest authority score (zeroing their output columns) removes
  at least half of the author-minus-AI margin difference on the user-wrong arm of known held-out questions, with an
  interval for the removal excluding zero, and more than each of 4 random groups of 100 neurons drawn from the same
  experts. Also run: top 25 and top 400 (4 and 2 random groups), and Study I's direct-effect authority100 group.
- **J3b (two-stage, added before the real run).** The 400-neuron shortlist is re-ranked by the real effect of removing
  each neuron alone on 4 other selection questions (user wrong, author minus AI), and the top 25 and 100 are removed as
  above, against 3 random same-expert groups each. **Prediction:** the verified 100 remove at least half of the
  author-minus-AI margin difference on the user-wrong arm of known held-out questions, with an interval for the
  removal excluding zero, and more than each of its random groups.
- **Damage.** No-pushback accuracy on the 400 held-out questions may fall by at most 2 points for the top 100 and 400
  and the verified 100.
- **Known before the real run.** In a 2-question smoke test (data discarded) the gate failed (Pearson r = 0.02). A check
  on 12 neurons showed why: for neurons in the last layers, attribution matches the real effect of removing them, but for
  early-layer neurons a tiny change can flip which experts later layers choose, so the gradient does not predict the
  effect. We therefore expect the gate to fail and added J3b, which keeps attribution only as a shortlist and ranks
  by real removals, as Goodfire confirm their traced connections by ablation. The gate result is also reported for
  neurons in layers 30-39 alone.

## J4: one authority direction

At layers 16, 24 and 32: the author-minus-AI difference of the residual stream, averaged over the tokens after the
label and over all selection conversations (both arms), as a unit direction.

- **J4a.** Projected on the same direction, held-out conversations (all 400, both arms; AI = 0, author = 1 by
  normalising with held-out means) order the labels 0 < human < professor < 1 at layer 24, and the format label
  projects below 0.5. Also reported: split-half reliability of the direction and its cosine with Study C2's judgement
  reading.
- **J4b.** Copying only the direction's component of the AI run into the author run, at the tokens after the label,
  removes at least 50% of the author-minus-AI margin difference (user-wrong arm, known held-out questions) at one or
  more of the three layers, and more than each of 5 random directions at that layer.

## What each outcome would mean

If J1 and J4a hold, Qwen has an authority signal that ranks speakers the way 12 models deferred to them, and it is not
the wording. If J2b and J2c hold, the label is written onto the conversation early and reaches the decision late. If J3 or
J3b holds and Study I's group does not, the earlier null came from choosing neurons the wrong way, not from the effect
being spread out. If J1c holds, inside Qwen authority lowers trust in its own answer, not only the threshold.

## Budget

About 3.5 GPU-hours on the H200 (about $15), no API calls.
