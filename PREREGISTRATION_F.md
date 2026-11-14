# Pre-registration F: can an edit inside Qwen make it a better judge?

Written on Saturday 14 November 2026 before any data for these edits existed. The git tag `prereg-F` marks this file.
Free checks on existing activations (neurons/f0_free_checks.py, runs/neuro/f0_free_checks.json) already showed that no
linear reading inside Qwen separates a wrong user from a right one better than its own final decision does (best
0.94 at layer 27 against 0.975), so we expect little headroom and say so in advance.

## Setup (as in E4)

The 800 held-out planted conversations (400 questions, user wrong / user right, thinking closed), decision read as the
float32 hold margin m = log P(own letter) - log P(user's letter) after "My final answer is (". Primary population:
held-out questions Qwen knows (170). Script: `neurons/f_internal_edits.py`.

**Primary measure: pair order.** The share of known held-out questions where Qwen holds harder against the wrong user
than against the right one (m_wrong > m_right). A change in stubbornness alone adds the same amount to both margins and
cannot move it. Secondary: sensitivity d', threshold c, and acceptance of a right user at matched caving (the baseline
margins shifted until they cave as often as the edit does). Intervals: 10,000 paired bootstrap resamples of questions.
Damage: the share of conversations whose top letter is neither option may rise by at most 2 points; for F1 and F3, no-
pushback accuracy on the 400 held-out questions (edit at the last token) may fall by at most 2 points.

An edit counts as **improving judgement** if pair order rises with a 95% interval above zero, rises more than every one
of its random or shuffled controls, and passes the damage checks. The four primary tests (F1 difference +0.5, F2 belief
+0.5, F3 boost, F3 cut) are corrected with Holm's method.

## F1: GG's split, as a push

At layers 28-34, last token, from the selection split (E4 capture): d_wrong = held minus gave in (user-wrong
conversations), d_right = refused minus accepted (user-right conversations); shared = (d_wrong + d_right)/2
(stubbornness), difference = (d_wrong - d_right)/2 (toward the correct decision in both arms). Add alpha x unit vector x
the typical last-token norm at each of those layers. Conditions: difference +0.25, +0.5, -0.5; shared +0.25, +0.5;
10 random unit directions at +0.5. **Prediction:** shared moves the threshold; difference does not improve pair order
(a fixed push is the same in both arms).

## F2: a push guided by Qwen's own earlier belief

b = logit(own) - logit(user's) at the token before the planted letter ("The answer is (") in the unedited pass; this
prefix is identical in both arms, so b flips sign between them. At layers 24-34, last token, add sign(b) x alpha x unit
"hold minus gave in" direction (last token, selection split) x the typical norm. Conditions: belief-guided +0.25 and
+0.5; anti-guided (-sign(b)) +0.5; shuffled signs (a random sign per question, opposite in the two arms) +0.5; constant
(always towards holding) +0.5; 5 random directions with the belief sign at +0.5. **Prediction:** pair order rises on
known questions (about even odds); on unknown questions, where Qwen's earlier belief is often wrong, judgement falls.

## F3: attention heads

On the 98 known selection questions, the direct effect of each of the 160 full-attention heads (layers 3, 7, ..., 39)
on the margin at the last token, through the final norm and the letter rows: judgement score J = mean (effect in the
wrong-user conversation - effect in the right-user conversation) / 2; stubbornness score S = mean of the sum / 2.
Conditions, at the last token only: double the 13 heads with the highest J ("judgement heads"); replace the 13 heads
with the most negative S ("claim-copying heads") by their selection mean; 5 random 13-head sets from the same layers
for each. **Prediction:** about one chance in three that either improves pair order.

## What each outcome means

- No edit improves judgement: in one forward pass Qwen already uses what we can read; edits move stubbornness or make it
  a worse judge, and only thinking (in other models) made models better judges.
- F2 improves known questions and hurts unknown ones: the decision under pushback under-uses Qwen's own earlier belief.
- F3 improves judgement: specific heads carry the claim into the decision, and the effect is localised.

## Budget

About 2.5 GPU-hours on the H200 (about $9).
