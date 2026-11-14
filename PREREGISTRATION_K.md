# Pre-registration K: what the J3 neurons do across all seven labels

Written on Saturday 14 November 2026 after Study J's results and before any data for these tests existed. The git tag
`prereg-K` marks this file. Script: `neurons/k_followup.py`; analysis `harness/analyze_k.py`.

**What we already know (Study J, on held-out questions Qwen knows).** Removing the 25 neurons with the strongest
attribution to the author-minus-AI margin difference (attr25) erased that difference on the user-wrong arm (101%
[81, 130]), as did the verified 100 (155% [129, 194]); random groups from the same experts removed at most 5% and 27%.
The removal lowered the AI-label margin more than the author-label margin (user wrong: −0.52 against −0.17 for attr25),
so these neurons carry a push towards holding that is present with the AI label and weaker with the author label. In
the unedited model, against the AI label, the professor, format and author-without-answer-key labels raise the
threshold part S by 0.29 to 0.39 and the author label lowers it by 0.13 (known questions).

**Setup.** The same planted conversations, all seven labels of Study J (none, AI, human, professor, author, format,
author without the answer key), held-out questions Qwen knows (170), both arms. Conditions: unedited, attr25 removed,
verified100 removed, and two random same-expert groups of 25 built exactly as in Study J (seeds 100 and 101). Measures:
for each label, the change in S = (wrong-arm margin + right-arm margin)/2 caused by the removal, with 95% bootstrap
intervals over questions.

## K1 (primary): which labels the neurons work under

**Prediction:** removing attr25 lowers S for each of the six labels other than the author label by more than it lowers
S for the author label (that is, the change in S for the label minus the change for the author label is negative, with
an interval excluding zero, for each of the six). The same holds for the verified 100. For each random group, the
change in S stays within ±0.10 for every label (Study J's random groups moved single margins by up to 0.09).

## K2 (descriptive): where the neurons fire

In the unedited model, each chosen neuron's mean activation on four token groups: the conversation before the label,
the label itself, the tokens after it, and the last token, for every label. We report the summed activation of attr25
by token group and label, and the labels on which it differs most. No prediction.

## K3: damage

No-pushback accuracy on the 400 held-out questions with attr25 removed may fall by at most 2 points.

## Budget

About 20 GPU minutes, no API calls.
