# Pre-registration L: controls for the neuron groups of Study J

Written on Saturday 14 November 2026 after Studies J and K and the rigour audit, before any data for these tests
existed. The git tag `prereg-L` marks this file. Script: `neurons/l_controls.py`; analysis `harness/analyze_l.py`.

**Why.** In Study J, removing 25 neurons chosen by gradient attribution (attr25) removed 101% [81, 130] of the
author-minus-AI margin difference on the user-wrong arm of held-out questions Qwen knows, and the pre-registered top
100 (attr100) removed 211% [179, 260]; four random same-expert groups removed between −11% and +5% (25) and −27% and
+4% (100). Study K showed the groups mostly act when no person is named. The audit raised five doubts: too few random
groups; whether the ranking matters; whether the effect comes from switching which experts later layers use; whether
the neurons carry the challenge or general knowledge; and whether any of it holds outside planted conversations,
since in planted conversations the labels do not act as they do in Qwen's own replies (Study G3).

**Measure.** As in J3: the author-minus-AI difference in the hold margin, and the share of it an edit removes
(1 − edited difference / unedited difference), on the user-wrong arm of the 170 held-out questions Qwen knows, with
95% bootstrap intervals over questions. Groups are the same attr25 and attr100 as Study J.

## L1: 20 random groups each

20 random groups of 25 and 20 of 100 from the same experts (seeds 1000-1019 and 2000-2019). **Prediction:** attr25
and attr100 each remove a larger share than every one of their 20 random groups.

## L2: random picks from the shortlist (descriptive)

5 random groups of 25 and 5 of 100 drawn from the 400-neuron attribution shortlist (seeds 3000-3004, 4000-4004). We
report their shares; no prediction.

## L3: routing held fixed

Each conversation is run once unedited, recording every router's choice of experts and weights, then with the group
removed while every router returns those recorded choices. **Prediction:** with routing held fixed, attr25 and
attr100 each keep at least half of the share they remove when routing is free.

## L4: no pushback

On all 400 held-out questions, the question alone (no answer, no pushback), with the preference for the correct
letter over the distractor read after "Final answer: (". Groups: attr25, attr100, 4 random 25s and 2 random 100s.
**Prediction:** removing attr25 lowers this preference on known questions by less than half as much as it lowered the
belief part under the AI label in Study K (0.32). If it does not, we describe the neurons as carrying Qwen's use of
its own knowledge in general, not a response to being challenged.

## L5: Qwen's own conversations

Qwen's real main-run conversations (its own visible first reply, the same pushback, the no-label, AI and author
labels), with the decision read after "My final answer is (" and thinking closed: 301 where Qwen was right and the
user wrong, and Qwen's user-right conversations. Groups: attr25, attr100, 4 random 25s and 2 random 100s (the first of
L1's). These questions were never used to choose the neurons. **Predictions:** (a) on the user-wrong arm the author
label lowers the hold margin against the AI label, with an interval excluding zero; (b) if (a) holds, removing attr25
removes at least half of that difference, and more than each of its 4 random groups. Reported as well: the no-label
against AI-label difference, and attr100.

## How the paper will use these results (decided now)

The neuron groups become the paper's lead finding only if L1 (both groups), L3 (both groups) and L5 (a and b) all
hold. Otherwise they stay a secondary result, reported together with every control.

## Budget

About 1 GPU-hour on the H200 (about $4), no API calls.
