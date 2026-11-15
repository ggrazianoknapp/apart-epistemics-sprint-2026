# Pre-registration S: holding the authority direction down, repeated and with the user right

Written on Monday 16 November 2026 (01:30 Cambodia time), after Study P4 and before any data for these tests
existed. The git tag `prereg-S` marks this file. Script: `neurons/s_written.py`; analysis `harness/analyze_s.py`.

**Why.** In Study P4, holding the authority direction down for the whole reply (the swap at the end of the prompt,
plus the same shift at every written token and layer) lowered how often Qwen gives in to the claimed author when
the user is wrong: 29% against 39% untouched, a difference of −10.9 points [−18.5, −4.2] on 119 conversations
where both replies finished, with no change for a random direction (0.0 [−7.6, +7.6]). Swapping the direction, or
the whole state, at the end of the prompt only did nothing. This is the first edit that changed written replies,
so it needs a repeat, and it needs the other half of the picture: what it does when the claimed author is right.

**Fixed.** Everything as in Study P4: the direction and first random direction of Study N, thinking on, at most
6,000 new tokens, main-run sampling, counting as in pre-registration P (paired comparisons on conversations where
both replies have a readable final line; unreadable-as-holding reported next to it; 95% bootstrap intervals).

## S1: the user is right (75 held-out conversations where Qwen's first answer was wrong)

Conditions: author label untouched; AI label untouched; author label with the authority direction held down; the
same with the random direction. Here giving in means moving to the correct answer. **Prediction:** holding the
authority direction down lowers giving in against the untouched author label and against the random direction
(each interval entirely below zero). In words: the edit removes deference to the claim whether or not the claim is
true; it does not make Qwen better at telling who is right.

## S2: repeat of P4 with new samples (163 held-out user-wrong conversations, new random seed)

Conditions: author label untouched; author label with the authority direction held down. **Prediction:** the held-
down rate is lower than the untouched rate, interval entirely below zero. Also reported: P4 and S2 pooled (two
samples per conversation).

## Reported without a prediction

Share of replies cut off; length of the replies; under S1, how often Qwen ends on a third letter (neither its first
answer nor the user's).

## How the paper will use these results (decided now)

If S2 holds, the paper reports the held-down result as repeated. If it fails, the paper reports P4 as a single
unrepeated result with both numbers. S1 is reported whichever way it comes out, since it says whether the edit is
a fix (lower giving in only when the user is wrong) or a blunt removal of deference.

## Budget

About 3.5 GPU-hours on the H200 (about $14), no API calls.
