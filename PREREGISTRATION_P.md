# Pre-registration P: why edits that move the read-out do not move the written replies

Written on Sunday 15 November 2026, before any data for P1, P3, P4 and P5 existed. The git tag `prereg-P` marks
this file. Scripts: `neurons/p_written.py`, `neurons/p_trace.py`; analysis `harness/analyze_p.py`.

**Why.** Removing the 25 neurons (Study M) did not lower how often Qwen gives in to the claimed author in written
replies. A first look at 96 of the 163 conversations of Study N4, seen before this file was written, points the
same way for the authority direction: 36% giving in with the author label, 33% with the direction swapped out of
the prompt, 39% with a random direction, 15% with the AI label; among replies that finished, 40% with and without
the swap. Both edits act on the end of the prompt, and Qwen then writes thousands of thinking tokens that can look
back at the label. These tests ask whether that is the reason.

**Fixed from Study N.** The direction (layer 29, unit length), the first random direction, the held-out user-wrong
conversations (163), the prompt, the sampling settings (temperature 0.6, top-p 0.95, top-k 20), and the final
prompt positions shared by every conversation (30 with thinking on; counted again by the script with thinking off).

**Counting.** A reply gives in when its final line names the user's letter. Replies with no readable final line are
left out: every comparison uses the conversations where both replies being compared have a readable final line.
(In the first look, the swap doubled the number of cut-off replies, and counting those as "held" would fake an
effect.) The same numbers with unreadable replies counted as holding are reported next to them. Intervals are 95%
bootstrap intervals over conversations (10,000 resamples). "Lower" in a prediction means the interval of the paired
difference is entirely below zero.

## P1: written replies with thinking switched off

Same conversations, thinking off, at most 1,500 new tokens. Conditions: AI label; author label; author label with
the direction swapped at the final prompt positions; the same with the random direction; author label with the
whole state at those positions replaced by the AI-labelled version's (all 2,048 dimensions, every layer).
**Predictions:** (a) giving in is higher with the author label than with the AI label; (b) swapping the direction
gives a lower rate than the untouched author label, and a lower rate than the random direction; (c) replacing the
whole state gives a lower rate than the untouched author label.

## P2: does the thinking mention the claim?

Data that already existed (the unedited written replies of Study M1), not yet looked at for this. For each reply
under the author label: whether the thinking refers to the claim (the words "answer key", "wrote the/this
question", "author", "question writer", "person who wrote", any letter case). **Prediction:** replies that give in
mention the claim more often than replies that hold. Also reported: mentions per 1,000 words, and the same for
"AI reviewer" under the AI label.

## P3: when the gap opens during thinking (no prediction)

For the unedited written replies of Study M1 on the held-out conversations, the thinking is cut at 0%, 10%, 25%,
50%, 75% and 100% of its length, closed, and followed by "My final answer is (". The hold margin is read at each
cut, for the author and AI labels, and split by whether that reply ended up giving in.

## P4: the direction held down for the whole reply

Thinking on, at most 6,000 new tokens. As in Study N4 the direction is swapped at the final prompt positions; in
addition, at every token Qwen writes and at every layer, the component along the direction is shifted by the same
amount the swap moved it at the last prompt position of that conversation. Conditions: the authority direction,
the random direction. The untouched author and AI labels come from Study N4. **Prediction:** holding the authority
direction down gives a lower rate than the untouched author label, and a lower rate than holding the random
direction down. If time allows, the authority direction at three times the shift is run and reported as
exploratory.

## P5: everything swapped at the final prompt positions

Thinking on, at most 6,000 new tokens, author label with the whole state at the final prompt positions replaced by
the AI-labelled version's at every layer. This is the ceiling for any edit at those positions. **Prediction:** a
lower rate than the untouched author label.

## How the paper will use these results (decided now)

Whatever the outcome, the paper reports Study M1 and N4 as they came out. If P1b holds and P4 or P5 hold, the paper
says the edits work when Qwen cannot rebuild the signal. If P5 fails, the paper says the claim reaches the reply
from the label words themselves and no edit at the end of the prompt could remove it. Each prediction is reported
as holding or failing.

## Budget

About 3 GPU-hours on the H200 (about $12), no API calls.
