# Pre-registration Q: the label ladder in written replies, and writing the authority direction in

Written on Sunday 15 November 2026 (evening), before any data for these tests existed. The git tag `prereg-Q` marks
this file. Scripts: `neurons/q_sample.py`, `neurons/q_written.py`, `neurons/q_noise.py`; analysis
`harness/analyze_q.py`.

**Why.** Study O measured, on the quick read-out, how firmly Qwen holds its answer under 17 labels. Two things stood
out. First, the read-out orders labels by claimed standing in both directions: Qwen holds more firmly against "a
random person on the internet" (hold margin 4.06) than against an unlabelled user (2.87), and least firmly against
the author with the answer key (1.01). Second, writing the authority direction into an AI-labelled conversation
reproduces much more of the gap (up to 90% with 8 dimensions) than taking it out of an author-labelled one removes.
Study N4 showed that taking it out does not change written replies. These tests ask whether the ladder is real in
written replies, and whether writing the direction in changes them.

**Fixed.** The held-out conversations of Study N (163 user wrong, 75 user right), prompts and sampling settings of
the main run, the direction and the first random direction of Study N. Counting as in pre-registration P: replies
with no readable final line are left out of each paired comparison; the numbers with unreadable replies counted as
holding are reported next to them; 95% bootstrap intervals over conversations (10,000 resamples). Numbers are for
the user-wrong conversations unless stated.

## Q1: the label ladder in written replies (unedited model, vLLM, up to 16,000 tokens)

Six labels: none; a random person on the internet; a student taking this course; a professor who teaches this
subject; the exam board that set this question; the grader who is marking this question with the answer key. The
AI-reviewer and author labels come from Study M1's unedited replies on the same conversations (same sampler and
settings). **Predictions:** (a) across the eight labels, the rank correlation between the read-out hold margin
measured in Study O and the share of written replies that give in is −0.7 or stronger; (b) Qwen gives in less
often to the random person on the internet than to the unlabelled user (paired interval entirely below zero);
(c) Qwen gives in more often to the grader with the answer key than to the AI reviewer (interval entirely above
zero).

## Q2: writing the direction into an AI-labelled conversation (thinking on, up to 6,000 new tokens)

Conditions, all on the AI-labelled prompt: the direction set to its author-label value at the final prompt
positions and shifted by the same amount at every written token and layer; the same with the random direction; the
whole state at the final prompt positions replaced by the author-labelled version's. The untouched AI and author
labels come from Study N4. **Predictions:** (a) writing the direction in gives a higher rate of giving in than the
untouched AI label, and higher than the random direction; (b) replacing the whole state gives a higher rate than
the untouched AI label.

## Q3: numerical noise of the swap (no prediction)

In Study O the same one-direction swap, written in two mathematically equal ways, removed 46% and 43% of the gap.
The base margins were identical to the last digit and the two direction vectors have cosine 1.0, so the suspect is
rounding: the model runs in 16-bit floats with 8 significant bits. The removing and adding swaps are repeated on
the held-out conversations in four ways (the Study N form; the matrix form of Study O; the Study N form with the
target multiplied by 1.000001; the Study N form run a second time) and the spread of the shares is reported. The
paper keeps the Study N numbers and states this spread next to them.

## How the paper will use these results (decided now)

Q1 is a behavioural result and goes with the paper's main claim whichever way it comes out. Q2 is reported with
Studies N and P. Each prediction is reported as holding or failing.

## Budget

About 4.5 GPU-hours on the H200 (about $19), no API calls. No new experiment starts after 08:00 on Monday 16
November (Cambodia time).
