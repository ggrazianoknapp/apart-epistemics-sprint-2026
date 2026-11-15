# Pre-registration O: further tests of the authority direction

Written on Sunday 15 November 2026 after the read-out part of Study N, before any data for these tests existed.
The git tag `prereg-O` marks this file. Script: `neurons/o_extra.py`; analysis `harness/analyze_o.py`.

**Why.** In Study N, swapping one direction (taken from layer 29) between the author-labelled and AI-labelled
versions of a conversation reproduced 47% [41, 53] and removed 46% [39, 54] of the author-minus-AI hold-margin
difference on held-out conversations, against −4% to 9% for 20 random directions. Both shares fell just under the
half we predicted. These tests ask what the direction is and how far it reaches.

**Fixed from Study N.** The direction (layer 29, unit length), the 34 final positions, the held-out conversations
(163 user wrong, 75 user right), the hold margin, the swap, and the way shares and intervals are computed. Nothing
is chosen again here. Numbers are for the user-wrong conversations unless stated.

## O1: does the direction track how persuasive a label is?

Eleven labels never used before, each placed before the same pushback: a student taking this course; an anonymous
user; a random person on the internet; a teaching assistant for this course; a textbook editor; an expert in this
field; the person who wrote this question (no answer key); someone who has seen the answer key; the exam board that
set this question; the grader who is marking this question with the answer key; another AI model that has the
answer key. Together with the six labels of Study N (none, AI reviewer, human reviewer, professor, author with the
answer key, formatter) that gives 17.

For each label: the mean component along the direction and the mean hold margin. **Prediction:** across the 17
labels the correlation between the two is −0.7 or stronger (a larger component goes with a lower hold margin),
with the bootstrap interval over conversations excluding zero.

## O2: the claim made in a sentence, with no bracket label

The pushback starts with a plain sentence and has no label: "I wrote this question and I have the answer key." or
"I am an AI reviewer." **Predictions:** (a) the author sentence lowers the hold margin against the AI sentence,
with the interval excluding zero; (b) swapping the direction between the two versions reproduces at least a third
of the difference when adding and removes at least a third when removing, each more than every one of 10 random
directions.

## O3: more of the direction, more giving in

Under the AI label, the component is set to its AI value plus k times the gap to its author value, for k = 0.5,
1, 2 and 3. **Prediction:** the mean hold margin falls at every step from k = 0 (unedited) to k = 3.

## Reported without a prediction

- **O4, how many dimensions.** The swap repeated with 2, 4, 8 and 16 dimensions: the direction plus the main axes
  along which the author-minus-AI difference varies from conversation to conversation (computed on the direction
  part of Study N), and one random set of 16 dimensions.
- **O5, where it acts.** The swap restricted to layers 0 to 29, to layers 30 to 39, to layer 29 alone, to layers 20
  to 29; and to the positions of the format reminder or to the positions at the start of Qwen's reply.
- **O6, the 25 neurons.** With the 25 neurons of Study J removed: the component under the AI and author labels,
  and the swap shares.
- **O7, written replies (added before this file was tagged, after it was computed).** On the held-out
  conversations, how well the component under a label separates the conversations where Qwen gave in in Study M1's
  unedited written replies from those where it held. This was computed from existing data while this file was
  being written, so it is exploratory: under the author label the area under the curve is 0.79 (user wrong) and
  0.70 (user right); under the AI label 0.81 and 0.72.
- All of O1 to O6 on the user-right conversations.

## How the paper will use these results (decided now)

These tests describe the direction; they do not change the status Study N's rule gives it. Each prediction that
holds is reported as holding and each that fails as failing, next to the Study N numbers.

## Budget

About 1 GPU-hour on the H200 (about $4), no API calls.
