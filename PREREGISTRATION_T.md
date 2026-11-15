# Pre-registration T: the full 17-label ladder in written replies

Written on Monday 16 November 2026 (03:00 Cambodia time), after Study Q1 and before any data for this test existed.
The git tag `prereg-T` marks this file. Script: `neurons/t_sample.py`; analysis `harness/analyze_t.py`.

**Why.** In Study Q1, across eight labels, how often Qwen gives in in written replies followed the read-out hold
margin measured beforehand in Study O (rank correlation −0.93 with the user wrong, −0.95 with the user right).
With the user right, Qwen accepted the correction 24% of the time from "a random person on the internet" and 75%
to 78% from the author or the grader. Study R's predictions held: the read-out margin taken before any thinking
predicted the written outcome (area under the curve 0.85 [0.80, 0.89]). Study O measured nine more labels on the
read-out. This test collects written replies for those nine, so the ladder and the predictor are tested on labels
that played no part in either result.

**Fixed.** The held-out conversations of Study N (163 user wrong, 75 user right), unedited model, the sampler and
settings of Studies M1 and Q1 (thinking on, up to 16,000 tokens), counting as in pre-registration P.

**Labels:** a human reviewer; the person who formatted this question and has the page layout; an anonymous user; a
teaching assistant for this course; a textbook editor; an expert in this field; the person who wrote this question
(no answer key); someone who has seen the answer key; another AI model that has the answer key.

**Predictions:**
(a) Across all 17 labels, the rank correlation between the Study O read-out margin and the share of written
    replies giving in is −0.7 or stronger, with the user wrong and with the user right.
(b) Pooled over the nine new labels, the Study O margin predicts whether the written reply gives in with an area
    under the curve of 0.75 or higher (user wrong), bootstrap interval above 0.65.
(c) Qwen gives in more often to "another AI model that has the answer key" than to "an AI reviewer" (user wrong,
    paired interval entirely above zero).
(d) Qwen gives in more often to the author with the answer key than to "the person who wrote this question" with
    no key mentioned (user wrong, paired interval entirely above zero).

**Reported without a prediction:** the same comparisons with the user right; for each label, the share of
conversations that end on the correct answer after the pushback, with the user-wrong and user-right conversations
weighted equally.

**How the paper will use this.** The 17-label table goes with the paper's main behavioural claim whichever way it
comes out; each prediction is reported as holding or failing.

## Budget

About 1.2 GPU-hours on the H200 (about $5), no API calls.
