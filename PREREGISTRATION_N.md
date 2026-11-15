# Pre-registration N: a direction for the authority claim inside Qwen

Written on Sunday 15 November 2026 after Study M, before any held-out data for these tests existed. The git tag
`prereg-N` marks this file. Scripts: `neurons/n_direction.py`, `neurons/n_written.py`; analysis
`harness/analyze_n.py`.

**Why.** Studies J to M showed that the 25 neurons support Qwen holding its own answer in general. They are not a
representation of the authority claim, and removing them leaves Qwen's response to the author label almost
unchanged. This study looks for the claim itself: one direction in Qwen's internal state that separates a message
labelled as coming from the question's author from the same message labelled as coming from an AI reviewer, and
tests whether swapping only that direction between the two versions changes what Qwen does.

**Data and splits.** Qwen's own main-run conversations, as in Study L5 (301 where Qwen was right and the user
wrong, 156 where the user was right). Each conversation is assigned by a hash of its question id to one of three
parts: 40% to compute the direction, 20% to choose the layer, 40% held out for every reported number.

**The direction.** For each conversation in the first part and each of the 40 layers, the layer's output with the
author label minus its output with the AI label, averaged over the final positions that are word for word the same
in every conversation (the format reminder at the end of the user's message and the start of Qwen's reply), then
averaged over conversations. This gives one candidate direction (unit length) per layer.

**The intervention: swapping one direction.** A conversation is run with one label (the source) and, at every
layer and each of those final positions, the component of the layer's output along the direction is recorded. The
same conversation is then run with the other label (the target), and at every layer and each of those positions
the component along the direction is set to the recorded value. Everything else in the model's state is left as it
is. The state has 2,048 dimensions, so this changes one of them.

- *Adding the claim:* target is the AI-labelled message, source is the author-labelled one.
- *Removing the claim:* target is the author-labelled message, source is the AI-labelled one.

**An earlier design, dropped before this file was tagged.** A first version added the direction at one layer and
removed it by editing the weights. Trial runs on 4 and 12 conversations (using the layer-choosing part for the
layer sweep) showed that the weight edit moved the hold margin under the AI label by more than the whole
author-minus-AI difference, so it could not isolate the claim. No held-out number was used to make this change.

**Choosing the layer (second part, user-wrong conversations).** Of the 40 candidate directions, the one with the
largest average of the share reproduced when adding and the share removed when removing.

**Measure.** Hold margin as in Study L5 (after "My final answer is (", thinking closed). Share reproduced =
(margin with the AI label after the swap − margin with the AI label) / (margin with the author label − margin with
the AI label). Share removed = (margin with the author label after the swap − margin with the author label) /
(margin with the AI label − margin with the author label). 95% bootstrap intervals over held-out conversations.

## Predictions (held-out user-wrong conversations)

- **N1, adding.** Swapping the direction into the AI-labelled message reproduces at least half of the
  author-minus-AI difference, and more than each of 20 random directions.
- **N2, removing.** Swapping the direction out of the author-labelled message removes at least half of the
  difference, and more than each of 20 random directions.
- **N3, other wording.** With the direction still taken from the original labels, N1 and N2 also hold between the
  reworded labels, and between the reworded labels with the reworded pushback (the wording of Study M2).
- **N4, written replies.** On the held-out user-wrong conversations, Qwen writes its reply to the author-labelled
  message with thinking on and the main run's sampling settings, three ways: unchanged, with the direction swapped
  out (source: the AI-labelled message), and with a random direction swapped the same way. The swap is applied to
  the prompt only; the reply is then generated freely. Prediction: Qwen gives in less often with the direction
  swapped out than unchanged, with the paired interval excluding zero, and less often than with the random
  direction. Replies are capped at 6,000 new tokens (the main run used 16,000) so that the run fits in the night;
  replies with no readable final line count as not having given in and are reported. N4 is run only if batched
  generation with the swap proves correct and fast enough to finish by 03:00 on Monday; if it is not run, the paper
  says so.

## Reported without a prediction

- Swapping the direction into a message with no label and into one from a human reviewer; swapping it out of a
  message from a professor.
- Two comparison directions built the same way and swapped the same way: human reviewer minus AI reviewer, and "the
  person who formatted this question and has the page layout" minus AI reviewer. They show whether the direction is
  about authority or about any named person.
- Where each label falls along the direction (mean component at the chosen layer for no label, AI, human,
  professor, author, formatter).
- How similar the direction is to the one computed from the reworded labels (cosine, per layer).
- The same swaps on the held-out user-right conversations.

## How the paper will use these results (decided now)

- If N1, N2 and N4 hold, the direction is reported in the main text as the paper's account of how the authority
  claim acts inside Qwen, next to the 25 neurons.
- If N1 and N2 hold and N4 fails or is not run, it is reported in the main text as a result on the internal
  read-out only, with that limit stated next to it.
- If N1 or N2 fails, the study goes to the appendix as a negative result.
- N3 decides whether the paper says the direction holds across wording.

## Budget

About 4 GPU-hours on the H200 (about $16), no API calls.
