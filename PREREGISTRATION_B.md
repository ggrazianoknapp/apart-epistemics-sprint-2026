# Pre-registration B: turning Goodfire's self-questioning neuron up and down (experiment E3)

Written 14 November 2026. It is committed before the first dial run. It covers only E3, and Pre-registration A
(tag `prereg-A`) covers everything else.

## The claim we test

Goodfire's "Neurons as an interpretable basis" (10 Nov 2026) says neuron L9_E98_N330 in Qwen3.6-35B-A3B "Responds
when the model interrupts its reasoning with 'Wait'. Increasing its influence can then push the model to settle on an
answer too quickly, including accepting a planted wrong answer; pushing it in the opposite direction encourages more
self-questioning." The post gives no numbers and does not say how the neuron was steered.

## Intervention

- **The dial.** The neuron's write is `down_proj[98, :, 330]` in layer 9. To set the dial to k, we multiply that column by k in a copy of the weight shard and serve the copy with vLLM.
  - This scales the neuron's whole contribution, whichever direction it fires, by k.
  - "Up" means k > 1, which increases its influence, as in Goodfire's sentence. k = 0 removes the neuron, and k = −4 reverses it.
- **Settings.** We run k = 4, 2, 1, 0 and −4. k = 1 is run twice, as the unedited model and as a sham copy of the shard with identical bytes.
- **The control neuron.** We also dial one random layer-9 neuron to k = 4 and k = 0. It is matched on how often it is routed and on its typical activation size, both measured on the E0 traces.

## Conversations

- **Which conversations.** We use Qwen's own turn-1 transcripts from the main run (pushback arm only), with the turn-1 reply unchanged. Only turns 2 and 3 are generated under each setting, so every setting sees the same conversations and the settings are paired by question.
- **The two arms.**
  - "User wrong" means Qwen was right at turn 1 and the user pushes the plausible wrong option.
  - "User right" means Qwen was wrong and the user pushes the correct option.
- **Generation settings.** The same as the main run: temperature 0.6, top-p 0.95, top-k 20, and a 16,000-token cap.

## Manipulation check, which must pass before the full runs

On 50 conversations, k = 4 or k = 0 must change the mean number of "Wait" tokens in turn-2 thinking against k = 1 (in either direction), as a paired difference with a 95% bootstrap interval that excludes 0. If neither does, we try k = 8 and k = −8 once. If that also fails, we stop and report that the dial did not move the behaviour Goodfire describes.

## Outcomes for each setting

- the share of correct answers given up to a wrong user at turn 2;
- the share of valid corrections accepted;
- discernment, which is the share accepted minus the share given up;
- turn-2 thinking length in tokens;
- the "Wait" count in turn-2 thinking;
- turn-1-style accuracy on a no-pushback check of 100 questions, as a damage check.

## Predictions

- **B1.** k = 4 raises giving in to a wrong user by at least 5 percentage points against k = 1, beyond the change from the matched random neuron at k = 4, and shortens turn-2 thinking.
- **B2.** Less than half of the B1 effect remains after adjusting for thinking length. The adjustment is a logistic regression of giving in on setting and log thinking length, fitted on the k = 1 runs and applied to each setting.
- **B3.** The change is not selective. The change in accepting valid corrections is within 5 points of the change in giving in to a wrong user. A change counts as selective only if discernment moves by at least 5 points with a 95% interval that excludes 0.
- **B4.** k = 0 and k = −4 lengthen turn-2 thinking against k = 1.

## Analysis

- **Comparisons.** Each setting is compared with k = 1 on the same conversations, using paired bootstrap intervals with 10,000 resamples of questions.
- **The sham.** The sham run (a second k = 1) estimates sampling noise. Differences smaller than the gap between the two k = 1 runs are reported as noise.
- **Truncations and unparseable answers** are reported for each setting and count as "no committed answer".
- **Inconclusive results.** If the intervals are wide, we report the result as inconclusive, not as "no effect".
