# Pre-registration I: Goodfire's recipe on all neurons, not only the labelled ones

Written on Saturday 14 November 2026 before any data for these tests existed. The git tag `prereg-I` marks this file.
Study H found that groups of the 3,244 neurons labelled in the public atlas (about 0.06% of the model) do not carry
stubbornness, judgement or the authority effect. Goodfire's own recipe searches all neurons for those that drive a
behaviour and removes a small group. We do that here over every routed-expert neuron in all 40 layers of
Qwen3.6-35B-A3B (40 x 256 x 512), with the same planted conversations, labels, selection/held-out split and measures as
pre-registration H. Script: `neurons/i_all_neurons.py`; analysis `harness/analyze_i.py`.

**Scoring.** At the last token, each routed neuron's direct effect on the hold margin (activation x its output column
dotted with the letter difference, through the final norm), summed over selection conversations.

**Controls.** For each chosen neuron, a random neuron from the same layer among the 500 whose summed absolute effect is
closest to it, so controls are equally active neurons with no selection on the behaviour.

## I1 (primary): the authority effect

Authority neurons: those whose effect shifts most towards the user's letter between the AI and author labels. Remove the
top 10, 25 and 100. Controls: 4 random sets of 25 and 2 of 100. **Prediction:** removing the top 25 shrinks the
authority effect (author minus AI change in threshold c, held-out questions) by at least half, with an interval for the
reduction excluding zero, and by more than every random set of 25.

## I2 (secondary): stubbornness and judgement

As H1, over all neurons: the 25 with the highest stubbornness score and the 25 with the highest judgement score among
the rest; 3 matched random sets each. **Predictions:** removing the stubbornness neurons lowers c by more than every
matched random set with pair order changing by less than 3 points; removing the judgement neurons lowers pair order by
more than every matched random set.

## Damage

No-pushback accuracy on the 400 held-out questions may fall by at most 2 points for the named groups; the share of
conversations whose top letter is neither option may rise by at most 2 points.

## Budget

About 2 GPU-hours (about $7), no API calls.
