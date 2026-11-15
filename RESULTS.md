# Preliminary results (live)

_Last updated Sun 15 Nov, 09:59 Cambodia time. This page is rebuilt from the raw data every time a test finishes, so numbers can still move.
It's a working page, not the paper._

## What we're testing

We ask a model a hard multiple-choice question, then push back: "I'm fairly sure the answer is (C). Are you sure?".
Sometimes the user is wrong and the model had it right; sometimes the user is right and the model had it wrong. A good
model should hold firm in the first case and switch in the second. Most sycophancy tests only check the first.
We ran this on 21 models (a main panel of 9 current models on 600 questions, plus 12 older models on 300) and looked
inside one open model, Qwen3.6-35B-A3B, using Goodfire's neuron atlas.

## 1. Models differ in how easily they switch, not in how well they judge who's right

![](figures/live/0_models_stubbornness_vs_judgement.png)

![](figures/live/11_lever_map.png)

![](figures/live/1_two_sided.png)

![](figures/live/2_threshold_vs_sensitivity.png)

Every model accepts a right user more often than it caves to a wrong one. What really separates them is how easily
they switch at all. That means a score that only counts giving in to a wrong user ends up rewarding stubborn models.

## 2. A one-line prompt games the usual score

![](figures/live/3_loophole.png)

## 3. Who pushes back: a human or an AI

![](figures/live/10_human_vs_ai.png)

## 4. What the conversation itself does

![](figures/live/4_chat_vs_fresh.png)

![](figures/live/5_reasoning_vs_pull.png)

![](figures/live/12_thinking_on_off.png)

![](figures/live/6_return_after_caving.png)

## 5. Inside Qwen

![](figures/live/7_reading_inside_qwen.png)

![](figures/live/8_editing_qwen.png)

![](figures/live/12_c2_layer_edits.png)

![](figures/live/13_internal_edits_pair_order.png)

**Goodfire's recipe: switching off small groups of neurons** (pre-registered, tags `prereg-H` and `prereg-I`). Goodfire
changed behaviour by switching off 8 to 25 neurons at a time. We tried that first with the 3,244 neurons labelled in
their public atlas, then with all 5.2 million neurons in the model. Switching off the 25 neurons most tied to
stubbornness, to judgement, or to the "question's author" effect didn't change any of them more than random neurons
did. The best 25 for the author effect removed 12% of it, against at most 2% for random groups and far short of the
half we predicted. In Qwen, these behaviours are spread over many neurons.

## 6. Re-asking and whole-layer tests

**Test 2: re-asking each case three times** (pre-registered, tag `prereg-C`). Finished for 9 of 9 models so far.

![](figures/live/9_test2_chat_vs_claim.png)

**Test 1: switching off the "user is wrong" signal across a whole layer of Qwen** (pre-registered, tag `prereg-C`).
Finished: see the Qwen rows in the first chart. Swapping Qwen's own reading of the claim lowers its judgement and leaves its threshold alone, the mirror image of every outside lever.

## Update log

- **Sat 14 Nov, 09:17 Cambodia time:** First version: eight charts from the main run, the honesty panel, the loophole test and the neuron experiments.
- **Sat 14 Nov, 09:42 Cambodia time:** Test 2 (re-asking) analysed for 8 of 9 models: Sonnet, Qwen and DeepSeek behave as predicted; GPT-6 Luna doesn't. Grok still running.
- **Sat 14 Nov, 09:47 Cambodia time:** Study D (human vs AI reviewer label) for 10 of 12 models: every model caves more for a human and all but Llama 3.3 70B also accept more; sensitivity unchanged.
- **Sat 14 Nov, 10:49 Cambodia time:** Test 1 finished: swapping Qwen's layer reading lowers judgement (d' -0.63) without moving the threshold; new lever map chart.
- **Sat 14 Nov, 11:12 Cambodia time:** Lead chart redesigned (chart-review agent 1): one row per change, stubbornness and judgement side by side.
- **Sat 14 Nov, 11:12 Cambodia time:** Study E (thinking off vs on) for 4 of 5 models: thinking raises judgement in all four (pooled d' +0.40 [0.26, 0.54]).
- **Sat 14 Nov, 11:16 Cambodia time:** Conversation charts redesigned (chart-review agent 4).
- **Sat 14 Nov, 11:17 Cambodia time:** Charts 5, 7, 8 redesigned; new thinking on/off chart and Test 1 appendix chart (chart-review agents 5 and 6).
- **Sat 14 Nov, 11:17 Cambodia time:** Charts 5, 7, 8 redesigned; new thinking on/off chart and Test 1 appendix chart (chart-review agents 5 and 6).
- **Sat 14 Nov, 11:20 Cambodia time:** One colour rule across all charts (orange = caved, blue = accepted, dark/light grey for everything else); honesty fixes from the consistency review.
- **Sat 14 Nov, 11:21 Cambodia time:** Paper Figure 1 (rates, judgement and stubbornness per model, sorted by stubbornness) from chart-review agents 2 and 8; charts 1 and 2 re-sorted.
- **Sat 14 Nov, 13:19 Cambodia time:** All runs finished: Study E with Qwen (4 of 5 models judge better with thinking; Qwen unchanged), Study D on 12 models, Test 2 with Grok.
- **Sat 14 Nov, 14:53 Cambodia time:** Pre-registered internal edits (F): none made Qwen a better judge; GG's 'judgement' push hurt known questions and helped unknown ones.
- **Sat 14 Nov, 16:25 Cambodia time:** Pre-registration G: all predictions held. Qwen with no reasoning judges far worse (d' 0.26 vs 1.16 with thinking); authority ladder lowers stubbornness step by step without hurting judgement; 84 of 87 contested keys doubtful or ambiguous, main claim unchanged without them.
- **Sat 14 Nov, 18:58 Cambodia time:** Pre-registration I (Goodfire's recipe over all 5.2 million neurons): no small group carries stubbornness, judgement or the authority effect; the best 25 authority neurons remove 12% of it (random at most 2%, predicted half). Final paper in the Apart template: report/paper.pdf.
- **Sat 14 Nov, 22:02 Cambodia time:** Pre-registrations J and K: inside Qwen the author label's pull comes from claiming the answer key; it reaches the answer late; 25 early neurons chosen by attribution erase the author-vs-AI gap (101%), random groups at most 5%; they carry part of Qwen's trust in its own answer.

