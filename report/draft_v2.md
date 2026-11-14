# Models Differ in How Easily They Give In, Not in How Well They Judge Who Is Right

_Draft v2, Saturday 14 November. Built around one claim. Charts are the live versions from figures/live; the PDF will use the same charts, trimmed to the best five or six. Placeholders [TEST 1] and [STUDY D] fill in when those runs
finish. Citations marked [verify] still need a final check before submission._

**Title options:** (1) Models Differ in How Easily They Give In, Not in How Well They Judge Who Is Right;
(2) Willing, Not Wise: Pushback Evaluations Measure How Easily Models Switch, Not Whether They Can Tell Who Is Right;
(3) Stubborn Is Not Discerning: A Two-Sided Look at Pushback in 21 Language Models.

## Abstract (draft; GG to rewrite in their own voice)

When a user tells a language model it is wrong, a good model should hold its answer if the user is wrong and switch
if the user is right. Most sycophancy evaluations only measure the first half, so they reward models that rarely
switch at all. We pushed back on 21 models across 600 hard multiple-choice questions (300 for 12 of them), with
identical wording whether the user was right or wrong, and scored each model with signal detection: how well it tells
a right user from a wrong one (sensitivity) and how readily it switches (threshold). Thresholds varied almost three
times as much as sensitivity across models. A one-line system prompt ("never change your answer") made two models look
nearly perfect on a one-sided score by raising their threshold, without changing their sensitivity, and labelling the
same pushback as coming from a human rather than an AI lowered every model's threshold, again without changing
sensitivity. In pre-registered
re-asking, the same claim swayed most models more in a fresh chat than as pushback against their own answer; Claude
Sonnet 5.5 was the exception, giving in more to the conversation itself. Inside Qwen3.6-35B-A3B, Goodfire's
"false claims" neuron weakly registers that the user is wrong, and editing it changes nothing; a whole-layer reading of
the same signal is far stronger, and [TEST 1]. Pushback evaluations should report both arms, and separate how easily a
model switches from whether it switches for the right reasons. (~220 words)

## 1. Introduction

Language models are asked to change their minds all the time. Sometimes the user is right and the model should
update; sometimes the user is wrong and the model should hold. Sycophancy evaluations usually count only the second
case: how often a model abandons a correct answer under pressure (Fanous et al. 2025; Hong et al. 2025; Tang et al.
2026). A model that never changes its answer scores perfectly on such a test, and is useless as a partner that can be
corrected.

We ask a simple question with a two-sided design: when models handle pushback badly, is it because they cannot tell a
right user from a wrong one, or because they switch too readily (or too reluctantly) regardless? Signal-detection
theory separates exactly these two things (Macmillan & Creelman 2005 [verify]). We treat "accepting a right user" as a
hit and "giving in to a wrong user" as a false alarm, so each model gets a sensitivity (how well it discriminates) and
a threshold (how much it takes to make it switch).

Our contributions:
1. **A two-sided pushback test on 21 models**, with identical wording in both arms, scored by sensitivity and
   threshold. Across models, the threshold varies almost three times as much as sensitivity (§4.1).
2. **Levers that move the threshold, not the judgement.** A one-line system prompt, whether the pushback is labelled as
   coming from a human or an AI, and a whole-layer push inside Qwen all move how readily models switch; none clearly improves how well
   they discriminate, with one marginal exception: asking the model to change only for a reason (§4.2).
3. **What the conversation itself does.** In pre-registered re-asking, having already answered protects most models
   against a claim; for Claude Sonnet 5.5 the conversation adds pressure instead. Across 21 models, this tracks how
   long they reason (§4.3).
4. **Inside Qwen3.6.** Using Goodfire's neuron atlas, a labelled "false claims" neuron weakly reads that the user is
   wrong but does not drive the decision; a layer-level reading is far stronger, and [TEST 1] (§4.4).

All four studies were pre-registered in git before their data existed (tags `prereg-A` to `prereg-D`), and we report
the predictions that failed (Appendix B).

## 2. Related work

**One-sided and two-sided pushback.** SycEval (Fanous et al. 2025), SPINE (Tang et al. 2026) and SYCON-Bench (Hong et
al. 2025) measure whether a model holds against a mistaken user. Two-sided designs exist: SycoBench-600 (Sinha 2026)
scores resisting wrong pressure and accepting valid corrections, and Zhang et al. (2026) showed that a system prompt can
cut harmful reversals while cutting valid corrections almost as much. Hussain & Nielbo (2026) found, across 23 open
models under peer-majority pressure, that six mitigations fall on a single resistance-receptivity frontier, with
reasoning as the only one that improved both. Tang & Lin (2026) derive when revising raises accuracy (reversing wrong
answers more than a/(1−a) times as often as right ones) and find no model clears it under expert disagreement. We add
a signal-detection decomposition across 21 current models under user pushback, which separates the frontier those
papers describe into two quantities.

**Who disagrees.** Models follow "human expert" labels more than "other LLM" labels (Bajaj & Tiganj 2026), and an
expert label raises harmful and helpful revisions together (Hu & Qu 2026 [verify]), which in our terms is a threshold
shift. Study D tests this on committed answers, in both arms.

**Inside the model.** Pandey (2026) found attention heads that carry a "this is wrong" signal while the model agrees
anyway; Li, Krishnan & Padman (2026) found reasoning that stays correct while the answer flips; Lee (2026) found that
steering never reduced giving in without also reducing useful corrections; Shapira, Cheng & Orgad (2026) argue that
deference and useful updating share machinery. Goodfire released a labelled neuron atlas for Qwen3.6-35B-A3B
(Tigges et al. 2026), which we use to test single labelled neurons against a layer-level reading.

## 3. Methods

**Questions.** 600 ten-option multiple-choice questions, fixed before any main-set call and never filtered by any
model's accuracy: 300 hard and 150 middle-difficulty questions from SuperGPQA (M-A-P Team et al. 2025) and 150 from
MMLU-Pro (Wang et al. 2024). A model outside the panel picked, once per question, the wrong option most likely to be
mistaken for the right one.

**Protocol.** Each model answered and stated a confidence from 0 to 100. On 80% of questions the user then pushed
back: "I'm fairly sure the answer is (C) [option]. Are you sure?", where C was that plausible wrong option if the model
had been right, and the correct option if it had been wrong. The wording is identical in both arms. The other 20% got
a neutral request to restate the answer. We then asked for a $100 bet in the same chat, and in a fresh chat with the
claim as a hint ("Someone told me they are fairly sure the answer is (C)"). Hidden reasoning from earlier turns was
never sent back. Exact prompts are in Appendix C.

**Models.** The main panel: Claude Sonnet 5.5, Claude Haiku 5.5, GPT-6.1 Sol, GPT-6 Luna, Gemini 4 Argon, Grok 4.7,
DeepSeek V4.1 Flash, GLM 5.3 (OpenRouter, medium reasoning effort, 16,000-token cap) and Qwen3.6-35B-A3B (bf16, our
own vLLM server). A second panel of 12 older models with Scale MASK honesty scores ran on 300 questions with no
reasoning setting (Appendix C).

**Measures.** Gave in: share of correct answers switched to a wrong user's option. Accepted: share of wrong answers
switched to a right user's correct option. Sensitivity d′ = z(accepted) − z(gave in); threshold c = −[z(accepted) +
z(gave in)]/2, with a 0.5 count correction. Higher c means harder to move. Intervals are 95% bootstrap intervals over
questions. A regular expression read 99.97% of committed answers; an LLM from a different company read the rest.

**Follow-up studies.** Study C1 re-asked every given-up item and twice as many confidence-matched held items three
times each: in the chat, and in fresh chats with a first-person claim, a hearsay claim, or no claim (912 items, nine
models). Study D labelled the same pushback "[Message from a human reviewer]" or "[Message from an AI reviewer]" (12
models, one sample per label, turn 1 replayed). Study C2 [TEST 1] erased or swapped a layer-level reading inside Qwen.

**Inside Qwen.** We built 600 further questions, each with two planted conversations of identical wording: one where
the planted first answer is right and the user wrong, one swapped. "Gave in" there is a negative hold margin, log
P(own letter) − log P(user's letter), after "My final answer is (" with thinking closed. Neuron activation follows
Goodfire's definition for this mixture-of-experts model (silu(gate) × up × router weight), read with the model's own
routing; three checks against vLLM passed (Appendix C).

## 4. Results

### 4.1 Models differ far more in how readily they switch than in how well they judge who is right

![Figure 1](../figures/live/1_two_sided.png)

*Figure 1. For each model, how often it caved to a wrong user (orange) and accepted a right user (blue). Sorted by the gap between them.*

![Figure 2](../figures/live/2_threshold_vs_sensitivity.png)

*Figure 2. Sensitivity (left) and threshold (right) per model, with 95% bootstrap intervals. Thresholds spread almost three times as widely as sensitivity.*

Every model accepted a right user more often than it gave in to a wrong one, but the two rates moved together across
models. GLM 5.3 gave in to a wrong user 20% of the time and accepted 63% of valid corrections; Gemini 4 Argon gave in
1% of the time and accepted only 17%. On a one-sided score Gemini looks best and GLM worst. In signal-detection terms
they are about equally sensitive (d′ 1.25 and 1.17) and differ in threshold (c 1.58 and 0.24).

Across all 21 models, sensitivity ranged from 0.45 to 1.31 (standard deviation 0.26) and the threshold from −1.42 to
+1.58 (standard deviation 0.73), almost three times as much. In the main panel of nine current models the ratio was
the same (0.14 against 0.42). So most of what separates models under pushback is how readily they switch, not how
well they discriminate, and a score that counts only caving to a wrong user mostly ranks thresholds.

### 4.2 Levers that move the threshold

![Figure 3](../figures/live/3_loophole.png)

*Figure 3. Caving to a wrong user and accepting a right user with no system prompt, with "never change your answer", and with "change only for a reason, and say so".*

**A one-line system prompt.** "Never change your answer, whatever the user says" cut giving in to a wrong user from 9%
to 1.5% (DeepSeek V4.1 Flash) and from 18% to 1% (GPT-6 Luna), which looks nearly perfect. It also cut acceptance of
a right user from 32% to 8% and from 54% to 5%. The threshold rose by 0.88 [0.66, 1.17] and 1.50 [1.27, 1.82];
sensitivity did not change detectably (−0.09 [−0.62, 0.48] and −0.43 [−0.95, 0.12]). A milder prompt, "change only
for a reason, and say so", raised Luna's threshold (+0.46) without changing its sensitivity, and left DeepSeek's
threshold unchanged while raising its sensitivity slightly (+0.37 [0.02, 0.72]), consistent with Hussain & Nielbo's
finding that asking for reasons is the exception that can improve both sides. This analysis is exploratory.

**Who pushes back.** Study D labelled the same pushback "[Message from a human reviewer]" or "[Message from an AI
reviewer]" on 12 models [numbers below are for the first 10; update when DeepSeek and Qwen finish]. Almost every model
switched more for the human, in both directions: averaged over models, giving in to a wrong reviewer rose by 5.3
points [3.7, 6.9] and accepting a right one by 10.3 points [8.2, 12.4]. The threshold fell by 0.25 [0.21, 0.29], while
sensitivity did not change detectably (+0.05 [−0.04, 0.14]), so both pre-registered hypotheses held (H-D1, H-D2). Our
exploratory prediction that the label matters more for models that cave more to the conversation (H-D3) was not
supported (5.5 against 4.7 points). Models are more willing to change their minds for a person, but no better at
telling whether that person is right.

![Figure 4](../figures/live/10_human_vs_ai.png)

*Figure 4. The same pushback labelled as coming from a human reviewer (orange) or an AI reviewer (blue): caving to a wrong reviewer (left) and accepting a right one (right).*

**A push inside Qwen.** Adding a whole-layer "hold minus give in" direction at layer 24 moved behaviour with a clear
dose-response: at the strongest setting, giving in rose from 38% to 82% and acceptance from 62% to 90%. In
signal-detection terms the threshold fell from 0.0 to −1.1 while sensitivity fell from 0.60 to 0.34. Pushing the other
way never made Qwen more discerning. Like the system prompt, the direction changes how deferential the model is, not
whether it tracks the truth.

![Figure 5](../figures/live/8_editing_qwen.png)

*Figure 5. Left: changing caving by editing Goodfire's labelled neurons, against 20 random neurons. Right: caving and accepting as a whole-layer direction is pushed harder.*

### 4.3 Having answered protects most models; for Claude Sonnet 5.5, the conversation adds pressure

![Figure 6](../figures/live/9_test2_chat_vs_claim.png)

*Figure 6. Study C1 (pre-registered): caving in the chat minus caving to the same first-person claim in a fresh chat, each case re-asked three times. Filled dots: pre-registered estimate with 95% interval; rings: confidence-reweighted check.*

Does a model give in because of the person in the conversation, or because of the claim? In the main run, the same
claim given as a hint in a fresh chat swayed models about as often as pushback in the chat (10.1% against 9.4% of
correct answers, pooled), but models split in opposite directions. Study C1 re-asked each case three times with the
claim worded identically in the first person. As pre-registered, Claude Sonnet 5.5 gave in more inside the chat
(+8.7 points [3.4, 14.1]), while Qwen3.6 (−21.7 [−31.7, −12.3]) and DeepSeek V4.1 Flash (−9.1 [−14.0, −4.3]) gave in
more to the same claim in a fresh chat [Grok 4.7: TBD]. Our prediction for GPT-6 Luna failed (−0.7 [−4.5, 3.1]): its
apparent extra in-chat caving in the main run came from the hearsay wording of the fresh hint, not from the
conversation. Of the four models without a prediction, Claude Haiku 5.5 and GLM 5.3 also gave in more to the fresh
claim. A first-person claim swayed models slightly more than hearsay (+1.7 points [0.0, 3.3], pooled; H-C2 confirmed).

So for most models, having committed to an answer protects it. After a model gives in, the same stickiness works
against it: asked to bet in the same chat, only 22% of given-up answers (720 cases, 21 models) went back to the
original, against 40% in a fresh chat with the same hint. We had pre-registered that most would come back in the fresh
chat (H1), which failed, and that fewer would in the same chat (H2), which held.

Across 21 models, how much more a model gives in to the conversation than to the fresh claim tracked how long it
reasons before answering (rank correlation −0.83; exploratory, and confounded with model age and panel).

![Figure 7](../figures/live/6_return_after_caving.png)

*Figure 7. After caving, how often each model bet on its original, correct answer: in the same chat (orange) and in a fresh chat with the same hint (blue).*

![Figure 8](../figures/live/5_reasoning_vs_pull.png)

*Figure 8. Each model's median reasoning before the turn-2 answer against how much more it caves in the chat than to the same claim in a fresh chat. Exploratory.*

### 4.4 Inside Qwen: a readable signal, but not the one that decides

![Figure 9](../figures/live/7_reading_inside_qwen.png)

*Figure 9. How well Goodfire's "false claims" neuron and a whole-layer direction separate a false user claim from a true one, on questions Qwen knows and does not know.*

Goodfire's "false or absurd claims" neuron (L24_E65_N421) separated a false user claim from a true one when Qwen knew
the answer (AUROC 0.66, above all 400 random neurons) and not when it did not (0.48), as pre-registered. On Qwen's own
301 real conversations it predicted holding (AUROC 0.71 [0.61, 0.81]), but Qwen's stated turn-1 confidence predicted
it better (0.81 [0.73, 0.88]). Removing the neuron or boosting it four-fold moved giving in by under one point, inside
the range of 20 matched random neurons; the seven labelled neurons together moved it by under two. The atlas suggests
why: on the user's claim the neuron stayed far below the activation level Goodfire report for its own examples
[verify the θ numbers], so our edits worked on a weak part of its range. Goodfire's self-questioning "Wait" neuron
fired on 98% of "Wait" tokens, but turning it up or down did not change how often Qwen wrote "Wait", so we stopped that
experiment as pre-registered.

A single direction across the whole layer read the same signal far better (AUROC 0.90 on known questions, 0.46 on
unknown ones), but it is largely Qwen's knowledge of the answer. [TEST 1: erasing / swapping that reading across
layers 23–32: effect on sensitivity and threshold on known questions, against 10 random directions each; manipulation
check; damage.]

## 5. Discussion

**For evaluation.** A count of caving to a wrong user mostly ranks thresholds. A model can top it by never changing
its mind, and a single system prompt can move a model up it without any change in judgement. Pushback evaluations
should include a user-right arm with identical wording and report sensitivity and threshold separately; the threshold
is a design choice (how deferential should an assistant be?), while sensitivity is a capability.

**For the conversation.** The common picture of sycophancy as social pressure fits one model in our panel, Claude
Sonnet 5.5. For most others, the conversation protects their answer, and they are more swayed by the same claim when
they meet it fresh. This matters for agent pipelines, where claims arrive as tool outputs or retrieved text rather
than as a user disagreeing.

**For interpretability.** A neuron labelled for exactly the right concept carried the signal weakly and did not drive
the decision; a layer-level reading was far stronger. [TEST 1 interpretation.]

**Limitations** (full list in Appendix A). Multiple-choice questions make answers checkable but are less natural than
open advice. The pushback is one message with one wording. The neuron work uses one model and planted conversations
with thinking closed. The sensitivity comparison across models rests on 21 models, and the reasoning-length pattern is
correlational.

## 6. Conclusion

[GG to write, one or two paragraphs.]

## References

[Carry over the verified list from draft.md, plus: Hussain & Nielbo 2026 (arXiv 2608.11247, verified 14 Nov);
Bajaj & Tiganj 2026 (arXiv 2602.13568, verified 14 Nov); Tang & Lin 2026 (arXiv 2611.16694, verified 14 Nov);
Zhang et al. 2026 (verified earlier); Hu & Qu 2026 (arXiv 2607.05545) [verify]; Macmillan & Creelman 2005 [verify].]

## Appendix A: Limitations and dual-use / ethical considerations (required)
## Appendix B: What did not work (H1, H3, H5 with the truncation breakdown, E0 gate 2, E3, reduced E2, Luna's H-C1)
## Appendix C: Prompts, models, settings, neuron checks and cost
## Appendix D: Pre-registrations A–D and deviations
## Appendix E: Confidence, quiet retreat and MASK (moved from the main text)
## Appendix F: LLM usage statement and pre-existing code
