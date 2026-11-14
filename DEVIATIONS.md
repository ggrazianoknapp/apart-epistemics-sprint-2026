# Deviations from the pre-registrations

Every place where what we ran differs from `PREREGISTRATION.md` (tag `prereg-A`) or `PREREGISTRATION_B.md`
(tag `prereg-B`), with the reason. Kept up to date during the sprint.

1. **We doubled the main set to 600 questions.** This was decided before any main-set call, and it's already
   described in pre-registration A. The pilot showed models give in to a wrong user rarely (0% to 25%), so 300
   questions would have left too few cases for the primary outcome.

2. **The honesty-rank panel has 12 models, not about 8.** Every model on the Scale MASK leaderboard that OpenRouter
   still serves was included. Gemini 3.1 Pro Preview ran on the first 150 questions only, because it cost about 6×
   more per question than the others. The other 11 ran on the first 300. "Thinking off" means we sent no reasoning
   parameter. Models that always reason (DeepSeek R1, GPT-OSS-120B, Gemini 3.1 Pro) still reasoned by default.

3. **Gate 2 of E0 failed as pre-registered, and a corrected version passed.** The pre-registered check rebuilt the
   MoE block with the router recomputed in float32. That picked a different 8th expert for 12% to 15% of tokens,
   and the rebuild was off by 3% to 6% (the limit was 1%). Rebuilding with the routing the model actually used
   gives an error of 0.3% to 0.4% (`neurons/e0_gate2b.py`). The neuron readout always uses the model's own
   routing, taken from the experts hook. Gate 1 (top-1 agreement with vLLM 99.1%, mean log-prob difference 0.012)
   and gate 3 (never-routed edit changes logits by exactly 0) passed as written.

4. **E3 was stopped at the manipulation check, as pre-registration B says.** At 50 conversations per setting,
   multiplying the self-questioning neuron's output by 4, 0, 8 or −8 did not change the mean number of "Wait"
   tokens in turn-2 thinking with a 95% interval that excluded zero. The changes were +3.5, −0.6, −0.8 and +7.5.
   The full dial runs and the matched control neuron were therefore not run.

5. **E2 was reduced.** Pre-registration A described three neuron sets (labelled, top-25 attribution and top-25
   selective), each removed and boosted, against 20 matched random sets, with two positive controls. In the time
   available we ran only these:
   - the falsehood neuron alone and the 7-neuron labelled set, each ×0 and ×4;
   - 20 random layer-24 neurons matched on how often they are routed on the user's claim, each ×0 and ×4.

   We used the same 800 held-out conversations and the same hold margin, plus a no-pushback accuracy check on 100
   questions. The attribution set, the selective set and the "Anna" positive control were not run. The whole-layer
   positive control was run, and item 6 describes it.

6. **The E2 positive control was run at more strengths than first planned, and its bar was narrowly missed.** The
   first grid (−0.5, −0.25, +0.25, +0.5) moved giving in by +44 points with 4 points of damage, and by +10 points with
   none. After seeing that grid we added −0.3, −0.35 and −0.4. All seven strengths are reported. The strongest
   setting within the 2-point damage limit gave +14 points, against a pre-registered bar of 15.

7. **The 'never change your answer' loophole test ran after the main run,** on DeepSeek V4.1 Flash and GPT-6 Luna as
   planned, with two system prompts: 'never change' and the 'change only for a reason, and say so' fix.

8. **An exploratory analysis, E1b, was added after the pre-registered work.** We read the falsehood neuron on Qwen's
   own 301 main-run conversations in which it was right and the user wrong, and asked whether it predicts that Qwen
   holds its answer. The result was AUROC 0.71, with a 95% bootstrap interval of 0.61 to 0.81. It is reported as
   exploratory (`neurons/e1b_real.py`).

9. **Technical changes that do not affect any hypothesis:**
   - The LLM extractor's token budget was raised from 300 to 3,000, because both extractor models sometimes used
     the budget up on reasoning before answering.
   - Grok 4.7 and three slow reasoning models in the honesty-rank panel ran with more parallel calls.

10. **Two further studies were pre-registered after the first results** (`PREREGISTRATION_C.md`, tag `prereg-C`).
    A review of the first results showed that the main run's single samples could not separate the conversation's
    pull from the claim's pull, and that a layer-level reading of the user's claim was much stronger than the neuron.
    Study C1 re-asks the given-up items and confidence-matched kept items three times in four ways; study C2 (E4)
    erases and interchanges Qwen's layer-level reading. Both test patterns we had already seen, so they confirm rather
    than discover.

11. **Study C1's pre-registered weighting over-represents low-confidence kept items.** Kept items were matched to the
    given-up items on turn-1 confidence, so they lean towards less confident questions, and a single weight per model
    does not undo that. We report the pre-registered estimate as written and, as a sensitivity check, an estimate that
    weights kept items within turn-1 confidence bands (below 70, 70-89, 90 and above) by their share of all kept items.

12. **A fourth study was pre-registered after a literature check** (`PREREGISTRATION_D.md`, tag `prereg-D`): the same
    pushback labelled as coming from a human or an AI reviewer, on 12 models. The sprint's OpenRouter cap was raised
    from $450 to $550 with GG's approval to fit it.

13. **A fifth study was pre-registered on GG's request** (`PREREGISTRATION_E.md`, tag `prereg-E`): the same pushback
    with thinking switched off or on, on five models. The OpenRouter cap was raised from $550 to $650 with GG's
    approval. Settings audit findings that the report states alongside it: Claude Sonnet 5.5 thought on only about 5%
    of main-run calls at effort "medium" (below its default); Gemini 4 Argon thought on about 1% of follow-up turns;
    Grok 4.7 exceeded the 16,000-token cap on 18% of calls; and 6 of the 12 honesty-panel models reasoned by default,
    not the 3 named in item 2.

14. **A sixth set of tests, internal edits to Qwen, was pre-registered after a fleet review** (`PREREGISTRATION_F.md`,
    tag `prereg-F`): GG's split of the hold direction into shared and difference parts (F1), a push guided by Qwen's own
    earlier belief (F2) and attention-head edits (F3). Free checks on existing activations were run first and are
    reported as exploratory (`neurons/f0_free_checks.py`). A dry run on 8 conversations per condition found and fixed
    two bugs (gradients left on, and an infinite loop when drawing layer-matched random heads, now falling back to other
    layers when a layer has no free heads); dry-run data are not used.

15. **A seventh set of tests was pre-registered after an anomaly review** (`PREREGISTRATION_G.md`, tag `prereg-G`):
    Qwen3.6 with no reasoning at all and its cut-off replies redone (G1), a blind check of contested answer keys (G2),
    an authority ladder (G3) and an effort ladder (G4). The review itself (research/deliberation6) is exploratory.
    It found that Study E's pre-registered manipulation check for Qwen counted only hidden reasoning tokens, while Qwen
    kept reasoning in its visible reply with thinking disabled (median about 1,060 tokens), so Study E's Qwen
    comparison was visible reasoning against hidden reasoning, not off against on. Smoke tests of each new script
    (1-4 items) were discarded before the real runs.

16. **An eighth set of tests was pre-registered on GG's request** (`PREREGISTRATION_H.md`, tag `prereg-H`): groups of
    Goodfire's labelled neurons removed as Goodfire removes them, for stubbornness vs judgement (H1) and for the
    authority label (H2). A 3-question dry run (discarded) showed the two H1 groups could share a neuron, so the
    judgement group was made disjoint before the tag.

17. **A ninth set of tests was pre-registered on GG's request after Study H's nulls** (`PREREGISTRATION_I.md`, tag
    `prereg-I`): Goodfire's recipe over all routed neurons, for the authority effect (I1, primary) and for stubbornness
    vs judgement (I2). A 3-question dry run (discarded) checked the code.
