# Deviations from the pre-registrations

Every difference we know of between what we pre-registered (`PREREGISTRATION.md`, tag `prereg-A`, and
`PREREGISTRATION_B.md` to `PREREGISTRATION_L.md`, tags `prereg-B` to `prereg-L`) and what we ran or reported, including
analysis code written or changed after the data, with the reason. Items 18 onwards were found by a 16-agent audit of
the whole project on 14-15 November (research/deliberation8, not public) and added before submission.

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

18. **The extraction check was never done.** Pre-registration A promised to validate the answer extractor against at
    least 100 hand-labelled replies and report Cohen's kappa. 150 items were prepared (`labels/`), but no one labelled
    them in time, so no kappa is reported. The regular expression read 99.97% of the main panel's committed answers and
    99.71% of the older panel's.

19. **The 32-question answer-key rule of pre-registration A was replaced.** It asked for a hand review of questions
    where 7 or more of 9 models agreed against the key at 80% confidence, and results with and without them. No hand
    review was done; Study G2 later used a different rule (two checker models, Claude Opus 4.6 and Gemini 3.1 Pro
    Preview, on 87 contested questions). Without the 32 questions, pooled acceptance of a right user moves from 40.5% to
    50.1%.

20. **The headline measures changed after the main data.** Pre-registration A scored judgement as discernment
    (accepted minus gave in). Sensitivity d′ and threshold c were first used in pre-registration D and
    `harness/sdt.py` (14 Nov 09:47), after the main run; the decomposition in §4.1 is therefore exploratory. H4,
    stated on discernment, held (GLM 5.3 0.43 [0.34, 0.52] against Gemini 4 Argon 0.15 [0.08, 0.24]) and is now
    reported. The reason for d′: a difference of rates depends on where a model's threshold sits.

21. **E1's random baseline was smaller and less matched than planned:** 400 random layer-24 neurons (31 of them
    matched on routing, none on activation scale) instead of 1,000 matched on layer, routing and scale. The best of the
    31 routing-matched neurons reached AUROC 0.59.

22. **E2 ran a ×4 boost instead of ×2, and its neutral-text damage check was never measured.** Its positive controls
    did not pass (the whole-layer control reached +14 points against a bar of 15, and the "Anna" control was not run),
    so by pre-registration A the E2 null is not interpreted; the paper now says so.

23. **C2 was reported in d′ and c instead of the pre-registered discernment.** Prediction 1 (swap brings known
    discernment below 0.40) failed: 0.84 to 0.71. The paper now reports this.

24. **Analysis code not in its tag or changed after results.** `neurons/e4_layer_reading.py` (named in prereg-C) was
    first committed with prereg-D, 26 minutes later, and patched four minutes after its results for a crash on the
    "all" token group (no change to computed values). `harness/analyze_i.py` (named in prereg-I) was first committed
    with its results, after two bugs were fixed: condition names copied from Study H, and the sign of the reduction
    interval. The same sign fix was applied to `harness/analyze_h.py`; no verdict changed. `harness/analyze_e.py` was
    committed 7.75 hours after `results/e.json`.

25. **Qwen's Study E result is quoted without the replacement G1 pre-registered.** Pre-registration G said Qwen's
    cut-off Study E replies would be replaced by redone ones; with the replacement Qwen's change in d′ is +0.16
    [−0.16, +0.47] (G1 part c), against +0.02 without. The paper now gives both.

26. **Fewer bootstrap resamples than pre-registered** (10,000): `harness/sdt.py` used 2,000, `harness/analyze_g.py`
    5,000, and Figure 1 600. Both scripts now use 10,000. Across the 50 intervals in `results/sdt.json` the largest
    change at either end is 0.034; one verdict changed (Mistral Medium 3's threshold interval, [−0.33, −0.003] before,
    now [−0.33, +0.003], which the paper does not use). Across the 11 intervals in `results/g.json` the largest change
    is 0.021, with no verdict changed. Figure 1 now takes its intervals from `results/sdt.json`.

27. **No correction for multiple tests.** Pre-registration A named a Benjamini-Hochberg correction for its secondary
    hypotheses; it was never applied. The paper now states which judgement results do not survive a Holm correction.

28. **G3's judgement prediction was scored on its point estimate** (author d′ change 0.13 against a limit of 0.15),
    while its interval reaches 0.23; Study D's matching prediction was scored on the whole interval. G4's prediction
    about the threshold was never computed (it holds).

29. **The never-change prompt was pre-registered as exploratory and the "change only for a reason" prompt was not
    pre-registered.** Both are now labelled exploratory in the paper, with denominators.

30. **Study J's census prediction passed on its pre-registered criterion** (the named neuron L24_E18_N122, negative
    side, ranked 219 of 6,488, within the top 5%), but that neuron reached under 1% of its atlas threshold; we report
    the pass and the activation together and do not read it as a credibility neuron firing.

31. **Study J3's attribution gate failed** (Pearson r = 0.18). Pre-registration J says the removals are then reported
    but not read as a test of Goodfire's method; an earlier draft did read them that way, and the paper no longer
    does. The verified 25 group (J3b, 18% [3, 32]) is now reported.

32. **The Study I analysis compared conditions that the paper's first draft described loosely**, and the honesty
    panel was described as 300 questions although Gemini 3.1 Pro Preview ran 150 (item 2). Both are corrected in the
    paper.


33. **Study L ran as pre-registered; two things were added after its data.** `harness/analyze_l.py` now also stores, for
    each edit, the share of Qwen's own conversations in which the next letter is the user's (hold margin below zero).
    It is descriptive, used only for Figure 1 (left), and leaves every pre-registered number unchanged (checked by
    comparing `results/l.json` before and after). The planted user-right arm of L3 has a gap too small to give a stable
    share (+0.09), so its shares (−117% to −200%) are reported but not interpreted.

34. **The paper was restructured around the neuron result after Study L**, following the rule written into
    pre-registration L before its data: the groups lead the paper only if L1 (both groups), L3 (both groups) and L5 (a
    and b) all hold. They did. L4 failed and is reported in the main text.

35. **The Fisher test in Appendix C (p = 0.017) had no script until after the audit.** `harness/fisher_c1.py` now
    computes it from the main-run files and reproduces the number; the paper also states that the split was chosen after
    seeing the pattern.

36. **Study B's dial check sits on the edge of passing for one setting** (addition to item 4). For k = 8 the change is
    −0.8 [−1.7, 0.0]: the interval ends at zero on the run of record and just excludes it under most bootstrap seeds.
    The paper's Appendix B now reports all four settings. We still read the study as not passed.

37. **Final review (15 Nov).** Three reviewing agents checked numbers, arguments and submission rules. Changes: sampled
    reply rates are now given next to the logit read-out; the atlas-coverage sentence was wrong and is corrected (the
    index covers ten of 40 layers); stale intervals fixed (G1 part c, J3, J1c); relevance patching is credited to
    Jafari et al. (2025); two exploratory analyses added on existing data (read-out against sampled replies, and the
    removal's effect controlling for the unedited margin).

38. **Study M: the top-100 copy was added after seeing M1** and is reported as exploratory. M1b compares the 25 only
    with the unedited model and the three random groups, as pre-registered.

39. **Study N's first design was dropped before its pre-registration was tagged.** Projecting the direction out of the
    weights moved the AI-label margin by more than the whole gap, so the design was changed to swapping one component
    between the two labelled versions of a conversation; the pre-registration describes the dropped design. N4 (written
    replies) used Hugging Face generation in left-padded batches with at most 6,000 new tokens, not the vLLM sampler
    with 16,000 used for unedited replies. Padding changes next-token probabilities slightly, so batches were sorted by
    length, all conditions of a conversation share a batch, and the padding is stored with each reply.

40. **Study O: one reported item was computed before the tag** (O7, how well the component separates Study M1's
    outcomes), and is marked exploratory in the pre-registration itself. The one-dimension case of O4 gave 43% where
    Study N gave 46% for the same swap; Study Q3 traced this to rounding in 16-bit arithmetic, not to a difference in
    the edit, and the paper reports the Study N numbers with the measured spread.

41. **Study P.** (a) The rule that paired comparisons use only conversations where both replies have a readable final
    line was introduced in pre-registration P, after a first look at 96 conversations of N4 showed the swap doubling
    cut-off replies; N4 itself is reported as pre-registered (unreadable counted as holding), and both countings are
    in `results/`. (b) P1 (thinking off) set a 1,500-token limit that more than half the replies exceeded, so it has
    about 50 usable pairs per comparison. (c) P2 used data that existed before the pre-registration (stated there).
    (d) The three-times strength run was announced in the pre-registration as exploratory.

42. **Studies Q1 and T take the AI-reviewer and author rows from Study M1's unedited replies** on the same held-out
    conversations (same sampler and settings, sampled a day earlier), as their pre-registrations say. Study R was
    registered after an exploratory look at Study P3 and before the Q1 replies existed; the Q chain's start time is in
    the box log.

43. **A second sample of S1 was added after seeing the first.** The pre-registered S1 sample failed its prediction
    (−6.8 points [−16.9, +3.4]). A second sample of the author label and the held-down direction was then run; alone
    it gives −13.3 [−25.0, −1.7] and pooled −9.6 [−17.6, −1.5]. Both are labelled exploratory.

44. **The direction's read-out result stays in the main text although pre-registration N sent it to the appendix.**
    N's rule: if N1 or N2 failed, the direction would be reported in the appendix as a negative result. Both failed
    on the half (47% and 46%). We keep two sentences on it in §4.4, with the miss stated, because the direction then
    passed two pre-registered tests in written replies (P4 and its repeat S2), which are the main result of that
    section. Pre-registration O said its tests would not change the status N's rule gave the direction; this item
    records that the later studies did.

45. **The paper was restructured a second time, after Studies M to T (16 Nov).** Under pre-registration L's rule the 25
    neurons led the earlier draft. Study M then showed they do not change written replies and remove 26% to 31% of the
    read-out difference with other wording, so they moved to §4.5 and Appendix C, and the title changed. The equal-weight
    accuracy by label and the sort order of Figure 1 were not pre-registered beyond "reported without a prediction".
