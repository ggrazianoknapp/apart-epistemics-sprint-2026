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
   questions. The attribution set, the selective set and the positive controls were not run.

6. **Technical changes that do not affect any hypothesis:**
   - The LLM extractor's token budget was raised from 300 to 3,000, because both extractor models sometimes used
     the budget up on reasoning before answering.
   - Grok 4.7 and three slow reasoning models in the honesty-rank panel ran with more parallel calls.
