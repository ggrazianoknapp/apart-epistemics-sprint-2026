"""E0 gates for the neuron readout (pre-registered): run on the GPU box with vLLM stopped.

1. Hugging Face reproduces vLLM's greedy tokens and log-probs on 8 reference generations
   (top-1 agreement >= 99%, mean |log-prob difference| <= 0.05).
2. The MoE block rebuilt from its weights matches the model's own output within 1% (layers 9, 14, 24),
   and the recomputed routing matches the routing the model used.
3. Zeroing the output weights of neurons in experts never routed on these inputs changes the logits by exactly 0.
4. Which pole of L9_E98_N330 (Goodfire's self-questioning neuron) fires on "Wait" in Qwen's own pilot thinking.
Writes /workspace/runs/e0_gates.json.
"""
import json, re, statistics as st
import torch
from readout_lib import load, Recorder, neuron_acts, rebuild_moe

tok, model = load()
res = {}
refs = json.load(open("/workspace/runs/vllm_reference.json"))

# 1. log-prob agreement with vLLM
agree, diffs = [], []
with torch.no_grad():
    for r in refs:
        ids = r["prompt_ids"] + r["gen_ids"]
        n = len(r["gen_ids"])
        logits = model(torch.tensor([ids], device="cuda"), logits_to_keep=n + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        gen = torch.tensor(r["gen_ids"], device="cuda")
        agree += (lp.argmax(-1) == gen).tolist()
        hf = lp.gather(1, gen[:, None])[:, 0].tolist()
        diffs += [abs(a - b) for a, b in zip(hf, r["gen_logprobs"]) if b is not None]
res["top1_agreement"] = sum(agree) / len(agree)
res["mean_abs_logprob_diff"] = st.mean(diffs)
res["gate1_pass"] = res["top1_agreement"] >= 0.99 and res["mean_abs_logprob_diff"] <= 0.05
print("gate 1:", res["top1_agreement"], res["mean_abs_logprob_diff"], res["gate1_pass"], flush=True)

# 2. MoE rebuild and routing
layers = [9, 14, 24]
rec = Recorder(model, layers, keep_output=True)
ids = torch.tensor([refs[0]["prompt_ids"] + refs[0]["gen_ids"]], device="cuda")
with torch.no_grad():
    model(ids, logits_to_keep=1)
rel_err, route_match = {}, {}
with torch.no_grad():
    for L in layers:
        x_in, y = rec.out[L]
        x2 = x_in.view(-1, x_in.shape[-1])
        rebuilt, top_i, top_w = rebuild_moe(model, L, x2)
        rel_err[L] = ((rebuilt - y.view(-1, y.shape[-1]).float()).norm() / y.float().norm()).item()
        _, idx, _ = rec.data[L]
        route_match[L] = (top_i.sort(-1).values == idx.sort(-1).values).all(-1).float().mean().item()
rec.remove()
res["moe_rel_err"], res["routing_match"] = rel_err, route_match
res["gate2_pass"] = all(v < 0.01 for v in rel_err.values()) and all(v > 0.99 for v in route_match.values())
print("gate 2:", rel_err, route_match, res["gate2_pass"], flush=True)

# 3. never-routed experts: zero some of their neurons' output weights; logits must not move at all
L = 24
rec = Recorder(model, [L])
used = set()
with torch.no_grad():
    for r in refs:
        model(torch.tensor([r["prompt_ids"] + r["gen_ids"]], device="cuda"), logits_to_keep=1)
        used |= set(rec.data[L][1].unique().tolist())
rec.remove()
unused = [e for e in range(256) if e not in used][:4]
res["never_routed_experts_L24"] = unused
if unused:
    D = model.model.layers[L].mlp.experts.down_proj
    with torch.no_grad():
        base = model(ids, logits_to_keep=50).logits.float()
        saved = D[unused, :, :16].clone()
        D[unused, :, :16] = 0
        edited = model(ids, logits_to_keep=50).logits.float()
        D[unused, :, :16] = saved
    res["never_routed_max_logit_change"] = (edited - base).abs().max().item()
    res["gate3_pass"] = res["never_routed_max_logit_change"] == 0.0
else:
    res["gate3_pass"] = None
print("gate 3:", res.get("never_routed_max_logit_change"), res["gate3_pass"], flush=True)

# 4. which pole of L9_E98_N330 fires on "Wait" in Qwen's own pilot thinking
calls = [json.loads(l) for l in open("/workspace/runs/pilot_qwen_calls.jsonl")]
qs = {json.loads(l)["qid"]: json.loads(l) for l in open("/workspace/data/questions_pilot.jsonl")}
LET = "ABCDEFGHIJ"
rec = Recorder(model, [9])
at_wait, before_wait, other, routed_frac = [], [], [], []
with torch.no_grad():
    for c in calls:
        if not c["step_id"].endswith("|t1") or not c.get("reasoning"):
            continue
        q = qs[c["step_id"].split("|")[1]]
        opts = "\n".join(f"({LET[i]}) {o}" for i, o in enumerate(q["options"]))
        user = f"Question: {q['question']}\n\nOptions:\n{opts}"
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        text = prompt + c["reasoning"]
        ids_ = tok(text, add_special_tokens=False).input_ids[:6000]
        model(torch.tensor([ids_], device="cuda"), logits_to_keep=1)
        x, idx, w = rec.data[9]
        a = neuron_acts(model, 9, x, idx, w, [98], [330])[:, 0].tolist()
        routed_frac.append(sum(1 for v in a if v != 0) / len(a))
        toks = tok.convert_ids_to_tokens(ids_)
        for i, t in enumerate(toks):
            word = tok.convert_tokens_to_string([t]).strip()
            if word == "Wait":
                at_wait.append(a[i])
                if i > 0:
                    before_wait.append(a[i - 1])
            else:
                other.append(a[i])
rec.remove()
nz = lambda v: [x for x in v if x != 0]
res["wait_neuron"] = {
    "n_wait_tokens": len(at_wait), "mean_at_wait": st.mean(at_wait) if at_wait else None,
    "mean_before_wait": st.mean(before_wait) if before_wait else None,
    "mean_other": st.mean(other), "share_routed_at_wait": len(nz(at_wait)) / max(1, len(at_wait)),
    "share_routed_other": len(nz(other)) / max(1, len(other)), "mean_share_routed_per_trace": st.mean(routed_frac)}
print("wait neuron:", res["wait_neuron"], flush=True)
json.dump(res, open("/workspace/runs/e0_gates.json", "w"), indent=1)
print("saved /workspace/runs/e0_gates.json")
