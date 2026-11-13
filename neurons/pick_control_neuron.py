"""Pick a random layer-9 neuron matched to L9_E98_N330 on routing frequency and activation size (Qwen's pilot thinking).
Writes /workspace/runs/control_neuron.json."""
import json, random
import torch
from readout_lib import load, Recorder, neuron_acts

LET = "ABCDEFGHIJ"
tok, model = load()
rng = random.Random(11)
cands = [(e, n) for e, n in {(rng.randrange(256), rng.randrange(512)) for _ in range(2200)} if e != 98][:2000]
pairs = [(98, 330)] + cands
calls = [json.loads(l) for l in open("/workspace/runs/pilot_qwen_calls.jsonl")]
qs = {json.loads(l)["qid"]: json.loads(l) for l in open("/workspace/data/questions_pilot.jsonl")}
rec = Recorder(model, [9])
routed = torch.zeros(len(pairs), device="cuda"); absum = torch.zeros(len(pairs), device="cuda"); ntok = 0
with torch.no_grad():
    for c in calls:
        if not c["step_id"].endswith("|t1") or not c.get("reasoning"):
            continue
        q = qs[c["step_id"].split("|")[1]]
        user = f"Question: {q['question']}\n\nOptions:\n" + "\n".join(f"({LET[i]}) {o}" for i, o in enumerate(q["options"]))
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + c["reasoning"]
        ids = tok(text, add_special_tokens=False).input_ids[:6000]
        model(torch.tensor([ids], device="cuda"), logits_to_keep=1)
        x, idx, w = rec.data[9]
        a = neuron_acts(model, 9, x, idx, w, [p[0] for p in pairs], [p[1] for p in pairs])
        routed += (a != 0).float().sum(0); absum += a.abs().sum(0); ntok += a.shape[0]
freq = (routed / ntok).tolist(); scale = (absum / routed.clamp(min=1)).tolist()
tf, ts = freq[0], scale[0]
best = min(range(1, len(pairs)), key=lambda i: abs(freq[i] - tf) / max(tf, 1e-6) + abs(scale[i] - ts) / max(ts, 1e-6))
out = {"target": {"pair": pairs[0], "routed_frac": tf, "mean_abs_when_routed": ts},
       "control": {"pair": pairs[best], "routed_frac": freq[best], "mean_abs_when_routed": scale[best]}}
json.dump(out, open("/workspace/runs/control_neuron.json", "w"), indent=1)
print(out)
