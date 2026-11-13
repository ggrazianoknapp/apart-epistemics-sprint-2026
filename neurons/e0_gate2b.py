"""Gate 2, second version (a deviation, reported as such): rebuild the MoE block output using the routing the model
actually used (top_k_index and top_k_weights seen by the experts hook) instead of recomputing the router.
The pre-registered version recomputed the router in float32 and picked a different 8th expert for 12-15% of tokens.
Appends the result to /workspace/runs/e0_gates.json."""
import json
import torch
import torch.nn.functional as F
from readout_lib import load, Recorder, INTER

tok, model = load()
refs = json.load(open("/workspace/runs/vllm_reference.json"))
layers = [9, 14, 24]
rec = Recorder(model, layers, keep_output=True)
ids = torch.tensor([refs[0]["prompt_ids"] + refs[0]["gen_ids"]], device="cuda")
res = {}
with torch.no_grad():
    model(ids, logits_to_keep=1)
    for L in layers:
        mlp = model.model.layers[L].mlp
        x, idx, w = rec.data[L]
        _, y = rec.out[L]
        xf = x.float()
        G, D = mlp.experts.gate_up_proj, mlp.experts.down_proj
        out = torch.zeros_like(xf)
        for e in idx.unique().tolist():
            tp, kp = (idx == e).nonzero(as_tuple=True)
            h = F.silu(xf[tp] @ G[e, :INTER].float().T) * (xf[tp] @ G[e, INTER:].float().T)
            out.index_add_(0, tp, (h @ D[e].float().T) * w[tp, kp].float().unsqueeze(-1))
        out += mlp.shared_expert(x).float() * torch.sigmoid(mlp.shared_expert_gate(x).float())
        res[L] = ((out - y.view(-1, y.shape[-1]).float()).norm() / y.float().norm()).item()
r = json.load(open("/workspace/runs/e0_gates.json"))
r["moe_rel_err_model_routing"] = res
r["gate2b_pass"] = all(v < 0.01 for v in res.values())
json.dump(r, open("/workspace/runs/e0_gates.json", "w"), indent=1)
print("gate 2b (model's own routing):", res, r["gate2b_pass"])
