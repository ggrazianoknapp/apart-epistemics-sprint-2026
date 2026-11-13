"""Read Goodfire-style MLP neurons in Qwen3.6-35B-A3B with Hugging Face Transformers.

A routed neuron (layer L, expert E, neuron N) has activation silu(x.G[E,N]) * (x.G[E,512+N]) * w_E, where x is the
MoE block input, G is experts.gate_up_proj [256, 1024, 2048] (gate rows 0-511, up rows 512-1023) and w_E is the
router weight of expert E for that token after the top-8 weights are renormalised to sum to 1 (0 if E is not routed).
This is Goodfire's "a * w_router". One pre-hook on model.model.layers[L].mlp.experts sees x, the chosen experts and
their weights, so we compute any neuron we want from the stored weights without patching the model.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/models/Qwen3.6-35B-A3B"
INTER = 512


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


class Recorder:
    """Stores (x, top_k_index, top_k_weights) at the MoE block of each chosen layer during a forward pass."""

    def __init__(self, model, layers, keep_output=False):
        self.data, self.out, self.handles = {}, {}, []
        for L in layers:
            mlp = model.model.layers[L].mlp
            self.handles.append(mlp.experts.register_forward_pre_hook(self._pre(L)))
            if keep_output:
                self.handles.append(mlp.register_forward_hook(self._post(L)))

    def _pre(self, L):
        def hook(mod, args):
            x, idx, w = args[0], args[1], args[2]
            self.data[L] = (x.detach(), idx.detach(), w.detach())
        return hook

    def _post(self, L):
        def hook(mod, args, output):
            self.out[L] = (args[0].detach(), output.detach())
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def neuron_acts(model, L, x, idx, w, experts, neurons):
    """Activations [T, n] for the (expert, neuron) pairs at layer L, in float32."""
    G = model.model.layers[L].mlp.experts.gate_up_proj
    E = torch.as_tensor(experts, device=x.device)
    N = torch.as_tensor(neurons, device=x.device)
    xf = x.float()
    a = F.silu(xf @ G[E, N, :].float().T) * (xf @ G[E, N + INTER, :].float().T)
    route = ((idx.unsqueeze(-1) == E.view(1, 1, -1)) * w.float().unsqueeze(-1)).sum(1)
    return a * route


def rebuild_moe(model, L, x):
    """Recompute the MoE block output from its weights (router, routed experts, shared expert)."""
    mlp = model.model.layers[L].mlp
    xf = x.float()
    probs = torch.softmax(xf @ mlp.gate.weight.float().T, dim=-1)
    top_w, top_i = torch.topk(probs, 8, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    G, D = mlp.experts.gate_up_proj, mlp.experts.down_proj
    out = torch.zeros_like(xf)
    for e in top_i.unique().tolist():
        tok_pos, k_pos = (top_i == e).nonzero(as_tuple=True)
        xe = xf[tok_pos]
        h = F.silu(xe @ G[e, :INTER].float().T) * (xe @ G[e, INTER:].float().T)
        out.index_add_(0, tok_pos, (h @ D[e].float().T) * top_w[tok_pos, k_pos].unsqueeze(-1))
    shared = mlp.shared_expert(x).float() * torch.sigmoid(mlp.shared_expert_gate(x).float())
    return out + shared, top_i, top_w
