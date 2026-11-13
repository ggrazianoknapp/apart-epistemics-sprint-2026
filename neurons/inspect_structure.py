"""Print the text model's module layout and the experts' forward signature, without loading weights."""
import inspect, torch
from transformers import AutoConfig, AutoModelForCausalLM
MODEL = "/workspace/models/Qwen3.6-35B-A3B"
cfg = AutoConfig.from_pretrained(MODEL)
with torch.device("meta"):
    m = AutoModelForCausalLM.from_config(cfg)
print(type(m).__name__)
for name, _ in list(m.named_modules())[:3]:
    print(" ", name)
L = m.model.layers[9]
print("layer 9:", type(L).__name__)
print(" mlp:", type(L.mlp).__name__, [n for n, _ in L.mlp.named_children()])
print(" experts params:", {n: tuple(p.shape) for n, p in L.mlp.experts.named_parameters()})
print(" gate params:", {n: tuple(p.shape) for n, p in L.mlp.gate.named_parameters()})
print(" experts.forward", inspect.signature(type(L.mlp.experts).forward))
print(" gate.forward", inspect.signature(type(L.mlp.gate).forward))
print(" mlp.forward", inspect.signature(type(L.mlp).forward))
src = inspect.getsource(type(L.mlp).forward)
print(src[:1500])
print("layer types:", getattr(cfg, "text_config", cfg).layer_types[:8], "...")
print("state dict key example:", [k for k in m.state_dict().keys() if ".layers.9.mlp" in k][:8])
