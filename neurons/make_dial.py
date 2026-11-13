"""Make a dialled copy of Qwen3.6-35B-A3B: neuron (layer, expert, neuron) has its output column multiplied by k.

All files are symlinks to the original except the one safetensors shard that holds that layer's experts.down_proj.
Usage: python3 make_dial.py K [LAYER EXPERT NEURON]   (default neuron: L9_E98_N330)
Output dir: /workspace/models/dial_L{L}_E{E}_N{N}_k{K}
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

SRC = "/workspace/models/Qwen3.6-35B-A3B"
k = float(sys.argv[1])
L, E, N = (int(a) for a in sys.argv[2:5]) if len(sys.argv) >= 5 else (9, 98, 330)
dst = f"/workspace/models/dial_L{L}_E{E}_N{N}_k{sys.argv[1]}"
key = f"model.language_model.layers.{L}.mlp.experts.down_proj"
shard = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"][key]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(SRC):
    p = os.path.join(dst, f)
    if f != shard and not os.path.exists(p) and not f.startswith("."):
        os.symlink(os.path.join(SRC, f), p)
t = load_file(f"{SRC}/{shard}")
before = t[key][E, :, N].float().clone()
t[key][E, :, N] = (t[key][E, :, N].float() * k).to(t[key].dtype)
assert torch.equal(t[key][E, :, N].float(), before * k) or k not in (0, 1, 2, 4, -2, -4, 8, -8), "inexact edit"
others_same = torch.equal(load_file(f"{SRC}/{shard}")[key][E, :, (N + 1) % 512], t[key][E, :, (N + 1) % 512])
save_file(t, f"{dst}/{shard}", metadata={"format": "pt"})
print(f"wrote {dst} (edited {key}[{E}, :, {N}] x {k} in {shard}; neighbour column unchanged: {others_same})")
