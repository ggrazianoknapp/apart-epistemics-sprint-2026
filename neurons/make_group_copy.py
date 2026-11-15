"""Make a copy of Qwen3.6-35B-A3B with one group of neurons removed (output column set to zero), for vLLM.

All files are symlinks to the original except the safetensors shards that hold an edited layer's experts.down_proj.
Usage: python3 make_group_copy.py GROUP     (attr25, or rand25_J as built in Study L)
Output dir: /workspace/models/m_GROUP
"""
import json, os, sys
from collections import defaultdict
from safetensors.torch import load_file, save_file

SRC = "/workspace/models/Qwen3.6-35B-A3B"
name = sys.argv[1]
if name.startswith("attr"):
    group = json.load(open("/workspace/runs/j3_results.json"))["groups"][name]
else:
    group = json.load(open("/workspace/runs/l_results.json"))["groups"][name]
group = [tuple(x) for x in group]
dst = f"/workspace/models/m_{name}"
wmap = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
by_shard = defaultdict(list)
for L, E, N in group:
    key = f"model.language_model.layers.{L}.mlp.experts.down_proj"
    by_shard[wmap[key]].append((key, E, N))
os.makedirs(dst, exist_ok=True)
for f in os.listdir(SRC):
    p = os.path.join(dst, f)
    if f not in by_shard and not os.path.lexists(p) and not f.startswith("."):
        os.symlink(os.path.join(SRC, f), p)
done = 0
for shard, edits in by_shard.items():
    t = load_file(f"{SRC}/{shard}")
    for key, E, N in edits:
        assert float(t[key][E, :, N].float().abs().sum()) > 0, "column already zero"
        t[key][E, :, N] = 0
        done += 1
    save_file(t, f"{dst}/{shard}", metadata={"format": "pt"})
# check by reloading: every edited column is zero and a neighbour column is untouched
for shard, edits in by_shard.items():
    t, o = load_file(f"{dst}/{shard}"), load_file(f"{SRC}/{shard}")
    for key, E, N in edits:
        assert float(t[key][E, :, N].float().abs().sum()) == 0
    key, E, N = edits[0]
    nb = next(n for n in range(512) if (int(key.split(".")[3]), E, n) not in group)
    assert bool((t[key][E, :, nb] == o[key][E, :, nb]).all())
print(f"wrote {dst}: {done} neurons removed across {len(by_shard)} shards, checked")
