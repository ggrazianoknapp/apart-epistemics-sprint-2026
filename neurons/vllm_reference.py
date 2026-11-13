"""Save vLLM's greedy continuations and per-token log-probs for 8 pilot prompts (reference for the E0 readout gate)."""
import json, requests
from transformers import AutoTokenizer
MODEL = "/workspace/models/Qwen3.6-35B-A3B"
tok = AutoTokenizer.from_pretrained(MODEL)
LETTERS = "ABCDEFGHIJ"
qs = [json.loads(l) for l in open("/workspace/data/questions_pilot.jsonl")][:8]
out = []
for q in qs:
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
    msg = f"Answer the following multiple-choice question.\n\nQuestion: {q['question']}\n\nOptions:\n{opts}"
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False).input_ids
    r = requests.post("http://127.0.0.1:8000/v1/completions", json={
        "model": "qwen", "prompt": ids, "max_tokens": 384, "temperature": 0, "logprobs": 1,
        "return_tokens_as_token_ids": True}, timeout=600).json()
    lp = r["choices"][0]["logprobs"]
    gen_ids = [int(t.split(":")[1]) for t in lp["tokens"]]
    out.append({"qid": q["qid"], "prompt_ids": ids, "gen_ids": gen_ids, "gen_logprobs": lp["token_logprobs"]})
json.dump(out, open("/workspace/runs/vllm_reference.json", "w"))
print(len(out), "references saved; tokens per gen:", [len(o["gen_ids"]) for o in out])
print("sample text:", repr(tok.decode(out[0]["gen_ids"][:60])))
