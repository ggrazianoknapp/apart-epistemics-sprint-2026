"""Async chat client for OpenRouter and our own vLLM server, with retries, a call cache and a hard spend cap."""
import asyncio, hashlib, json, os, random, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER = "https://openrouter.ai/api/v1"
VLLM = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
QWEN_LOCAL = "qwen-local"  # model name we use for our own vLLM server

# Sprint spend is measured against the key's usage when the sprint started (13 Nov 2026).
BASELINE_USAGE = 5327.12
SPEND_CAP = float(os.environ.get("SPRINT_SPEND_CAP", "550"))


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY missing from .env")


class SpendCapReached(RuntimeError):
    pass


class Client:
    def __init__(self, cache_path, max_concurrency=16):
        self.key = load_key()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30))
        self.sem = asyncio.Semaphore(max_concurrency)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                r = json.loads(line)
                self.cache[r["cache_key"]] = r
        self.lock = asyncio.Lock()
        self.key_lock = asyncio.Lock()
        self._last_check, self._usage = 0.0, None

    async def sprint_spend(self):
        """Key usage above the sprint baseline, refreshed at most once a minute (one request at a time)."""
        async with self.key_lock:
            if self._usage is None or time.time() - self._last_check > 60:
                for attempt in range(6):
                    try:
                        r = await self.http.get(f"{OPENROUTER}/key", headers={"Authorization": f"Bearer {self.key}"})
                        self._usage = r.json()["data"]["usage"]
                        self._last_check = time.time()
                        break
                    except Exception as e:  # rate limit or network blip: keep the last known value and retry
                        print(f"key check failed ({attempt+1}): {repr(e)[:120]}", file=sys.stderr, flush=True)
                        await asyncio.sleep(5 * (attempt + 1))
                if self._usage is None:
                    raise RuntimeError("could not read the key's usage; refusing to spend blind")
        return self._usage - BASELINE_USAGE

    async def chat(self, model, messages, step_id, max_tokens=16000, reasoning="medium", temperature=None,
                   provider=None):
        """One chat completion, cached by step_id so reruns resume instead of paying twice."""
        ck = hashlib.sha1(f"{model}|{step_id}".encode()).hexdigest()
        if ck in self.cache:
            return self.cache[ck]
        local = model == QWEN_LOCAL
        if not local and await self.sprint_spend() >= SPEND_CAP:
            raise SpendCapReached(f"sprint spend reached the ${SPEND_CAP:.0f} cap")
        body = {"model": "qwen" if local else model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if local:
            body.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        else:
            body["usage"] = {"include": True}
            if reasoning:
                body["reasoning"] = {"effort": reasoning}
            if provider:  # pin one provider so a model is never served by differently quantised copies
                body["provider"] = {"order": [provider], "allow_fallbacks": False}
        url = f"{VLLM if local else OPENROUTER}/chat/completions"
        headers = {} if local else {"Authorization": f"Bearer {self.key}"}
        async with self.sem:
            for attempt in range(8):
                try:
                    r = await self.http.post(url, json=body, headers=headers)
                    if r.status_code == 200:
                        d = r.json()
                        if "choices" in d and d["choices"]:
                            break
                        err = json.dumps(d)[:300]
                    else:
                        err = f"HTTP {r.status_code}: {r.text[:300]}"
                        if "Workspace lifetime budget" in r.text or r.status_code == 402:
                            raise SpendCapReached(err)
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    err = repr(e)[:300]
                print(f"retry {attempt+1} {model} {step_id}: {err[:160]}", file=sys.stderr, flush=True)
                await asyncio.sleep(min(120, 2 ** attempt + random.random() * 3))
            else:
                rec = {"cache_key": ck, "model": model, "step_id": step_id, "error": err}
                await self._write(rec)
                return rec
        msg = d["choices"][0]["message"]
        u = d.get("usage") or {}
        rec = {"cache_key": ck, "model": model, "step_id": step_id,
               "content": msg.get("content") or "",
               "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
               "finish_reason": d["choices"][0].get("finish_reason"),
               "provider": d.get("provider"), "model_returned": d.get("model"),
               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               "cost": u.get("cost"), "t": time.time()}
        await self._write(rec)
        return rec

    async def _write(self, rec):
        async with self.lock:
            self.cache[rec["cache_key"]] = rec
            with self.cache_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
