"""Provider-agnostic API judge backend (OpenAI-compatible).

Drop-in replacement for scoring.JudgeBackend: exposes the same
`.generate(list[str]) -> list[str]` interface, so the score and judge_eval
stages use it unchanged. Works with any OpenAI-compatible endpoint
(DeepSeek, OpenAI, Together, Fireworks, vLLM's own server, Ollama, ...) by
setting three config values: base_url, model, api_key_env.

Production essentials:
  * bounded concurrency (async, capped) so we saturate the API without
    tripping rate limits;
  * retry with exponential backoff + jitter on transient errors;
  * on-disk cache keyed by a hash of (model, prompt, sampling params) so a
    crash never re-bills completed calls and re-runs are free;
  * deterministic ordering: outputs are returned in input order regardless
    of completion order.

Requires: pip install openai
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Optional


class APIJudgeBackend:
    """OpenAI-compatible chat-completions backend with the JudgeBackend API."""

    def __init__(self, model: str, base_url: str,
                 api_key_env: str = "DEEPSEEK_API_KEY",
                 max_new_tokens: int = 16,
                 temperature: float = 0.0,
                 max_concurrency: int = 16,
                 max_retries: int = 6,
                 cache_dir: str | Path = "outputs/_api_cache",
                 enable_thinking: bool = False,
                 system_prompt: Optional[str] = None,
                 request_timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt
        self.request_timeout = request_timeout

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found in env var {api_key_env!r}. "
                f"Set it, e.g.: export {api_key_env}=sk-...")
        self.api_key = api_key

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # tokenizer kept for parity with vLLM backend (chat templating is
        # server-side here, so this is unused but harmless to skip).
        self.tokenizer = None

    # -- caching -------------------------------------------------------------

    def _cache_key(self, prompt: str) -> str:
        h = hashlib.sha256()
        # Include everything that changes the output, so cache is safe.
        h.update(self.model.encode())
        h.update(b"\x00")
        h.update(str(self.temperature).encode())
        h.update(b"\x00")
        h.update(str(self.max_new_tokens).encode())
        h.update(b"\x00")
        h.update(str(self.enable_thinking).encode())
        h.update(b"\x00")
        h.update((self.system_prompt or "").encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        # shard into subdirs to avoid one giant directory
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[str]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())["output"]
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, output: str) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"output": output}))

    # -- request -------------------------------------------------------------

    def _messages(self, prompt: str) -> list[dict]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def _one(self, client, sem, prompt: str) -> str:
        key = self._cache_key(prompt)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # DeepSeek V4 / OpenAI both accept extra_body; thinking-disable is a
        # no-op for providers that don't support it.
        extra_body = {}
        if not self.enable_thinking:
            extra_body["thinking"] = {"type": "disabled"}

        async with sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.chat.completions.create(
                        model=self.model,
                        messages=self._messages(prompt),
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                        timeout=self.request_timeout,
                        extra_body=extra_body or None,
                    )
                    out = (resp.choices[0].message.content or "").strip()
                    self._cache_put(key, out)
                    return out
                except Exception as e:  # noqa: BLE001 - provider-agnostic
                    if attempt == self.max_retries - 1:
                        # Give up on this prompt: return empty so the parser
                        # records a parse-failure rather than crashing the run.
                        self._cache_put(key, "")
                        return ""
                    backoff = min(2 ** attempt, 30) + random.uniform(0, 1)
                    await asyncio.sleep(backoff)
        return ""

    async def _run(self, prompts: list[str]) -> list[str]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._one(client, sem, p) for p in prompts]
        return await asyncio.gather(*tasks)

    # -- public interface (matches scoring.JudgeBackend) ---------------------

    def generate(self, user_msgs: list[str]) -> list[str]:
        """Synchronous wrapper; returns outputs in input order."""
        if not user_msgs:
            return []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Already inside an event loop (rare in this CLI); use a fresh one.
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                self._run(user_msgs))
        return asyncio.run(self._run(user_msgs))


def build_backend(cfg_judge: dict):
    """Factory: returns vLLM/HF JudgeBackend or APIJudgeBackend by config.

    cfg_judge["backend"] in {"vllm", "hf", "api"}.
    For "api", expects: model, base_url, api_key_env, and optionally
    max_concurrency, max_new_tokens, enable_thinking.
    """
    backend = cfg_judge.get("backend", "vllm")
    if backend == "api":
        return APIJudgeBackend(
            model=cfg_judge["model"],
            base_url=cfg_judge["base_url"],
            api_key_env=cfg_judge.get("api_key_env", "DEEPSEEK_API_KEY"),
            max_new_tokens=cfg_judge.get("max_new_tokens", 16),
            temperature=cfg_judge.get("temperature", 0.0),
            max_concurrency=cfg_judge.get("max_concurrency", 16),
            enable_thinking=cfg_judge.get("enable_thinking", False),
            cache_dir=cfg_judge.get("cache_dir", "outputs/_api_cache"),
        )
    from src.scoring import JudgeBackend
    return JudgeBackend(
        cfg_judge["model"], backend, cfg_judge.get("max_new_tokens", 16),
        cfg_judge.get("max_model_len", 4096),
        cfg_judge.get("tensor_parallel_size", 1),
        cfg_judge.get("gpu_memory_utilization", 0.9),
        cfg_judge.get("enable_thinking", False))
