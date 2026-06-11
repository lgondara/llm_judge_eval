"""Scoring strategies: the experimental treatment.

Five strategies share an identical judge model, identical generation
parameters, and minimally different prompts, so the *only* manipulated
variable is the scoring format:

  binary             pointwise  {0,1}     ACCEPT/REJECT
  likert_5           pointwise  1-5       ordinal rating
  numeric_100        pointwise  0-100     fine-grained score
  anchored           pairwise   [-2,+2]   vs. fixed per-prompt reference
  pairwise_tournament pairwise  BT score  within-prompt round robin

All raw scores are mapped to [0,1] (normalized_score) for comparability;
selection always uses within-strategy percentile ranks, never raw values.
"""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Judge backend (vLLM preferred; HF fallback for debugging)
# ----------------------------------------------------------------------------

class JudgeBackend:
    """Batched greedy generation behind a single .generate(prompts) call."""

    def __init__(self, model_name: str, backend: str = "vllm",
                 max_new_tokens: int = 16, max_model_len: int = 4096,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 enable_thinking: bool = False):
        self.model_name = model_name
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        if backend == "vllm":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, max_model_len=max_model_len,
                           tensor_parallel_size=tensor_parallel_size,
                           gpu_memory_utilization=gpu_memory_utilization)
            self.sampling = SamplingParams(temperature=0.0,
                                           max_tokens=max_new_tokens)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.model.eval()

    def _apply_template(self, user_msg: str) -> str:
        messages = [{"role": "user", "content": user_msg}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, user_msgs: list[str]) -> list[str]:
        prompts = [self._apply_template(m) for m in user_msgs]
        if self.backend == "vllm":
            outs = self.llm.generate(prompts, self.sampling)
            return [o.outputs[0].text.strip() for o in outs]
        # HF fallback (slow; debugging only)
        import torch
        results = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt",
                                    truncation=True, max_length=3584)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False,
                                          max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(text.strip())
        return results


# ----------------------------------------------------------------------------
# Strategy definitions
# ----------------------------------------------------------------------------

@dataclass
class ScoreRecord:
    sample_id: str
    raw: Optional[float]
    normalized: Optional[float]   # None => parse failure
    raw_output: str


class Strategy(ABC):
    name: str

    @abstractmethod
    def build_prompts(self, candidates: list) -> list[str]: ...

    @abstractmethod
    def parse(self, outputs: list[str], candidates: list) -> list[ScoreRecord]: ...

    def score(self, backend: JudgeBackend, candidates: list,
              batch_size: int = 256) -> list[ScoreRecord]:
        prompts = self.build_prompts(candidates)
        outputs = []
        for i in range(0, len(prompts), batch_size):
            outputs.extend(backend.generate(prompts[i:i + batch_size]))
        return self.parse(outputs, candidates)


_POINTWISE_TEMPLATE = """You are evaluating the quality of a response to an instruction. \
Consider helpfulness, correctness, completeness, and clarity.

Instruction: {instruction}

Response: {response}

{format_line}"""


def _extract_number(text: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


class BinaryStrategy(Strategy):
    name = "binary"
    FORMAT = "Is this response of acceptable quality? Reply with exactly one word: ACCEPT or REJECT."

    def build_prompts(self, candidates):
        return [_POINTWISE_TEMPLATE.format(instruction=c.instruction,
                                           response=c.response,
                                           format_line=self.FORMAT)
                for c in candidates]

    def parse(self, outputs, candidates):
        recs = []
        for out, c in zip(outputs, candidates):
            up = out.upper()
            if "ACCEPT" in up and "REJECT" not in up:
                raw, norm = 1.0, 1.0
            elif "REJECT" in up:
                raw, norm = 0.0, 0.0
            else:
                raw, norm = None, None
            recs.append(ScoreRecord(c.sample_id, raw, norm, out))
        return recs


class LikertStrategy(Strategy):
    name = "likert_5"
    FORMAT = ("Rate the response on a 1-5 scale "
              "(1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent). "
              "Reply with only the number.")

    def build_prompts(self, candidates):
        return [_POINTWISE_TEMPLATE.format(instruction=c.instruction,
                                           response=c.response,
                                           format_line=self.FORMAT)
                for c in candidates]

    def parse(self, outputs, candidates):
        recs = []
        for out, c in zip(outputs, candidates):
            v = _extract_number(out)
            if v is not None and 1 <= v <= 5:
                recs.append(ScoreRecord(c.sample_id, v, (v - 1) / 4, out))
            else:
                recs.append(ScoreRecord(c.sample_id, None, None, out))
        return recs


class NumericStrategy(Strategy):
    name = "numeric_100"
    FORMAT = ("Rate the response on a 0-100 scale, where 0 is unusable and "
              "100 is flawless. Reply with only the number.")

    def build_prompts(self, candidates):
        return [_POINTWISE_TEMPLATE.format(instruction=c.instruction,
                                           response=c.response,
                                           format_line=self.FORMAT)
                for c in candidates]

    def parse(self, outputs, candidates):
        recs = []
        for out, c in zip(outputs, candidates):
            v = _extract_number(out)
            if v is not None and 0 <= v <= 100:
                recs.append(ScoreRecord(c.sample_id, v, v / 100, out))
            else:
                recs.append(ScoreRecord(c.sample_id, None, None, out))
        return recs


_PAIRWISE_TEMPLATE = """You are comparing two responses to the same instruction. \
Consider helpfulness, correctness, completeness, and clarity.

Instruction: {instruction}

Response A: {response_a}

Response B: {response_b}

Which response is better? Reply with exactly one of:
A++ (A much better), A+ (A slightly better), TIE, B+ (B slightly better), B++ (B much better)."""

_VERDICT_MAP = {"A++": 2.0, "A+": 1.0, "TIE": 0.0, "B+": -1.0, "B++": -2.0}


def _parse_verdict(text: str) -> Optional[float]:
    up = text.upper()
    # Longest tokens first so "A++" is not matched as "A+".
    for tok in ("A++", "B++", "A+", "B+", "TIE"):
        if tok in up:
            return _VERDICT_MAP[tok]
    return None


class AnchoredStrategy(Strategy):
    """Each candidate compared against a fixed per-prompt reference.

    Reference = the median-ref_score response for the prompt (or middle by
    length when annotations are absent), held fixed across all candidates of
    that prompt. Score in [-2,+2] is the candidate's margin over the anchor.
    """
    name = "anchored"

    def __init__(self, reference_policy: str = "dataset_median"):
        self.reference_policy = reference_policy
        self._anchors: dict[str, str] = {}

    def _build_anchors(self, candidates):
        by_prompt: dict[str, list] = {}
        for c in candidates:
            by_prompt.setdefault(c.prompt_id, []).append(c)
        for pid, group in by_prompt.items():
            key = (lambda c: c.ref_score) if all(
                g.ref_score is not None for g in group) else (
                lambda c: len(c.response))
            group_sorted = sorted(group, key=key)
            self._anchors[pid] = group_sorted[len(group_sorted) // 2].response

    def build_prompts(self, candidates):
        self._build_anchors(candidates)
        return [_PAIRWISE_TEMPLATE.format(instruction=c.instruction,
                                          response_a=c.response,
                                          response_b=self._anchors[c.prompt_id])
                for c in candidates]

    def parse(self, outputs, candidates):
        recs = []
        for out, c in zip(outputs, candidates):
            v = _parse_verdict(out)
            norm = None if v is None else (v + 2) / 4
            recs.append(ScoreRecord(c.sample_id, v, norm, out))
        return recs


class PairwiseTournamentStrategy(Strategy):
    """Within-prompt round robin; Bradley-Terry strengths -> [0,1] scores.

    Comparison count is C(k,2) per prompt (k<=4 for UltraFeedback => <=6),
    so the cost is bounded and directly comparable to pointwise budgets.
    """
    name = "pairwise_tournament"

    def __init__(self, max_comparisons_per_prompt: int = 6):
        self.max_comp = max_comparisons_per_prompt
        self._pairs: list[tuple] = []

    def build_prompts(self, candidates):
        by_prompt: dict[str, list] = {}
        for c in candidates:
            by_prompt.setdefault(c.prompt_id, []).append(c)
        self._pairs = []
        prompts = []
        for pid, group in by_prompt.items():
            combos = list(itertools.combinations(group, 2))[: self.max_comp]
            for a, b in combos:
                self._pairs.append((a, b))
                prompts.append(_PAIRWISE_TEMPLATE.format(
                    instruction=a.instruction,
                    response_a=a.response, response_b=b.response))
        return prompts

    def parse(self, outputs, candidates):
        # Aggregate verdicts into win counts, then one-step BT estimate.
        wins: dict[str, float] = {c.sample_id: 0.0 for c in candidates}
        games: dict[str, float] = {c.sample_id: 0.0 for c in candidates}
        for out, (a, b) in zip(outputs, self._pairs):
            v = _parse_verdict(out)
            if v is None:
                continue
            p_a = 1 / (1 + math.exp(-v))   # logistic margin -> win prob
            wins[a.sample_id] += p_a
            wins[b.sample_id] += 1 - p_a
            games[a.sample_id] += 1
            games[b.sample_id] += 1
        recs = []
        for c in candidates:
            if games[c.sample_id] == 0:
                recs.append(ScoreRecord(c.sample_id, None, None, ""))
            else:
                wr = wins[c.sample_id] / games[c.sample_id]
                recs.append(ScoreRecord(c.sample_id, wr, wr, ""))
        return recs


STRATEGIES = {
    "binary": BinaryStrategy,
    "likert_5": LikertStrategy,
    "numeric_100": NumericStrategy,
    "anchored": AnchoredStrategy,
    "pairwise_tournament": PairwiseTournamentStrategy,
}


# ----------------------------------------------------------------------------
# Distribution diagnostics (mechanism analysis)
# ----------------------------------------------------------------------------

def score_diagnostics(records: list[ScoreRecord]) -> dict:
    vals = np.array([r.normalized for r in records if r.normalized is not None])
    n = len(records)
    if len(vals) == 0:
        return {"parse_rate": 0.0}
    unique, counts = np.unique(vals, return_counts=True)
    tie_mass = float(np.sum((counts / len(vals)) ** 2))  # P(two random samples tie)
    return {
        "parse_rate": float(len(vals) / n),
        "n_unique_values": int(len(unique)),
        "tie_probability": tie_mass,
        "effective_range_p95_p5": float(np.percentile(vals, 95) - np.percentile(vals, 5)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float(((vals - vals.mean()) ** 3).mean() / (vals.std() ** 3 + 1e-12)),
    }
