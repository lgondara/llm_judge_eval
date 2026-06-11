"""Data loading: UltraFeedback with natural quality variance.

Each UltraFeedback prompt carries 4 model completions annotated with GPT-4
overall_score (1-10). We use these annotations ONLY as an external reference
ranking for mechanism analysis -- never as a training signal -- so the
curation pipeline remains fully judge-driven.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional




@dataclass
class Candidate:
    sample_id: str
    prompt_id: str
    instruction: str
    response: str
    ref_score: Optional[float] = None   # GPT-4 overall_score (reference only)
    scores: dict = field(default_factory=dict)  # strategy -> normalized score


def load_ultrafeedback(
    num_prompts: int,
    num_eval_prompts: int,
    min_chars: int = 20,
    max_chars: int = 6000,
    seed: int = 42,
) -> tuple[list[Candidate], list[dict]]:
    """Return (train_candidates, eval_prompts).

    eval_prompts: [{prompt_id, instruction, reference_response}] where the
    reference is the highest GPT-4-rated completion (used as the fixed
    opponent in downstream win-rate evaluation).
    """
    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    train_candidates: list[Candidate] = []
    eval_prompts: list[dict] = []
    n_train, n_eval = 0, 0

    for idx in indices:
        if n_train >= num_prompts and n_eval >= num_eval_prompts:
            break
        item = ds[idx]
        instruction = (item.get("instruction") or "").strip()
        completions = item.get("completions") or []
        if not instruction or len(completions) < 2:
            continue

        cands = []
        for j, comp in enumerate(completions):
            resp = (comp.get("response") or "").strip()
            if not (min_chars <= len(resp) <= max_chars):
                continue
            try:
                ref = float(comp.get("overall_score"))
            except (TypeError, ValueError):
                ref = None
            cands.append((j, resp, ref))
        if len(cands) < 2:
            continue

        pid = f"uf_{idx}"
        if n_eval < num_eval_prompts:
            # Held-out eval prompt: reference = best GPT-4-rated completion.
            scored = [c for c in cands if c[2] is not None]
            best = max(scored or cands, key=lambda c: c[2] or 0.0)
            eval_prompts.append({
                "prompt_id": pid,
                "instruction": instruction,
                "reference_response": best[1],
            })
            n_eval += 1
        elif n_train < num_prompts:
            for j, resp, ref in cands:
                train_candidates.append(Candidate(
                    sample_id=f"{pid}_r{j}", prompt_id=pid,
                    instruction=instruction, response=resp, ref_score=ref,
                ))
            n_train += 1

    return train_candidates, eval_prompts


def save_candidates(cands: list[Candidate], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cands:
            f.write(json.dumps(asdict(c)) + "\n")


def load_candidates(path: str | Path) -> list[Candidate]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(Candidate(**json.loads(line)))
    return out
