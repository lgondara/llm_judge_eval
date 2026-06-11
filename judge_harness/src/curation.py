"""Curation: convert strategy scores into training subsets.

Selection protocol (per_prompt_top1_then_global):
  1. Within each prompt, keep the highest-scored response (ties broken by a
     seeded RNG -- tie-handling is itself part of the phenomenon under study,
     so we log how many selections were tie-broken).
  2. Rank the surviving (prompt, response) pairs globally by score and keep
     the top retention fraction.

Percentile-based retention makes subsets comparable across strategies whose
raw distributions are incommensurable (a known pitfall: judges cluster raw
scores on strategy- and model-specific values).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class CurationResult:
    strategy: str
    retention: float
    sample_ids: list[str]
    n_tiebroken: int          # selections decided by RNG due to score ties
    overlap_key: frozenset    # for cross-strategy Jaccard


def curate(candidates: list, scores: dict[str, float], retention: float,
           seed: int = 42) -> CurationResult:
    """scores: sample_id -> normalized score (parse failures excluded)."""
    rng = random.Random(seed)

    by_prompt: dict[str, list] = {}
    for c in candidates:
        if c.sample_id in scores:
            by_prompt.setdefault(c.prompt_id, []).append(c)

    survivors, n_tiebroken = [], 0
    for pid, group in by_prompt.items():
        best_score = max(scores[c.sample_id] for c in group)
        best = [c for c in group if scores[c.sample_id] == best_score]
        if len(best) > 1:
            n_tiebroken += 1
        survivors.append(rng.choice(best))

    survivors.sort(key=lambda c: (scores[c.sample_id], rng.random()),
                   reverse=True)
    k = max(1, int(len(survivors) * retention))
    chosen = survivors[:k]
    ids = [c.sample_id for c in chosen]
    return CurationResult("", retention, ids, n_tiebroken, frozenset(ids))


def curate_random(candidates: list, retention: float, seed: int) -> CurationResult:
    rng = random.Random(seed)
    by_prompt: dict[str, list] = {}
    for c in candidates:
        by_prompt.setdefault(c.prompt_id, []).append(c)
    survivors = [rng.choice(g) for g in by_prompt.values()]
    rng.shuffle(survivors)
    k = max(1, int(len(survivors) * retention))
    ids = [c.sample_id for c in survivors[:k]]
    return CurationResult("random", retention, ids, 0, frozenset(ids))


def curate_length(candidates: list, retention: float, seed: int = 42) -> CurationResult:
    """Longest-response heuristic: the verbosity-bias control."""
    lengths = {c.sample_id: float(len(c.response)) for c in candidates}
    res = curate(candidates, lengths, retention, seed)
    res.strategy = "length"
    return res


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / max(1, len(a | b))


def selection_overlap_matrix(results: list[CurationResult]) -> dict:
    out = {}
    for i, r1 in enumerate(results):
        for r2 in results[i + 1:]:
            if r1.retention == r2.retention:
                out[(r1.strategy, r2.strategy, r1.retention)] = jaccard(
                    r1.overlap_key, r2.overlap_key)
    return out


def ref_ranking_correlation(candidates: list, scores: dict[str, float]) -> dict:
    """Spearman/Kendall of judge scores vs GPT-4 reference annotations.

    Mechanism analysis only -- reference scores never enter curation.
    """
    from scipy import stats
    pairs = [(scores[c.sample_id], c.ref_score) for c in candidates
             if c.sample_id in scores and c.ref_score is not None]
    if len(pairs) < 10:
        return {}
    s, r = zip(*pairs)
    rho, _ = stats.spearmanr(s, r)
    tau, _ = stats.kendalltau(s, r)
    return {"spearman_vs_ref": float(rho), "kendall_vs_ref": float(tau),
            "n": len(pairs)}
