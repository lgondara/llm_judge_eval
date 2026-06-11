"""Statistical analysis for the strategy comparison.

Inference plan (pre-registered in the paper):

Primary question: Is there a universal best scoring strategy, or is the
optimal strategy task-dependent? Evidence for task-dependence requires
BOTH of the following (a quasi-null claim cannot rest on absence of a
winner alone):
  (1) Within-task differences exist: per-task Friedman test across
      strategies (blocks = eval prompts) is significant, with pairwise
      paired bootstrap + Holm identifying which strategies differ.
  (2) The strategy ordering is unstable across tasks: low Kendall's W
      concordance of strategy rankings across tasks, plus a permutation
      test on the strategy x task interaction (seed-level scores,
      task-standardized, main effects removed).

Universality is supported instead if (1) holds and Kendall's W is high
with the same strategy at the top of every task ranking.

Secondary machinery:
  * Unit of analysis (judged tasks): per-prompt outcome in {0, 0.5, 1},
    averaged over fine-tuning seeds within a cell. Benchmark tasks
    (IFEval, GSM8K) use per-instance accuracy as the unit.
  * Effect sizes: win-rate / accuracy differences with 95% bootstrap CIs.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """Paired bootstrap on per-prompt outcome differences (a - b)."""
    assert len(a) == len(b)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"mean_diff": float(diff.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_value": min(1.0, p)}


def holm_correction(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        out[key] = running_max
    return out


def analyze(results_dir: str | Path, retention_rates: list[float],
            strategies: list[str], seeds: list[int]) -> pd.DataFrame:
    """Aggregate eval JSONs -> summary table + pairwise tests.

    Expects files: {results_dir}/{strategy}_ret{R}_seed{S}/eval.json
    """
    results_dir = Path(results_dir)
    rows, per_prompt = [], {}

    for strat, ret, seed in itertools.product(strategies, retention_rates, seeds):
        f = results_dir / f"{strat}_ret{int(ret*100)}_seed{seed}" / "eval.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        rows.append({"strategy": strat, "retention": ret, "seed": seed,
                     "win_rate": data["win_rate"]})
        per_prompt.setdefault((strat, ret), []).append(
            np.array(data["per_prompt"]))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Seed-averaged per-prompt outcomes per cell.
    cell_outcomes = {k: np.mean(np.stack(v), axis=0)
                     for k, v in per_prompt.items()}

    summary = (df.groupby(["strategy", "retention"])["win_rate"]
                 .agg(["mean", "std", "count"]).reset_index())
    print("\n=== Win-rate summary (mean over seeds) ===")
    print(summary.to_string(index=False))

    for ret in retention_rates:
        cells = {s: cell_outcomes[(s, ret)] for s in strategies
                 if (s, ret) in cell_outcomes}
        if len(cells) < 2:
            continue
        names = list(cells)
        # Omnibus Friedman test (blocks = prompts).
        if len(names) >= 3:
            fr_stat, fr_p = stats.friedmanchisquare(*[cells[n] for n in names])
            print(f"\nRetention {ret:.0%}: Friedman chi2={fr_stat:.2f}, "
                  f"p={fr_p:.4f}")
        pvals, effects = {}, {}
        for a, b in itertools.combinations(names, 2):
            res = paired_bootstrap_diff(cells[a], cells[b])
            pvals[(a, b)] = res["p_value"]
            effects[(a, b)] = res
        adj = holm_correction(pvals)
        for (a, b), res in effects.items():
            print(f"  {a} vs {b}: d={res['mean_diff']:+.3f} "
                  f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}], "
                  f"p_holm={adj[(a, b)]:.4f}")

    return df


def curation_curve_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Trapezoidal AUC of win-rate over retention, per strategy."""
    out = []
    for strat, g in df.groupby("strategy"):
        m = g.groupby("retention")["win_rate"].mean().sort_index()
        if len(m) >= 2:
            out.append({"strategy": strat,
                        "auc": float(np.trapz(m.values, m.index.values)),
                        "peak": float(m.max()),
                        "peak_retention": float(m.idxmax())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Task-dependence analysis: the core of the revised research question
# ----------------------------------------------------------------------------

def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Concordance of strategy rankings across tasks.

    rank_matrix: (n_tasks, n_strategies) ranks, 1 = best within each task.
    W -> 1: tasks agree on the strategy ordering (universality).
    W -> 0: orderings are task-idiosyncratic (task-dependence).
    """
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    R = rank_matrix.sum(axis=0)
    S = float(((R - R.mean()) ** 2).sum())
    return 12 * S / (m ** 2 * (n ** 3 - n))


def interaction_permutation_test(df: pd.DataFrame, n_perm: int = 10000,
                                 seed: int = 0) -> dict:
    """Permutation test for strategy x task interaction.

    df columns: strategy, task, seed, score (seed-level cell scores).
    Scores are z-standardized within task (tasks live on different scales),
    main effects are removed, and the interaction sum of squares of the
    residual cell means is compared against a null built by permuting
    strategy labels within task (preserving task structure).
    """
    d = df.copy()
    d["z"] = d.groupby("task")["score"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))

    def interaction_ss(frame):
        cell = frame.groupby(["task", "strategy"])["z"].mean().unstack()
        resid = cell.sub(cell.mean(axis=1), axis=0) - cell.mean(axis=0) \
            + cell.values.mean()
        return float((resid.values ** 2).sum())

    observed = interaction_ss(d)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = d.copy()
        p["strategy"] = p.groupby("task")["strategy"].transform(
            lambda s: rng.permutation(s.values))
        if interaction_ss(p) >= observed:
            count += 1
    return {"interaction_ss": observed,
            "p_value": (count + 1) / (n_perm + 1)}


def task_dependence_report(df: pd.DataFrame) -> dict:
    """Full task-dependence analysis.

    df columns: strategy, task, seed, score. One row per (cell, seed).
    Returns rankings per task, Kendall's W, interaction test, and the
    per-task winner with its margin over the runner-up.
    """
    cell_means = (df.groupby(["task", "strategy"])["score"]
                    .mean().unstack())                     # tasks x strategies
    ranks = cell_means.rank(axis=1, ascending=False)
    w = kendalls_w(ranks.values)
    inter = interaction_permutation_test(df)

    winners = {}
    for task, row in cell_means.iterrows():
        ordered = row.sort_values(ascending=False)
        winners[task] = {"best": ordered.index[0],
                         "margin_over_2nd": float(ordered.iloc[0] - ordered.iloc[1])}

    # Top-1 agreement: fraction of task pairs sharing the same winner.
    # Kendall's W can stay high while the winner flips (lower-ranked
    # strategies keep a stable order), so this is the practically
    # decisive statistic for the "no universal best" claim.
    tops = [v["best"] for v in winners.values()]
    n_pairs = len(tops) * (len(tops) - 1) / 2
    agree = sum(1 for i in range(len(tops)) for j in range(i + 1, len(tops))
                if tops[i] == tops[j])
    top1_agreement = agree / n_pairs if n_pairs else float("nan")

    print("\n=== Task-dependence analysis ===")
    print("Strategy ranks per task (1 = best):")
    print(ranks.to_string())
    print(f"\nKendall's W across tasks: {w:.3f} "
          f"(1 = universal ordering, 0 = task-idiosyncratic)")
    print(f"Top-1 agreement across task pairs: {top1_agreement:.2f} "
          f"(1 = same winner everywhere)")
    print(f"Strategy x task interaction: SS={inter['interaction_ss']:.3f}, "
          f"permutation p={inter['p_value']:.4f}")
    for t, info in winners.items():
        print(f"  {t}: best={info['best']} "
              f"(+{info['margin_over_2nd']:.3f} over runner-up)")
    return {"kendalls_w": w, "top1_agreement": top1_agreement,
            "interaction": inter,
            "ranks": ranks.to_dict(), "winners": winners}
