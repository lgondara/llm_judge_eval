# LLM-Judge Scoring Strategies for Data Curation — Experimental Harness

**Research question:** Is there a universal best judge scoring strategy
(binary, Likert-5, numeric-100, anchored, pairwise tournament) for
fine-tuning data curation — or does the downstream task dictate which
strategy wins?

**Hypothesis (task-contingency):** Coarse formats excel where quality is
near-categorical (verifiable tasks: code, math); fine-grained/comparative
formats excel where quality is graded (open-ended instruction following).
The design is robust to either outcome: a universal winner is actionable
guidance; an interaction refutes the field's implicit universality
assumption.

This harness replaces the monolithic `experiment-N.py` scripts with a
staged, resumable pipeline designed for a hard two-week submission deadline.

## Key design changes vs. previous iterations

| Previous (experiment-5.py)         | This harness                              | Why |
|------------------------------------|-------------------------------------------|-----|
| Synthetic degradation (truncation, fillers) | UltraFeedback: natural variance from 4 real model completions per prompt | Synthetic degradations are length-detectable; reviewers will call it out |
| Single seed                        | 3 fine-tuning seeds per cell + bootstrap CIs, Friedman, Holm | Strategy effects may be smaller than fine-tuning variance |
| Held-out loss as the metric        | Position-swapped judged win-rate vs fixed references | Loss ≠ instruction quality |
| Same judge for curation and eval   | Qwen3-8B curates, Llama-3.1-8B evaluates  | Breaks the circularity critique |
| HF `generate` loop                 | vLLM batched scoring                       | ~20-50× scoring throughput |
| 4 strategies                       | + pairwise tournament (Bradley-Terry)      | The format Landesberg 2026 shows recovers selection signal |

## Pipeline

```
prepare -> score -> curate -> train -> generate -> judge_eval -> analyze
            (A40)            (4090s,            (4090s)  (A40)
                              parallel by cell)
```

```bash
pip install -r requirements.txt
python run.py --config configs/main.yaml --stage prepare
python run.py --config configs/main.yaml --stage score        # once; cached
python run.py --config configs/main.yaml --stage curate
# Parallelize training across GPUs by cell:
CUDA_VISIBLE_DEVICES=0 python run.py --stage train --cell binary_ret25_seed42 &
CUDA_VISIBLE_DEVICES=1 python run.py --stage train --cell likert_5_ret25_seed42 &
# ...or run everything serially:
python run.py --config configs/main.yaml --stage train
python run.py --config configs/main.yaml --stage generate
python run.py --config configs/main.yaml --stage judge_eval
python run.py --config configs/main.yaml --stage analyze
```

All stages are idempotent: completed cells are skipped, so crashed runs
resume where they left off.

## Experiment grid and compute budget

- Data: 4,000 UltraFeedback prompts × ~4 responses = ~16k candidates;
  300 held-out eval prompts.
- Scoring: 5 strategies. Pointwise = 16k calls each; anchored = 16k;
  tournament = 4k × C(4,2) = 24k. Total ≈ 88k short judge calls.
  **vLLM on the A40: roughly 3–6 hours.**
- Cells: 5 strategies × 3 retentions × 3 seeds = 45, plus random
  (3×3=9), length (3×3=9), full (3) → **66 LoRA runs**.
  At ~1k–2k samples/run, Qwen3-4B LoRA ≈ 20–40 min/run on a 4090 →
  **~2–3 days wall-clock across your three GPUs.**
- Generation + judged eval: 66 × 300 prompts ≈ 20k generations + 40k
  judge calls (position-swapped) → **~1 day.**

Total: comfortably under one week of compute, leaving week two for
analysis, writing, and one contingency re-run.

## Statistical analysis (pre-registered)

"No universal best" is a quasi-null claim, so it requires positive
evidence on two fronts (`src/analysis.py:task_dependence_report`):

1. **Within-task differences exist** — per-task Friedman omnibus
   (blocks = eval prompts/instances), then pairwise paired bootstrap
   (10k resamples) with Holm correction.
2. **The ordering is unstable across tasks** — top-1 winner agreement
   across task pairs (the practically decisive statistic), Kendall's W
   ranking concordance, and a permutation test on the strategy x task
   interaction (task-standardized scores, main effects removed,
   strategy labels permuted within task).

Note: Kendall's W can remain high while the *winner* flips (lower-ranked
strategies keep a stable order), so report top-1 agreement and per-task
winner margins alongside W.

Tasks: judged win-rate (per-prompt outcome ∈ {0, 0.5, 1}, seed-averaged),
IFEval prompt-level strict accuracy, GSM8K exact match (per-instance) —
all via the same paired machinery. Mechanism analysis: curation-curve
AUC, tie probability, effective dynamic range, cross-strategy selection
Jaccard, Spearman/Kendall vs GPT-4 reference annotations (reference
scores never enter curation).

## Reviewer-vulnerability checklist (address in the paper)

1. **Single judge model** — if time permits, re-run `score` with one
   alternative judge (config swap) on a subset; otherwise acknowledge in
   Limitations and note the harness makes replication trivial.
2. **LLM-judged downstream eval** — mitigated by cross-family judge +
   position swapping; optionally add IFEval/GSM8K via lm-eval-harness
   (`evaluation.optional_benchmarks`) for a judge-free signal.
3. **Task coverage** — primary design tests strategy x evaluation-task
   interaction on one curation pool; the stronger strategy x domain claim
   needs the optional code pool (`configs/code.yaml`, CodeUltraFeedback +
   HumanEval/MBPP execution-based eval). Run it if week-1 compute
   finishes early; otherwise scope claims to evaluation-task interaction.
4. **Comparison-budget fairness** — tournament uses ≤6 comparisons/prompt
   vs 4 pointwise calls; report cost-normalized results in an appendix.

## Repository layout

```
configs/main.yaml          general pool: UltraFeedback, 3 eval tasks
configs/code.yaml          optional code pool (strategy x domain claim)
run.py                     stage-based orchestrator
src/data.py                UltraFeedback loading (natural quality variance)
src/scoring.py             5 strategies + vLLM backend + diagnostics
src/curation.py            percentile selection, baselines, overlap, ref corr
src/training.py            LoRA SFT (one function per cell)
src/evaluation.py          cross-family judged win-rate, position-swapped
src/analysis.py            bootstrap CIs, Friedman, Holm, curation curves
paper/intro_related_work.tex   Introduction + Related Work (+ bib stubs)
```
