#!/usr/bin/env python
"""Stage-based orchestrator.

Stages are independent so they can be scheduled on different GPUs:

  python run.py --config configs/main.yaml --stage prepare
  python run.py --config configs/main.yaml --stage score            # A40
  python run.py --config configs/main.yaml --stage curate
  python run.py --config configs/main.yaml --stage train --cell binary_ret25_seed42   # 4090s, parallel
  python run.py --config configs/main.yaml --stage train             # or all cells serially
  python run.py --config configs/main.yaml --stage generate
  python run.py --config configs/main.yaml --stage judge_eval        # A40
  python run.py --config configs/main.yaml --stage analyze

Scoring happens ONCE; every retention rate reuses the cached scores.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_ultrafeedback, save_candidates, load_candidates
from src.scoring import STRATEGIES, JudgeBackend, score_diagnostics
from src.curation import (curate, curate_random, curate_length,
                          selection_overlap_matrix, ref_ranking_correlation)


def get_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def out_dir(cfg) -> Path:
    p = Path(cfg["experiment"]["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def all_cells(cfg):
    strategies = [s["name"] for s in cfg["strategies"]]
    strategies += cfg["curation"]["baselines"]
    cells = []
    for strat, ret, seed in itertools.product(
            strategies, cfg["curation"]["retention_rates"],
            cfg["experiment"]["seeds"]):
        if strat == "full" and ret != cfg["curation"]["retention_rates"][0]:
            continue   # full-data baseline runs once per seed
        cells.append(f"{strat}_ret{int(ret*100)}_seed{seed}")
    return cells


# --------------------------------------------------------------------------

def stage_prepare(cfg):
    d = cfg["data"]
    cands, eval_prompts = load_ultrafeedback(
        d["num_prompts"], d["num_eval_prompts"],
        d["min_response_chars"], d["max_response_chars"], d["shuffle_seed"])
    o = out_dir(cfg)
    save_candidates(cands, o / "candidates.jsonl")
    (o / "eval_prompts.json").write_text(json.dumps(eval_prompts, indent=2))
    print(f"prepared {len(cands)} candidates, {len(eval_prompts)} eval prompts")


def stage_score(cfg):
    o = out_dir(cfg)
    cands = load_candidates(o / "candidates.jsonl")
    j = cfg["judge"]
    backend = JudgeBackend(j["model"], j["backend"], j["max_new_tokens"],
                           j["max_model_len"], j["tensor_parallel_size"],
                           j["gpu_memory_utilization"],
                           j.get("enable_thinking", False))
    diagnostics = {}
    for s in cfg["strategies"]:
        cls = STRATEGIES[s["name"]]
        kwargs = {k: v for k, v in s.items() if k != "name"}
        strat = cls(**kwargs)
        recs = strat.score(backend, cands, j["batch_size"])
        scores = {r.sample_id: r.normalized for r in recs
                  if r.normalized is not None}
        (o / f"scores_{s['name']}.json").write_text(json.dumps(scores))
        diagnostics[s["name"]] = score_diagnostics(recs)
        diagnostics[s["name"]]["ref_correlation"] = ref_ranking_correlation(
            cands, scores)
        print(f"{s['name']}: {diagnostics[s['name']]}")
    (o / "score_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def stage_curate(cfg):
    o = out_dir(cfg)
    cands = load_candidates(o / "candidates.jsonl")
    results = []
    for s in cfg["strategies"]:
        scores = json.loads((o / f"scores_{s['name']}.json").read_text())
        for ret in cfg["curation"]["retention_rates"]:
            r = curate(cands, scores, ret, seed=cfg["data"]["shuffle_seed"])
            r.strategy = s["name"]
            results.append(r)
            (o / f"subset_{s['name']}_ret{int(ret*100)}.json").write_text(
                json.dumps({"sample_ids": r.sample_ids,
                            "n_tiebroken": r.n_tiebroken}))
    for ret in cfg["curation"]["retention_rates"]:
        for seed in cfg["experiment"]["seeds"]:
            r = curate_random(cands, ret, seed)
            (o / f"subset_random_ret{int(ret*100)}_seed{seed}.json").write_text(
                json.dumps({"sample_ids": r.sample_ids}))
        rl = curate_length(cands, ret)
        (o / f"subset_length_ret{int(ret*100)}.json").write_text(
            json.dumps({"sample_ids": rl.sample_ids}))
        results.append(rl)
    overlap = selection_overlap_matrix(results)
    (o / "selection_overlap.json").write_text(json.dumps(
        {f"{a}|{b}|{r}": v for (a, b, r), v in overlap.items()}, indent=2))
    print("curation complete; overlap matrix written")


def stage_train(cfg, cell: str | None):
    from src.training import train_cell
    o = out_dir(cfg)
    cands = load_candidates(o / "candidates.jsonl")
    index = {c.sample_id: c for c in cands}
    cells = [cell] if cell else all_cells(cfg)
    for c in cells:
        strat, ret_s, seed_s = c.rsplit("_", 2)
        ret, seed = int(ret_s[3:]) / 100, int(seed_s[4:])
        run_dir = o / "runs" / c
        if (run_dir / "adapter").exists():
            print(f"skip {c} (exists)"); continue
        if strat == "full":
            ids = [x.sample_id for x in cands]
        elif strat == "random":
            ids = json.loads((o / f"subset_random_ret{int(ret*100)}_seed{seed}.json"
                              ).read_text())["sample_ids"]
        else:
            ids = json.loads((o / f"subset_{strat}_ret{int(ret*100)}.json"
                              ).read_text())["sample_ids"]
        print(f"training {c}: {len(ids)} samples")
        train_cell(index, ids, cfg["training"], run_dir, seed)


def stage_generate(cfg, cell: str | None):
    from src.evaluation import generate_responses
    o = out_dir(cfg)
    eval_prompts = json.loads((o / "eval_prompts.json").read_text())
    cells = [cell] if cell else all_cells(cfg)
    for c in cells:
        run_dir = o / "runs" / c
        adapter = run_dir / "adapter"
        outp = run_dir / "responses.json"
        if not adapter.exists() or outp.exists():
            continue
        print(f"generating {c}")
        resp = generate_responses(str(adapter), cfg["training"]["base_model"],
                                  eval_prompts)
        outp.write_text(json.dumps(resp))


def stage_judge_eval(cfg):
    from src.evaluation import judged_win_rate, save_eval
    o = out_dir(cfg)
    eval_prompts = json.loads((o / "eval_prompts.json").read_text())
    e = cfg["evaluation"]
    backend = JudgeBackend(e["eval_judge_model"], cfg["judge"]["backend"],
                           max_new_tokens=16)
    for c in all_cells(cfg):
        run_dir = o / "runs" / c
        rp, ep = run_dir / "responses.json", run_dir / "eval.json"
        if not rp.exists() or ep.exists():
            continue
        responses = json.loads(rp.read_text())
        result = judged_win_rate(backend, eval_prompts, responses)
        save_eval(result, ep)
        print(f"{c}: win_rate={result['win_rate']:.3f}")


def stage_analyze(cfg):
    from src.analysis import analyze, curation_curve_auc
    o = out_dir(cfg)
    strategies = ([s["name"] for s in cfg["strategies"]]
                  + cfg["curation"]["baselines"])
    df = analyze(o / "runs", cfg["curation"]["retention_rates"],
                 strategies, cfg["experiment"]["seeds"])
    if not df.empty:
        df.to_csv(o / "results.csv", index=False)
        auc = curation_curve_auc(df)
        print("\n=== Curation-curve AUC ===")
        print(auc.to_string(index=False))
        auc.to_csv(o / "curation_auc.csv", index=False)


STAGES = {"prepare": stage_prepare, "score": stage_score,
          "curate": stage_curate, "analyze": stage_analyze,
          "judge_eval": stage_judge_eval}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES) + ["train", "generate"])
    ap.add_argument("--cell", default=None,
                    help="single cell id for train/generate (GPU parallelism)")
    args = ap.parse_args()
    cfg = get_cfg(args.config)
    if args.stage == "train":
        stage_train(cfg, args.cell)
    elif args.stage == "generate":
        stage_generate(cfg, args.cell)
    else:
        STAGES[args.stage](cfg)
