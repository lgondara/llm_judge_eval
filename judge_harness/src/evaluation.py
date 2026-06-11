"""Downstream evaluation: judged win-rate against fixed references.

Design safeguards against circularity:
  * The eval judge is a DIFFERENT model family from the curation judge.
  * The opponent is a fixed reference response per eval prompt (the
    top GPT-4-rated UltraFeedback completion), identical across all cells.
  * Every pair is judged in both (A,B) and (B,A) orders; a win counts only
    if consistent across orders, otherwise it is a tie (0.5).

The per-prompt outcome in {0, 0.5, 1} is the unit of analysis for the
bootstrap in analysis.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from .scoring import JudgeBackend, _parse_verdict, _PAIRWISE_TEMPLATE


def generate_responses(adapter_path: str, base_model: str,
                       eval_prompts: list[dict], max_new_tokens: int = 512) -> list[str]:
    """Generate eval-prompt responses with the fine-tuned model (greedy)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    outs = []
    for item in eval_prompts:
        messages = [{"role": "user", "content": item["instruction"]}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        if "<think>" in resp and "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        outs.append(resp)

    del model
    import gc, torch as _t
    gc.collect(); _t.cuda.empty_cache()
    return outs


def judged_win_rate(eval_judge: JudgeBackend, eval_prompts: list[dict],
                    model_responses: list[str], batch_size: int = 128) -> dict:
    """Position-swapped pairwise judging vs reference. Returns per-prompt outcomes."""
    fwd, rev = [], []
    for item, resp in zip(eval_prompts, model_responses):
        fwd.append(_PAIRWISE_TEMPLATE.format(
            instruction=item["instruction"],
            response_a=resp, response_b=item["reference_response"]))
        rev.append(_PAIRWISE_TEMPLATE.format(
            instruction=item["instruction"],
            response_a=item["reference_response"], response_b=resp))

    def run(prompts):
        out = []
        for i in range(0, len(prompts), batch_size):
            out.extend(eval_judge.generate(prompts[i:i + batch_size]))
        return out

    fwd_out, rev_out = run(fwd), run(rev)

    outcomes = []
    for f, r in zip(fwd_out, rev_out):
        vf, vr = _parse_verdict(f), _parse_verdict(r)
        # vf > 0 => model wins in forward order; vr < 0 => model wins reversed.
        f_win = vf is not None and vf > 0
        f_loss = vf is not None and vf < 0
        r_win = vr is not None and vr < 0
        r_loss = vr is not None and vr > 0
        if f_win and r_win:
            outcomes.append(1.0)
        elif f_loss and r_loss:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)   # inconsistent or tie or parse failure

    return {"win_rate": sum(outcomes) / len(outcomes),
            "per_prompt": outcomes}


def save_eval(result: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def run_lm_eval_benchmarks(adapter_path: str, base_model: str,
                           tasks: list[str], out_path: str | Path,
                           limit: int | None = 500) -> dict:
    """Judge-free downstream signal via lm-evaluation-harness.

    Returns {task: {"score": float, "per_instance": [...]}} so benchmark
    tasks plug into the same paired-bootstrap machinery as judged tasks.
    Requires: pip install lm-eval
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=base_model, peft=adapter_path,
              dtype="bfloat16", batch_size="auto")
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit,
                                      log_samples=True)
    out = {}
    primary = {"ifeval": "prompt_level_strict_acc,none",
               "gsm8k": "exact_match,strict-match"}
    for task in tasks:
        metrics = results["results"].get(task, {})
        key = primary.get(task)
        score = metrics.get(key)
        if score is None:   # fall back to first accuracy-like metric
            score = next((v for k, v in metrics.items()
                          if isinstance(v, float) and "acc" in k or "match" in k),
                         None)
        samples = results.get("samples", {}).get(task, [])
        per_instance = []
        for s in samples:
            for k in ("acc", "exact_match", "prompt_level_strict_acc"):
                if k in s:
                    per_instance.append(float(s[k]))
                    break
        out[task] = {"score": float(score) if score is not None else None,
                     "per_instance": per_instance}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out
