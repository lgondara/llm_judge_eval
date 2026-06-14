"""LoRA supervised fine-tuning on curated subsets.

One function = one (strategy, retention, seed) cell. Hyperparameters are
fixed across all cells so the only varying factor is the curated subset.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path


def train_cell(candidate_index: dict, sample_ids: list[str], cfg: dict,
               run_dir: str | Path, seed: int) -> str:
    """Fine-tune base model on the given subset; return adapter path."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling, set_seed)

    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(c):
        messages = [{"role": "user", "content": c.instruction},
                    {"role": "assistant", "content": c.response}]
        try:
            return tok.apply_chat_template(messages, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False)

    texts = [fmt(candidate_index[sid]) for sid in sample_ids]

    def tokenize(batch):
        return tok(batch["text"], truncation=True,
                   max_length=cfg["max_seq_len"])

    ds = Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"],
                      lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"])
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=str(run_dir / "ckpt"),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        bf16=cfg.get("bf16", True),
        # Memory-saving knobs (default on; matter on 24GB cards).
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "adamw_torch"),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    if cfg.get("gradient_checkpointing", True):
        # Required so checkpointing produces grads through a frozen base model.
        model.config.use_cache = False
        model.enable_input_require_grads()
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()

    adapter_path = run_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tok.save_pretrained(str(adapter_path))
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"n_samples": len(sample_ids), "seed": seed}, f)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return str(adapter_path)
