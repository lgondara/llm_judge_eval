"""
LLM-as-Judge Scoring Strategies Experiment

Simple script to test: which scoring strategy produces best downstream performance?

Usage:
    python experiment.py --num_samples 1000 --retention 0.25
"""

import json
import random
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# CONFIGURATION
# ============================================================================

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# SCORING PROMPTS
# ============================================================================

BINARY_PROMPT = """Rate this response. Reply with only ACCEPT or REJECT.

Instruction: {instruction}

Response: {response}

Decision:"""

LIKERT_PROMPT = """Rate this response from 1-5 (1=very bad, 5=excellent). Reply with only the number.

Instruction: {instruction}

Response: {response}

Rating:"""

NUMERIC_PROMPT = """Rate this response from 0-100. Reply with only the number.

Instruction: {instruction}

Response: {response}

Score:"""

ANCHORED_PROMPT = """Compare Response A and Response B. Which is better?
Reply with only: A++, A+, TIE, B+, or B++

Instruction: {instruction}

Response A (Reference): {anchor}

Response B (Target): {response}

Verdict:"""

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(num_samples=1000, seed=42):
    """Load instruction-response pairs from UltraFeedback."""
    print(f"Loading {num_samples} samples...")
    
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    samples = []
    for item in dataset:
        prompt = item["prompt"]
        # Use the chosen response
        chosen = item.get("chosen", [])
        if len(chosen) >= 2:
            response = chosen[1].get("content", "")
            if prompt and response:
                samples.append({"instruction": prompt, "response": response})
        
        if len(samples) >= num_samples:
            break
    
    random.seed(seed)
    random.shuffle(samples)
    
    print(f"Loaded {len(samples)} samples")
    return samples

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def load_judge_model():
    """Load the judge model."""
    print(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=32):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def score_binary(model, tokenizer, samples):
    """Score samples with binary (accept/reject) strategy."""
    print("\nScoring with BINARY strategy...")
    scores = []
    
    for sample in tqdm(samples):
        prompt = BINARY_PROMPT.format(**sample)
        output = generate(model, tokenizer, prompt).lower()
        
        if "accept" in output:
            score = 1.0
        elif "reject" in output:
            score = 0.0
        else:
            score = 0.5  # ambiguous
        
        scores.append(score)
    
    return scores


def score_likert(model, tokenizer, samples):
    """Score samples with Likert (1-5) strategy."""
    print("\nScoring with LIKERT strategy...")
    scores = []
    
    for sample in tqdm(samples):
        prompt = LIKERT_PROMPT.format(**sample)
        output = generate(model, tokenizer, prompt)
        
        # Extract number
        score = 3.0  # default
        for char in output:
            if char.isdigit() and 1 <= int(char) <= 5:
                score = float(char)
                break
        
        # Normalize to 0-1
        scores.append((score - 1) / 4)
    
    return scores


def score_numeric(model, tokenizer, samples):
    """Score samples with numeric (0-100) strategy."""
    print("\nScoring with NUMERIC strategy...")
    scores = []
    
    for sample in tqdm(samples):
        prompt = NUMERIC_PROMPT.format(**sample)
        output = generate(model, tokenizer, prompt)
        
        # Extract number
        score = 50.0  # default
        import re
        numbers = re.findall(r'\d+', output)
        if numbers:
            score = min(max(float(numbers[0]), 0), 100)
        
        # Normalize to 0-1
        scores.append(score / 100)
    
    return scores


def score_anchored(model, tokenizer, samples, anchor_response):
    """Score samples by comparing against a fixed anchor response."""
    print("\nScoring with ANCHORED strategy...")
    print(f"  Anchor: {anchor_response[:80]}...")
    scores = []
    
    # Score mapping: A++ (anchor much better) to B++ (target much better)
    score_map = {
        "a++": 0.0,   # anchor much better
        "a+": 0.25,   # anchor slightly better
        "tie": 0.5,   # equal
        "b+": 0.75,   # target slightly better
        "b++": 1.0    # target much better
    }
    
    for sample in tqdm(samples):
        prompt = ANCHORED_PROMPT.format(
            instruction=sample["instruction"],
            response=sample["response"],
            anchor=anchor_response
        )
        output = generate(model, tokenizer, prompt).lower().strip()
        
        # Parse verdict
        score = 0.5  # default to tie
        for key, val in score_map.items():
            if key in output:
                score = val
                break
        
        scores.append(score)
    
    return scores


def get_anchor_response(model, tokenizer, instruction):
    """Generate a reference response to use as anchor."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return generate(model, tokenizer, prompt, max_tokens=256)


# ============================================================================
# DATA CURATION
# ============================================================================

def select_top_k(samples, scores, retention_rate):
    """Select top k% of samples based on scores."""
    paired = list(zip(samples, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    
    k = int(len(paired) * retention_rate)
    selected = [p[0] for p in paired[:k]]
    
    print(f"Selected {len(selected)} samples (top {retention_rate:.0%})")
    return selected


# ============================================================================
# TRAINING
# ============================================================================

def finetune(samples, output_name, num_epochs=1):
    """Finetune base model on selected samples using LoRA."""
    print(f"\nFinetuning on {len(samples)} samples -> {output_name}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare data
    train_texts = []
    for s in samples:
        text = f"### Instruction:\n{s['instruction']}\n\n### Response:\n{s['response']}"
        train_texts.append(text)
    
    # Simple training loop
    from torch.utils.data import DataLoader as TorchDataLoader
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        total_loss = 0
        random.shuffle(train_texts)
        
        for text in tqdm(train_texts, desc=f"Epoch {epoch+1}"):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} loss: {total_loss / len(train_texts):.4f}")
    
    # Save
    model.save_pretrained(output_name)
    tokenizer.save_pretrained(output_name)
    
    return output_name


# ============================================================================
# EVALUATION (simplified - just measures loss on held-out data)
# ============================================================================

def evaluate(model_path, test_samples):
    """Evaluate model on test samples (perplexity)."""
    print(f"\nEvaluating {model_path}...")
    
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    total_loss = 0
    
    for s in tqdm(test_samples):
        text = f"### Instruction:\n{s['instruction']}\n\n### Response:\n{s['response']}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        with torch.no_grad():
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(test_samples)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"  Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return {"loss": avg_loss, "perplexity": perplexity}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main(args):
    print("=" * 60)
    print("LLM-as-Judge Scoring Strategies Experiment")
    print("=" * 60)
    
    # Load data
    all_samples = load_data(args.num_samples, args.seed)
    
    # Split train/test
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Load judge
    judge_model, judge_tokenizer = load_judge_model()
    
    # Score with each strategy
    strategies = {
        "binary": lambda m, t, s: score_binary(m, t, s),
        "likert": lambda m, t, s: score_likert(m, t, s),
        "numeric": lambda m, t, s: score_numeric(m, t, s),
    }
    
    all_scores = {}
    for name, score_fn in strategies.items():
        all_scores[name] = score_fn(judge_model, judge_tokenizer, train_samples)
    
    # Anchored strategy - use a median-quality response as reference
    # First, get numeric scores to find median sample
    numeric_scores = all_scores["numeric"]
    median_idx = sorted(range(len(numeric_scores)), key=lambda i: numeric_scores[i])[len(numeric_scores)//2]
    anchor_response = train_samples[median_idx]["response"]
    all_scores["anchored"] = score_anchored(judge_model, judge_tokenizer, train_samples, anchor_response)
    
    # Also add random baseline
    all_scores["random"] = [random.random() for _ in train_samples]
    
    # Print score statistics
    print("\n" + "=" * 60)
    print("SCORE STATISTICS")
    print("=" * 60)
    for name, scores in all_scores.items():
        import numpy as np
        print(f"{name:10s}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
              f"unique={len(set(scores))}")
    
    # Free judge model memory
    del judge_model, judge_tokenizer
    torch.cuda.empty_cache()
    
    # Train and evaluate for each strategy
    results = {}
    
    for strategy_name, scores in all_scores.items():
        print("\n" + "=" * 60)
        print(f"STRATEGY: {strategy_name}")
        print("=" * 60)
        
        # Select top samples
        selected = select_top_k(train_samples, scores, args.retention)
        
        # Finetune
        model_path = f"model_{strategy_name}_ret{int(args.retention*100)}"
        finetune(selected, model_path, num_epochs=args.epochs)
        
        # Evaluate
        metrics = evaluate(model_path, test_samples)
        results[strategy_name] = metrics
    
    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<12} {'Loss':<10} {'Perplexity':<10}")
    print("-" * 32)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["loss"]):
        print(f"{name:<12} {metrics['loss']:<10.4f} {metrics['perplexity']:<10.2f}")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--retention", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
