"""
LLM-as-Judge Scoring Strategies Experiment

Simple script to test: which scoring strategy produces best downstream performance?

Usage:
    python experiment.py --num_samples 1000 --retention 0.25
"""

import json
import random
import re
import argparse
import gc
from tqdm import tqdm
import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# CONFIGURATION
# ============================================================================

# Options:
# - Llama: "meta-llama/Llama-3.1-8B-Instruct"
# - Qwen3: "Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"
# - Mistral: "mistralai/Mistral-7B-Instruct-v0.3"

JUDGE_MODEL = "Qwen/Qwen3-8B"
BASE_MODEL = "Qwen/Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA target modules (architecture-specific)
# - Llama/Mistral: ["q_proj", "k_proj", "v_proj", "o_proj"]
# - Qwen: ["q_proj", "k_proj", "v_proj", "o_proj"] (same)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================================================================
# SCORING PROMPTS
# ============================================================================

# Note: For Qwen3, thinking output is stripped in generate() function

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
    """Generate response from model using proper chat template."""
    
    # Use chat template for instruction-tuned models
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template (works for Llama, Qwen, Mistral, etc.)
    # enable_thinking=False disables Qwen3's thinking mode for faster inference
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Qwen3 specific - ignored by other models
        )
    except TypeError:
        # Fallback for models that don't support enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
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
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Handle Qwen3 thinking mode: strip <think>...</think> tags if still present
    if "<think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    return response


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
    return generate(model, tokenizer, instruction, max_tokens=256)


def score_distributional(model, tokenizer, samples):
    """
    Distributional scoring: Extract probability distribution over quality levels.
    
    Instead of just taking argmax, we:
    1. Get logits for quality tokens (1-5)
    2. Compute softmax distribution
    3. Use expected value as score
    4. Optionally weight by confidence (1 - entropy)
    
    Inspired by "Distributional LLM-as-a-Judge" (Chen et al., NeurIPS 2025)
    """
    print("\nScoring with DISTRIBUTIONAL strategy...")
    scores = []
    
    # Tokens for 1-5 scale
    quality_tokens = ["1", "2", "3", "4", "5"]
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in quality_tokens]
    
    prompt_template = """Rate this response quality from 1-5. Output only the number.

Instruction: {instruction}

Response: {response}

Rating:"""
    
    for sample in tqdm(samples):
        prompt = prompt_template.format(**sample)
        
        # Use chat template with thinking disabled
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the next token position
            next_token_logits = outputs.logits[0, -1, :]
            
            # Extract logits for quality tokens only
            quality_logits = next_token_logits[token_ids]
            
            # Softmax to get distribution
            probs = torch.softmax(quality_logits, dim=0).cpu().numpy()
            
            # Expected value: sum(p_i * i) for i in 1..5
            values = np.array([1, 2, 3, 4, 5])
            expected = np.sum(probs * values)
            
            # Entropy (uncertainty measure)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(5)  # uniform distribution
            confidence = 1 - (entropy / max_entropy)
            
            # Score = expected value, normalized to 0-1
            # Could also weight by confidence: score * confidence
            score = (expected - 1) / 4
            
        scores.append(score)
    
    return scores


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
        target_modules=LORA_TARGET_MODULES
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare data using chat template
    train_texts = []
    for s in samples:
        messages = [
            {"role": "user", "content": s['instruction']},
            {"role": "assistant", "content": s['response']}
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        train_texts.append(text)
    
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
                max_length=512,  # reduced from 1024
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
    
    # Clean up
    del model, optimizer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_name


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_perplexity(model, tokenizer, test_samples):
    """Evaluate perplexity on held-out samples."""
    total_loss = 0
    
    for s in tqdm(test_samples, desc="Perplexity"):
        messages = [
            {"role": "user", "content": s['instruction']},
            {"role": "assistant", "content": s['response']}
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        with torch.no_grad():
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(test_samples)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def evaluate_quality(model, tokenizer, test_samples, judge_model=None, judge_tokenizer=None, num_samples=50):
    """
    Evaluate response quality using LLM-as-judge on held-out prompts.
    
    This is a fairer evaluation than MMLU since we're testing instruction-following
    on the same distribution we trained on.
    """
    if judge_model is None:
        print("  No judge model provided, skipping quality eval...")
        return None
    
    num_samples = min(num_samples, len(test_samples))
    scores = []
    
    judge_prompt = """Rate this response from 1-5 (1=very bad, 5=excellent).

Instruction: {instruction}

Response: {response}

Consider: helpfulness, accuracy, relevance, and clarity.
Reply with only the number (1-5).

Rating:"""
    
    for i in tqdm(range(num_samples), desc="Quality eval"):
        sample = test_samples[i]
        
        # Generate response from finetuned model
        try:
            messages = [{"role": "user", "content": sample["instruction"]}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # Strip thinking tags if present
            if "<think>" in response:
                parts = response.split("</think>")
                if len(parts) > 1:
                    response = parts[-1].strip()
            
        except Exception as e:
            print(f"  Error generating response: {e}")
            continue
        
        # Judge the response
        prompt = judge_prompt.format(instruction=sample["instruction"], response=response)
        judge_output = generate(judge_model, judge_tokenizer, prompt, max_tokens=8)
        
        # Parse score
        score = 3.0  # default
        for char in judge_output:
            if char.isdigit() and 1 <= int(char) <= 5:
                score = float(char)
                break
        
        scores.append(score)
    
    avg_score = np.mean(scores) if scores else 0
    return avg_score


def evaluate(model_path, test_samples, judge_model=None, judge_tokenizer=None, quality_samples=50):
    """Evaluate model on perplexity and judge-based quality."""
    print(f"\nEvaluating {model_path}...")
    
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model fresh each time to avoid adapter conflicts
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Perplexity on held-out data
    avg_loss, perplexity = evaluate_perplexity(model, tokenizer, test_samples)
    print(f"  Perplexity: {perplexity:.2f} (loss: {avg_loss:.4f})")
    
    # Judge-based quality score
    quality = evaluate_quality(model, tokenizer, test_samples, judge_model, judge_tokenizer, quality_samples)
    if quality is not None:
        print(f"  Quality Score: {quality:.2f}/5.0")
    
    # Clean up
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "loss": avg_loss, 
        "perplexity": perplexity,
        "quality": quality
    }


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
        "distributional": lambda m, t, s: score_distributional(m, t, s),
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
        print(f"{name:15s}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
              f"unique={len(set(scores))}")
    
    # Free judge model during training
    del judge_model, judge_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Judge model unloaded for training phase.")
    
    # Train all models first
    model_paths = {}
    
    for strategy_name, scores in all_scores.items():
        print("\n" + "=" * 60)
        print(f"TRAINING: {strategy_name}")
        print("=" * 60)
        
        # Select top samples
        selected = select_top_k(train_samples, scores, args.retention)
        
        # Finetune
        model_path = f"model_{strategy_name}_ret{int(args.retention*100)}"
        finetune(selected, model_path, num_epochs=args.epochs)
        model_paths[strategy_name] = model_path
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Reload judge for evaluation
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    judge_model, judge_tokenizer = load_judge_model()
    
    # Evaluate all models
    results = {}
    
    for strategy_name, model_path in model_paths.items():
        print(f"\n--- {strategy_name} ---")
        metrics = evaluate(
            model_path, 
            test_samples, 
            judge_model=judge_model, 
            judge_tokenizer=judge_tokenizer,
            quality_samples=args.quality_samples
        )
        results[strategy_name] = metrics
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Clean up judge
    del judge_model, judge_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Loss':<10} {'Perplexity':<12} {'Quality':<10}")
    print("-" * 47)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["loss"]):
        quality_str = f"{metrics['quality']:.2f}/5" if metrics.get('quality') is not None else "N/A"
        print(f"{name:<15} {metrics['loss']:<10.4f} {metrics['perplexity']:<12.2f} {quality_str:<10}")
    
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
    parser.add_argument("--quality_samples", type=int, default=50, help="Number of samples for judge-based quality eval")
    args = parser.parse_args()
    
    main(args)
