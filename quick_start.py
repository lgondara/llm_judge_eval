#!/usr/bin/env python3
"""
Quick Start Script for LLM-as-Judge Scoring Strategies Experiment

This script runs a smaller-scale version of the experiment for testing
and development purposes.

Usage:
    python scripts/quick_start.py --samples 1000 --strategies binary likert_5
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, prepare_training_data
from src.scoring_strategies import create_scoring_strategy, ScoringAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_scoring_only(args):
    """
    Run only the scoring phase for quick analysis.
    
    This is useful for:
    - Testing different scoring strategies
    - Analyzing score distributions
    - Debugging scoring prompts
    """
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading {args.samples} samples from {args.data_source}")
    
    loader = DataLoader(
        source=args.data_source,
        domain_filter=args.domain,
        max_samples=args.samples,
        seed=args.seed
    )
    samples = loader.load()
    
    # Convert to dict format
    sample_dicts = [
        {"id": s.id, "instruction": s.instruction, "response": s.response}
        for s in samples
    ]
    
    logger.info(f"Loaded {len(sample_dicts)} samples")
    
    # Score with each strategy
    all_results = {}
    
    for strategy_name in args.strategies:
        logger.info(f"\nScoring with: {strategy_name}")
        
        scorer = create_scoring_strategy(
            strategy_name=strategy_name,
            model_name=args.judge_model,
            max_new_tokens=128,
            temperature=0.0
        )
        
        # Score subset for quick testing
        test_samples = sample_dicts[:args.score_limit] if args.score_limit else sample_dicts
        results = scorer.score_batch(test_samples, show_progress=True)
        
        all_results[strategy_name] = results
        
        # Analyze
        stats = ScoringAnalyzer.compute_statistics(results)
        
        logger.info(f"\n{strategy_name} Statistics:")
        logger.info(f"  Samples scored: {stats['n_samples']}")
        logger.info(f"  Parse success rate: {stats['parse_success_rate']:.2%}")
        logger.info(f"  Raw scores: mean={stats['raw_score']['mean']:.2f}, "
                   f"std={stats['raw_score']['std']:.2f}, "
                   f"range=[{stats['raw_score']['min']:.2f}, {stats['raw_score']['max']:.2f}]")
        logger.info(f"  Normalized: mean={stats['normalized_score']['mean']:.3f}, "
                   f"std={stats['normalized_score']['std']:.3f}")
        logger.info(f"  Unique values: {stats['unique_values']}")
        logger.info(f"  Tie rate: {stats['tie_rate']:.2%}")
        logger.info(f"  Effective dynamic range: {stats['effective_dynamic_range']:.2f}")
        
        # Save scores
        scores_data = [
            {
                "sample_id": r.sample_id,
                "raw_score": r.raw_score,
                "normalized_score": r.normalized_score,
                "raw_output": r.raw_output[:200],  # Truncate for readability
                "metadata": r.metadata
            }
            for r in results
        ]
        
        with open(output_dir / f"scores_{strategy_name}.json", 'w') as f:
            json.dump(scores_data, f, indent=2)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_source": args.data_source,
            "domain": args.domain,
            "samples": len(sample_dicts),
            "judge_model": args.judge_model,
            "strategies": args.strategies
        },
        "statistics": {
            name: ScoringAnalyzer.compute_statistics(results)
            for name, results in all_results.items()
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return all_results


def run_curation_analysis(args, scores=None):
    """
    Analyze how different strategies perform at data curation.
    
    Without training, we can still analyze:
    - Which samples each strategy selects at different retention rates
    - Overlap between strategies
    - Score correlations
    """
    import numpy as np
    
    output_dir = Path(args.output_dir)
    
    # Load scores if not provided
    if scores is None:
        scores = {}
        for strategy in args.strategies:
            scores_path = output_dir / f"scores_{strategy}.json"
            if scores_path.exists():
                with open(scores_path) as f:
                    data = json.load(f)
                scores[strategy] = data
    
    if not scores:
        logger.error("No scores found. Run scoring first.")
        return
    
    # Analyze selection overlap at different retention rates
    retention_rates = [0.10, 0.25, 0.50, 0.75]
    
    logger.info("\nCuration Analysis")
    logger.info("=" * 60)
    
    for retention in retention_rates:
        logger.info(f"\nRetention Rate: {retention:.0%}")
        
        # Get selected samples for each strategy
        selected = {}
        for strategy, strategy_scores in scores.items():
            if isinstance(strategy_scores, list):
                # From run_scoring_only
                score_list = [(s['sample_id'], s['normalized_score']) for s in strategy_scores]
            else:
                # From ScoringResult objects
                score_list = [(r.sample_id, r.normalized_score) for r in strategy_scores]
            
            score_list.sort(key=lambda x: x[1], reverse=True)
            k = int(len(score_list) * retention)
            selected[strategy] = set(s[0] for s in score_list[:k])
        
        # Compute pairwise overlap
        strategies = list(selected.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                overlap = len(selected[s1] & selected[s2])
                union = len(selected[s1] | selected[s2])
                jaccard = overlap / union if union > 0 else 0
                logger.info(f"  {s1} ∩ {s2}: {overlap}/{len(selected[s1])} ({jaccard:.2%} Jaccard)")
    
    # Score correlations
    logger.info("\nScore Correlations (Spearman)")
    
    # Build score matrix
    sample_ids = None
    score_matrix = {}
    
    for strategy, strategy_scores in scores.items():
        if isinstance(strategy_scores, list):
            score_dict = {s['sample_id']: s['normalized_score'] for s in strategy_scores}
        else:
            score_dict = {r.sample_id: r.normalized_score for r in strategy_scores}
        
        if sample_ids is None:
            sample_ids = list(score_dict.keys())
        
        score_matrix[strategy] = [score_dict.get(sid, 0) for sid in sample_ids]
    
    from scipy.stats import spearmanr
    
    strategies = list(score_matrix.keys())
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            corr, pval = spearmanr(score_matrix[s1], score_matrix[s2])
            logger.info(f"  {s1} vs {s2}: ρ={corr:.3f} (p={pval:.2e})")


def main():
    parser = argparse.ArgumentParser(description="Quick start for scoring experiments")
    
    # Data arguments
    parser.add_argument("--data-source", type=str, 
                       default="HuggingFaceH4/ultrafeedback_binarized",
                       help="HuggingFace dataset path")
    parser.add_argument("--domain", type=str, default="general",
                       choices=["code", "math", "general"],
                       help="Domain filter")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to load")
    parser.add_argument("--score-limit", type=int, default=None,
                       help="Limit number of samples to score (for quick testing)")
    
    # Model arguments
    parser.add_argument("--judge-model", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Judge model for scoring")
    
    # Strategy arguments
    parser.add_argument("--strategies", nargs="+",
                       default=["binary", "likert_5", "numeric_100"],
                       help="Scoring strategies to test")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./quick_start_output",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Mode
    parser.add_argument("--mode", type=str, default="score",
                       choices=["score", "analyze", "both"],
                       help="What to run")
    
    args = parser.parse_args()
    
    if args.mode in ["score", "both"]:
        scores = run_scoring_only(args)
    else:
        scores = None
    
    if args.mode in ["analyze", "both"]:
        run_curation_analysis(args, scores)


if __name__ == "__main__":
    main()
