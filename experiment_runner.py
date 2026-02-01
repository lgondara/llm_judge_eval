"""
Main Experiment Runner

Orchestrates the full pipeline:
1. Load data
2. Score with each strategy
3. Curate based on retention rates
4. Finetune models
5. Evaluate downstream
6. Analyze results
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import pandas as pd
import numpy as np

from src.data_loader import DataLoader, ConversationSample, prepare_training_data
from src.scoring_strategies import (
    create_scoring_strategy,
    ScoringResult,
    ScoringAnalyzer
)
from src.trainer import TrainingConfig, run_training_experiment
from src.evaluator import evaluate_trained_model, EvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str
    seed: int
    output_dir: str
    
    # Data
    data_source: str
    domain_filter: str
    max_samples: int
    
    # Judge
    judge_model: str
    
    # Scoring strategies to test
    strategies: List[str]
    
    # Curation
    retention_rates: List[float]
    
    # Training
    base_model: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_replicates: int
    
    # Evaluation
    benchmarks: List[str]


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return ExperimentConfig(
        name=config['experiment']['name'],
        seed=config['experiment']['seed'],
        output_dir=config['experiment']['output_dir'],
        data_source=config['data']['source'],
        domain_filter=config['data']['domain_filter'],
        max_samples=config['data']['max_samples'],
        judge_model=config['judge']['model_name'],
        strategies=[s['name'] for s in config['scoring_strategies']],
        retention_rates=config['curation']['retention_rates'],
        base_model=config['training']['base_model'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_replicates=config['training']['num_replicates'],
        benchmarks=[b['name'] for b in config['evaluation']['benchmarks']]
    )


class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.scores: Dict[str, List[ScoringResult]] = {}
        self.training_results: Dict[str, str] = {}  # Maps run_name -> model_path
        self.eval_results: Dict[str, Dict[str, EvaluationResult]] = {}
        
    def run_full_experiment(self):
        """Execute the complete experiment pipeline."""
        logger.info(f"Starting experiment: {self.config.name}")
        start_time = datetime.now()
        
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        samples = self._load_data()
        
        # Step 2: Score with each strategy
        logger.info("Step 2: Scoring with each strategy")
        self._run_scoring(samples)
        
        # Step 3: Run training experiments for each configuration
        logger.info("Step 3: Running training experiments")
        self._run_training_experiments(samples)
        
        # Step 4: Evaluate all trained models
        logger.info("Step 4: Evaluating trained models")
        self._run_evaluations()
        
        # Step 5: Analyze and save results
        logger.info("Step 5: Analyzing results")
        self._analyze_and_save_results()
        
        elapsed = datetime.now() - start_time
        logger.info(f"Experiment complete! Total time: {elapsed}")
    
    def _load_data(self) -> List[ConversationSample]:
        """Load and preprocess data."""
        loader = DataLoader(
            source=self.config.data_source,
            domain_filter=self.config.domain_filter,
            max_samples=self.config.max_samples,
            seed=self.config.seed
        )
        samples = loader.load()
        
        # Save sample IDs for reproducibility
        sample_ids = [s.id for s in samples]
        with open(self.output_dir / "sample_ids.json", 'w') as f:
            json.dump(sample_ids, f)
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _run_scoring(self, samples: List[ConversationSample]):
        """Score all samples with each strategy."""
        
        # Convert samples to dict format
        sample_dicts = [
            {
                "id": s.id,
                "instruction": s.instruction,
                "response": s.response
            }
            for s in samples
        ]
        
        for strategy_name in self.config.strategies:
            logger.info(f"Scoring with strategy: {strategy_name}")
            
            # Create scorer
            scorer = create_scoring_strategy(
                strategy_name=strategy_name,
                model_name=self.config.judge_model
            )
            
            # Score all samples
            results = scorer.score_batch(sample_dicts)
            self.scores[strategy_name] = results
            
            # Save scores
            scores_path = self.output_dir / f"scores_{strategy_name}.json"
            self._save_scores(results, scores_path)
            
            # Compute and log statistics
            stats = ScoringAnalyzer.compute_statistics(results)
            logger.info(f"  {strategy_name} stats: mean={stats['normalized_score']['mean']:.3f}, "
                       f"std={stats['normalized_score']['std']:.3f}, "
                       f"tie_rate={stats['tie_rate']:.3f}")
    
    def _save_scores(self, results: List[ScoringResult], path: Path):
        """Save scoring results to JSON."""
        data = [
            {
                "sample_id": r.sample_id,
                "raw_score": r.raw_score,
                "normalized_score": r.normalized_score,
                "strategy": r.strategy
            }
            for r in results
        ]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _run_training_experiments(self, samples: List[ConversationSample]):
        """Run all training configurations."""
        
        for strategy_name in self.config.strategies:
            scores = self.scores[strategy_name]
            score_dict = {r.sample_id: r.normalized_score for r in scores}
            
            for retention in self.config.retention_rates:
                # Skip 100% for analysis (baseline)
                # Actually, include it as full-data baseline
                
                for replicate in range(self.config.num_replicates):
                    run_name = f"{strategy_name}_ret{int(retention*100)}_rep{replicate}"
                    logger.info(f"Training: {run_name}")
                    
                    # Prepare curated data
                    sample_dicts = [
                        {
                            "id": s.id,
                            "instruction": s.instruction,
                            "response": s.response
                        }
                        for s in samples
                    ]
                    
                    train_data = prepare_training_data(
                        sample_dicts,
                        score_dict,
                        retention,
                        selection_method="percentile"
                    )
                    
                    # Configure training
                    train_config = TrainingConfig(
                        base_model=self.config.base_model,
                        num_epochs=self.config.num_epochs,
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                        seed=self.config.seed + replicate,
                        output_dir=str(self.output_dir / "models")
                    )
                    
                    # Train
                    model_path = run_training_experiment(
                        train_data=train_data,
                        config=train_config,
                        experiment_name=run_name
                    )
                    
                    self.training_results[run_name] = model_path
        
        # Also train random baseline at each retention rate
        self._train_random_baselines(samples)
    
    def _train_random_baselines(self, samples: List[ConversationSample]):
        """Train models with random selection (no curation signal)."""
        import random
        
        for retention in self.config.retention_rates:
            for replicate in range(self.config.num_replicates):
                run_name = f"random_ret{int(retention*100)}_rep{replicate}"
                logger.info(f"Training baseline: {run_name}")
                
                # Random selection
                random.seed(self.config.seed + replicate)
                k = int(len(samples) * retention)
                selected = random.sample(samples, k)
                
                train_data = [
                    {
                        "id": s.id,
                        "instruction": s.instruction,
                        "response": s.response
                    }
                    for s in selected
                ]
                
                train_config = TrainingConfig(
                    base_model=self.config.base_model,
                    num_epochs=self.config.num_epochs,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    seed=self.config.seed + replicate,
                    output_dir=str(self.output_dir / "models")
                )
                
                model_path = run_training_experiment(
                    train_data=train_data,
                    config=train_config,
                    experiment_name=run_name
                )
                
                self.training_results[run_name] = model_path
    
    def _run_evaluations(self):
        """Evaluate all trained models."""
        for run_name, model_path in self.training_results.items():
            logger.info(f"Evaluating: {run_name}")
            
            results = evaluate_trained_model(
                base_model_name=self.config.base_model,
                adapter_path=model_path,
                benchmarks=self.config.benchmarks
            )
            
            self.eval_results[run_name] = results
            
            # Log results
            for benchmark, result in results.items():
                logger.info(f"  {benchmark}: {result.score:.4f}")
    
    def _analyze_and_save_results(self):
        """Analyze results and save comprehensive report."""
        
        # Create results dataframe
        rows = []
        for run_name, results in self.eval_results.items():
            # Parse run name
            parts = run_name.split('_')
            strategy = parts[0]
            retention = int(parts[1].replace('ret', '')) / 100
            replicate = int(parts[2].replace('rep', ''))
            
            for benchmark, result in results.items():
                rows.append({
                    'strategy': strategy,
                    'retention': retention,
                    'replicate': replicate,
                    'benchmark': benchmark,
                    'metric': result.metric,
                    'score': result.score
                })
        
        df = pd.DataFrame(rows)
        
        # Save raw results
        df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Compute aggregate statistics
        agg_results = df.groupby(
            ['strategy', 'retention', 'benchmark']
        ).agg({
            'score': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        agg_results.to_csv(self.output_dir / "results_aggregated.csv")
        
        # Generate summary report
        self._generate_report(df)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_report(self, df: pd.DataFrame):
        """Generate markdown summary report."""
        report = [
            f"# Experiment Report: {self.config.name}",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\n## Configuration",
            f"- Data source: {self.config.data_source}",
            f"- Domain filter: {self.config.domain_filter}",
            f"- Max samples: {self.config.max_samples}",
            f"- Judge model: {self.config.judge_model}",
            f"- Base model: {self.config.base_model}",
            f"- Strategies: {', '.join(self.config.strategies)}",
            f"- Retention rates: {self.config.retention_rates}",
            f"\n## Results Summary"
        ]
        
        # Best configuration per benchmark
        for benchmark in df['benchmark'].unique():
            bench_df = df[df['benchmark'] == benchmark]
            
            # Aggregate by strategy and retention
            agg = bench_df.groupby(['strategy', 'retention'])['score'].mean().reset_index()
            best = agg.loc[agg['score'].idxmax()]
            
            report.append(f"\n### {benchmark}")
            report.append(f"Best: {best['strategy']} @ {best['retention']:.0%} retention = {best['score']:.4f}")
            
            # Table of all results
            pivot = agg.pivot(index='retention', columns='strategy', values='score')
            report.append("\n" + pivot.round(4).to_markdown())
        
        # Key findings
        report.append("\n## Key Findings")
        report.append("\n*Analysis to be completed based on results*")
        
        # Save report
        with open(self.output_dir / "report.md", 'w') as f:
            f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge Scoring Strategies Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "score", "train", "eval", "analyze"],
        default="all",
        help="Which stage to run"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    runner = ExperimentRunner(config)
    
    if args.stage == "all":
        runner.run_full_experiment()
    else:
        logger.info(f"Running stage: {args.stage}")
        # Implement partial runs as needed
        raise NotImplementedError(f"Stage {args.stage} not yet implemented")


if __name__ == "__main__":
    main()
