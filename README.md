# LLM-as-Judge Scoring Strategies for Data Curation

## Scoring Strategies

| Strategy | Type | Scale | Description |
|----------|------|-------|-------------|
| Binary | Pointwise | {0, 1} | Accept/Reject decision |
| Likert | Pointwise | 1-5 | Ordinal quality rating |
| Numeric | Pointwise | 0-100 | Fine-grained continuous score |
| Anchored | Pairwise | -2 to +2 | Comparison against reference |

## Project Structure

```
llm_judge_curation/
├── configs/
│   └── experiment_config.yaml    # Main experiment configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── scoring_strategies.py     # Scoring strategy implementations
│   ├── trainer.py                # LoRA finetuning
│   ├── evaluator.py              # Downstream benchmark evaluation
│   └── experiment_runner.py      # Main orchestration
├── scripts/
│   └── quick_start.py            # Quick testing script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test Scoring Strategies (No Training)

```bash
python scripts/quick_start.py \
    --data-source HuggingFaceH4/ultrafeedback_binarized \
    --domain code \
    --samples 500 \
    --score-limit 100 \
    --judge-model meta-llama/Llama-3.1-8B-Instruct \
    --strategies binary likert_5 numeric_100 \
    --mode both
```

This will:
- Load 500 samples, score the first 100
- Compare score distributions across strategies
- Analyze selection overlap at different retention rates

### 2. Full Experiment

```bash
python -m src.experiment_runner --config configs/experiment_config.yaml
```

This runs the complete pipeline:
1. Load and filter data
2. Score with all strategies
3. Curate data at multiple retention rates
4. Finetune models (LoRA)
5. Evaluate on benchmarks
6. Generate analysis report

## Configuration

Edit `configs/experiment_config.yaml`:

```yaml
experiment:
  name: "my_experiment"
  seed: 42

data:
  source: "lmsys/lmsys-chat-1m"
  domain_filter: "code"
  max_samples: 50000

judge:
  model_name: "meta-llama/Llama-3.1-70B-Instruct"

curation:
  retention_rates: [0.10, 0.25, 0.50, 0.75, 1.00]

training:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  num_epochs: 3
  num_replicates: 3

evaluation:
  benchmarks:
    - name: "humaneval"
    - name: "mbpp"
```

## Expected Outputs

```
outputs/my_experiment/
├── sample_ids.json           # For reproducibility
├── scores_binary.json        # Raw scores per strategy
├── scores_likert_5.json
├── scores_numeric_100.json
├── scores_anchored.json
├── models/                   # Trained LoRA adapters
│   ├── binary_ret25_rep0/
│   ├── likert_5_ret25_rep0/
│   └── ...
├── results.csv               # All evaluation results
├── results_aggregated.csv    # Aggregated statistics
└── report.md                 # Summary analysis
```

## Key Metrics

### Score Distribution Analysis
- **Tie rate**: Fraction of samples with identical scores (problematic for ranking)
- **Effective dynamic range**: 95th - 5th percentile (usable score spread)
- **Parse success rate**: How often the judge output could be parsed

### Downstream Utility
- **Pass@1**: Primary metric for code benchmarks
- **Curation curve**: Performance vs retention rate
- **Optimal retention**: Best performing retention rate per strategy

## License

MIT
