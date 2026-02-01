"""
LLM-as-Judge Scoring Strategies Module

Implements different scoring approaches:
1. Binary (keep/reject)
2. Likert (1-5 scale)
3. Numeric (0-100 continuous)
4. Anchored (pairwise comparison with reference)

Each strategy uses a different prompt template and parsing logic.
"""

import re
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result from scoring a single sample."""
    sample_id: str
    raw_score: float
    normalized_score: float  # Always 0-1 for comparability
    raw_output: str
    strategy: str
    metadata: Dict[str, Any]


class BaseScoringStrategy(ABC):
    """Abstract base class for scoring strategies."""
    
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Load model and tokenizer
        dtype = getattr(torch, torch_dtype)
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @abstractmethod
    def get_prompt(self, instruction: str, response: str, **kwargs) -> str:
        """Generate the scoring prompt."""
        pass
    
    @abstractmethod
    def parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        """Parse model output to extract score."""
        pass
    
    @abstractmethod
    def normalize_score(self, raw_score: float) -> float:
        """Normalize score to 0-1 range."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the scoring strategy."""
        pass
    
    def score_sample(
        self, 
        sample_id: str,
        instruction: str, 
        response: str,
        **kwargs
    ) -> ScoringResult:
        """Score a single sample."""
        prompt = self.get_prompt(instruction, response, **kwargs)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Parse score
        raw_score, metadata = self.parse_output(raw_output)
        normalized = self.normalize_score(raw_score)
        
        return ScoringResult(
            sample_id=sample_id,
            raw_score=raw_score,
            normalized_score=normalized,
            raw_output=raw_output,
            strategy=self.strategy_name,
            metadata=metadata
        )
    
    def score_batch(
        self,
        samples: List[Dict[str, str]],
        show_progress: bool = True,
        **kwargs
    ) -> List[ScoringResult]:
        """Score a batch of samples."""
        results = []
        iterator = tqdm(samples, desc=f"Scoring ({self.strategy_name})") if show_progress else samples
        
        for sample in iterator:
            result = self.score_sample(
                sample_id=sample["id"],
                instruction=sample["instruction"],
                response=sample["response"],
                **kwargs
            )
            results.append(result)
        
        return results


class BinaryScoringStrategy(BaseScoringStrategy):
    """Binary scoring: Accept (1) or Reject (0)."""
    
    PROMPT_TEMPLATE = """You are evaluating the quality of an AI assistant's response.

## Instruction
{instruction}

## Response
{response}

## Task
Determine if this response should be ACCEPTED or REJECTED for training a high-quality AI assistant.

Consider:
- Is the response helpful and relevant to the instruction?
- Is the response accurate and factually correct?
- Is the response well-structured and coherent?
- Does the response avoid harmful or inappropriate content?

Respond with exactly one word: ACCEPT or REJECT

Decision:"""
    
    @property
    def strategy_name(self) -> str:
        return "binary"
    
    def get_prompt(self, instruction: str, response: str, **kwargs) -> str:
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
    
    def parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        output_lower = output.strip().lower()
        
        if "accept" in output_lower:
            return 1.0, {"decision": "accept"}
        elif "reject" in output_lower:
            return 0.0, {"decision": "reject"}
        else:
            # Ambiguous - default to middle
            return 0.5, {"decision": "ambiguous", "raw": output}
    
    def normalize_score(self, raw_score: float) -> float:
        return raw_score  # Already 0-1


class LikertScoringStrategy(BaseScoringStrategy):
    """Likert scale scoring: 1-5."""
    
    PROMPT_TEMPLATE = """You are evaluating the quality of an AI assistant's response.

## Instruction
{instruction}

## Response
{response}

## Task
Rate this response on a scale of 1-5:

1 - Very Poor: Response is incorrect, harmful, or completely unhelpful
2 - Poor: Response has significant issues or is mostly unhelpful
3 - Acceptable: Response is adequate but has room for improvement
4 - Good: Response is helpful, accurate, and well-structured
5 - Excellent: Response is exceptionally helpful, comprehensive, and well-crafted

Provide your rating as a single number (1-5).

Rating:"""
    
    @property
    def strategy_name(self) -> str:
        return "likert_5"
    
    def get_prompt(self, instruction: str, response: str, **kwargs) -> str:
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
    
    def parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        # Extract first number from output
        numbers = re.findall(r'\b([1-5])\b', output)
        
        if numbers:
            score = int(numbers[0])
            return float(score), {"parsed_score": score}
        else:
            # Try to find any digit
            digits = re.findall(r'\d+', output)
            if digits:
                score = min(max(int(digits[0]), 1), 5)
                return float(score), {"parsed_score": score, "fallback": True}
            
            # Default to middle
            return 3.0, {"parsed_score": 3, "failed_parse": True, "raw": output}
    
    def normalize_score(self, raw_score: float) -> float:
        # Map 1-5 to 0-1
        return (raw_score - 1) / 4


class NumericScoringStrategy(BaseScoringStrategy):
    """Numeric scoring: 0-100 scale."""
    
    PROMPT_TEMPLATE = """You are evaluating the quality of an AI assistant's response.

## Instruction
{instruction}

## Response
{response}

## Task
Rate this response on a scale of 0-100, where:

0-20: Very poor quality - incorrect, harmful, or unhelpful
21-40: Below average - significant issues present
41-60: Average - acceptable but unremarkable
61-80: Good - helpful, accurate, and well-structured
81-100: Excellent - exceptional quality, comprehensive and insightful

Consider these criteria:
- Relevance to the instruction
- Accuracy and correctness
- Clarity and coherence
- Completeness
- Helpfulness

Provide your rating as a single integer from 0-100.

Score:"""
    
    @property
    def strategy_name(self) -> str:
        return "numeric_100"
    
    def get_prompt(self, instruction: str, response: str, **kwargs) -> str:
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
    
    def parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        # Extract numbers from output
        numbers = re.findall(r'\b(\d{1,3})\b', output)
        
        if numbers:
            # Take the first number in valid range
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= 100:
                    return float(num), {"parsed_score": num}
            
            # If no valid range, clamp the first number
            score = min(max(int(numbers[0]), 0), 100)
            return float(score), {"parsed_score": score, "clamped": True}
        
        # Default to middle
        return 50.0, {"parsed_score": 50, "failed_parse": True, "raw": output}
    
    def normalize_score(self, raw_score: float) -> float:
        return raw_score / 100


class AnchoredScoringStrategy(BaseScoringStrategy):
    """
    Anchored pairwise comparison scoring.
    
    Compares target response against a fixed anchor response,
    following RAEE methodology for improved calibration.
    """
    
    PROMPT_TEMPLATE = """You are comparing two AI assistant responses to the same instruction.

## Instruction
{instruction}

## Response A (Reference)
{anchor_response}

## Response B (Target)
{target_response}

## Task
Compare Response B to Response A and rate their relative quality:

-2: Response B is significantly worse than Response A
-1: Response B is slightly worse than Response A
 0: Response B is about equal to Response A
+1: Response B is slightly better than Response A
+2: Response B is significantly better than Response A

Consider: accuracy, helpfulness, clarity, and completeness.

Provide your rating as a single integer from -2 to +2.

Comparison Rating:"""
    
    def __init__(
        self,
        model_name: str,
        anchor_responses: Optional[Dict[str, str]] = None,
        anchor_strategy: str = "median",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.anchor_responses = anchor_responses or {}
        self.anchor_strategy = anchor_strategy
    
    @property
    def strategy_name(self) -> str:
        return "anchored"
    
    def set_anchors(self, anchors: Dict[str, str]):
        """Set anchor responses for comparison."""
        self.anchor_responses = anchors
    
    def get_prompt(
        self, 
        instruction: str, 
        response: str,
        anchor_response: Optional[str] = None,
        **kwargs
    ) -> str:
        # Use provided anchor or lookup
        if anchor_response is None:
            anchor_response = self.anchor_responses.get(
                instruction,
                "This is a placeholder anchor response."
            )
        
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            anchor_response=anchor_response,
            target_response=response
        )
    
    def parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        # Look for -2, -1, 0, +1, +2 patterns
        patterns = [
            r'(-?2)',
            r'(-?1)',
            r'\b(0)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                score = int(matches[0])
                if -2 <= score <= 2:
                    return float(score), {"parsed_score": score}
        
        # Default to tie
        return 0.0, {"parsed_score": 0, "failed_parse": True, "raw": output}
    
    def normalize_score(self, raw_score: float) -> float:
        # Map -2 to +2 → 0 to 1
        return (raw_score + 2) / 4
    
    def score_sample(
        self,
        sample_id: str,
        instruction: str,
        response: str,
        anchor_response: Optional[str] = None,
        **kwargs
    ) -> ScoringResult:
        """Override to pass anchor_response."""
        prompt = self.get_prompt(
            instruction, 
            response, 
            anchor_response=anchor_response
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        raw_score, metadata = self.parse_output(raw_output)
        normalized = self.normalize_score(raw_score)
        
        return ScoringResult(
            sample_id=sample_id,
            raw_score=raw_score,
            normalized_score=normalized,
            raw_output=raw_output,
            strategy=self.strategy_name,
            metadata=metadata
        )


def create_scoring_strategy(
    strategy_name: str,
    model_name: str,
    **kwargs
) -> BaseScoringStrategy:
    """Factory function to create scoring strategies."""
    strategies = {
        "binary": BinaryScoringStrategy,
        "likert_5": LikertScoringStrategy,
        "numeric_100": NumericScoringStrategy,
        "anchored": AnchoredScoringStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](model_name, **kwargs)


class ScoringAnalyzer:
    """Analyze scoring results to understand strategy characteristics."""
    
    @staticmethod
    def compute_statistics(results: List[ScoringResult]) -> Dict[str, Any]:
        """Compute statistics for a set of scoring results."""
        raw_scores = [r.raw_score for r in results]
        normalized_scores = [r.normalized_score for r in results]
        
        # Compute tie rate (for discrete strategies)
        unique_scores = len(set(raw_scores))
        tie_rate = 1 - (unique_scores / len(raw_scores)) if raw_scores else 0
        
        # Score distribution statistics
        stats = {
            "strategy": results[0].strategy if results else "unknown",
            "n_samples": len(results),
            "raw_score": {
                "mean": np.mean(raw_scores),
                "std": np.std(raw_scores),
                "min": np.min(raw_scores),
                "max": np.max(raw_scores),
                "median": np.median(raw_scores)
            },
            "normalized_score": {
                "mean": np.mean(normalized_scores),
                "std": np.std(normalized_scores),
                "min": np.min(normalized_scores),
                "max": np.max(normalized_scores),
                "median": np.median(normalized_scores)
            },
            "unique_values": unique_scores,
            "tie_rate": tie_rate,
            "effective_dynamic_range": np.percentile(raw_scores, 95) - np.percentile(raw_scores, 5)
        }
        
        # Parse success rate
        failed_parses = sum(1 for r in results if r.metadata.get("failed_parse", False))
        stats["parse_success_rate"] = 1 - (failed_parses / len(results)) if results else 0
        
        return stats
    
    @staticmethod
    def compute_score_distribution(
        results: List[ScoringResult],
        bins: int = 20
    ) -> Dict[str, np.ndarray]:
        """Compute histogram of score distribution."""
        raw_scores = [r.raw_score for r in results]
        normalized_scores = [r.normalized_score for r in results]
        
        raw_hist, raw_edges = np.histogram(raw_scores, bins=bins)
        norm_hist, norm_edges = np.histogram(normalized_scores, bins=bins)
        
        return {
            "raw_histogram": raw_hist,
            "raw_bin_edges": raw_edges,
            "normalized_histogram": norm_hist,
            "normalized_bin_edges": norm_edges
        }


if __name__ == "__main__":
    # Test with a small example (requires model)
    print("Scoring strategies module loaded successfully")
    print(f"Available strategies: binary, likert_5, numeric_100, anchored")
