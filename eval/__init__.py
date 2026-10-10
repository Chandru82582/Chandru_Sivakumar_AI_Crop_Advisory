"""Crop Advisory AI Model Evaluation Package.

Provides DeepEval and Ragas evaluation suites, golden test benchmarks,
domain metrics, and custom judge rubrics for the Tamil Nadu Crop Advisory System.
"""

from eval.eval_metrics import (
    AgronomicSafetyEvaluator,
    FarmerReadabilityEvaluator,
    SQLGroundingEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextPrecisionEvaluator
)
from eval.evaluator import AgriculturalModelEvaluator

__all__ = [
    "AgronomicSafetyEvaluator",
    "FarmerReadabilityEvaluator",
    "SQLGroundingEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextPrecisionEvaluator",
    "AgriculturalModelEvaluator"
]
