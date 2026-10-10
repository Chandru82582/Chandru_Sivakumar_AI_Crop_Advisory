"""Ragas Integration Adapter for Agricultural Advisory Evaluation.

Provides integration with Ragas RAG Triad evaluation pipeline:
- Maps agricultural benchmark test cases into Ragas evaluation dataset
- Evaluates Context Precision, Context Recall, Faithfulness, and Answer Relevance
"""

from typing import Any, Optional

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def prepare_ragas_dataset(case_results: list[dict[str, Any]]) -> Any:
    """Transform batch evaluation results into a HuggingFace Dataset for Ragas."""
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "ragas is not installed. Install via: pip install ragas datasets"
        )

    records = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }

    for res in case_results:
        records["question"].append(res.get("query", ""))
        records["answer"].append(res.get("answer", ""))
        
        # Ground truth
        gt = res.get("ground_truth_facts", [])
        records["ground_truth"].append(" \n".join(gt) if gt else "N/A")

        # Context
        ctx = res.get("context", "")
        records["contexts"].append([ctx] if ctx else [""])

    return Dataset.from_dict(records)


def run_ragas_evaluation(dataset: Any, llm: Any = None) -> dict[str, Any]:
    """Execute Ragas benchmark evaluation across RAG Triad metrics."""
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed.")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm
    )

    return dict(result)
