"""Core Agricultural Model Evaluation Engine.

Orchestrates running the golden benchmark test cases against the Crop Advisory AI
components (Farm Assistant, Advisory Orchestrator, RAG Retriever, and SQL Agent)
and evaluates them across all 6 agronomic metrics.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, date

from core.rag_retriever import RAGRetriever
from core.sql_agent import FarmHistoryAgent
from app import get_crop_calendar_events, query_farm_assistant
from eval.eval_metrics import (
    AgronomicSafetyEvaluator,
    FarmerReadabilityEvaluator,
    SQLGroundingEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextPrecisionEvaluator,
)


class AgriculturalModelEvaluator:
    """Evaluates Crop Advisory model responses against domain benchmarks."""

    def __init__(
        self,
        test_cases_path: Optional[str] = None,
        retriever: Optional[RAGRetriever] = None,
        sql_agent: Optional[FarmHistoryAgent] = None,
        llm: Any = None
    ):
        """Initialize evaluator with components and golden benchmark dataset."""
        base_dir = Path(__file__).resolve().parent.parent
        self.test_cases_path = (
            Path(test_cases_path) if test_cases_path
            else base_dir / "eval" / "agri_test_cases.json"
        )

        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.test_cases = self.dataset.get("test_cases", [])
        self.retriever = retriever if retriever is not None else RAGRetriever()
        self.sql_agent = sql_agent if sql_agent is not None else FarmHistoryAgent()

        # Initialize LLM judge
        if llm is not None:
            self.llm = llm
        else:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                except Exception:
                    self.llm = None
            else:
                self.llm = None

    def evaluate_case(self, test_case: dict) -> dict[str, Any]:
        """Execute a single test case through the farm assistant and evaluate all metrics."""
        query = test_case["query"]
        farmer_name = test_case.get("farmer_name", "Farmer")
        top_crop = test_case.get("target_crop", "Rice")
        confidence = test_case.get("confidence", 0.90)
        features = test_case.get("features", {
            "N": 80, "P": 35, "K": 40, "pH": 6.8, "avg_temp_c": 28.0, "total_rainfall_mm": 900.0
        })

        # Generate calendar events for assistant context
        calendar_events = get_crop_calendar_events(
            crop=top_crop,
            season="Samba / Kharif",
            n=features.get("N", 80),
            p=features.get("P", 35),
            k=features.get("K", 40),
            ph=features.get("pH", 6.8),
            sowing_date=date.today()
        )

        # Retrieve RAG context if applicable
        rag_context = ""
        if test_case.get("requires_rag", False):
            try:
                rag_context = self.retriever.get_context_string(f"{top_crop} {query}")
            except Exception:
                rag_context = ""

        # Fetch SQL context if applicable
        sql_rows = []
        if test_case.get("requires_sql", False):
            try:
                sql_res = self.sql_agent.ask_farm_history("SELECT * FROM crop_history")
                sql_rows = sql_res.get("results", [])
            except Exception:
                sql_rows = []

        start_time = time.time()
        # Call the Farm Assistant
        assistant_output = query_farm_assistant(
            query=query,
            farmer_name=farmer_name,
            top_crop=top_crop,
            confidence=confidence,
            features=features,
            calendar_events=calendar_events,
            sql_agent=self.sql_agent,
            retriever=self.retriever,
            llm=self.llm
        )
        latency = round(time.time() - start_time, 2)
        answer = assistant_output.get("answer", "")

        # Run 6 Domain Metric Evaluators
        m_safety = AgronomicSafetyEvaluator.evaluate(answer, test_case)
        m_readability = FarmerReadabilityEvaluator.evaluate(answer, test_case)
        m_sql = SQLGroundingEvaluator.evaluate(answer, test_case, sql_rows)
        m_faithfulness = FaithfulnessEvaluator.evaluate_with_llm(answer, test_case, rag_context, llm=self.llm)
        m_relevance = AnswerRelevanceEvaluator.evaluate_with_llm(query, answer, test_case, llm=self.llm)
        m_context = ContextPrecisionEvaluator.evaluate(rag_context, test_case)

        # Calculate composite score (weighted mean)
        # Weights: Safety 25%, Faithfulness 20%, Relevance 20%, SQL 15%, Readability 10%, Context 10%
        composite = (
            m_safety["score"] * 0.25 +
            m_faithfulness["score"] * 0.20 +
            m_relevance["score"] * 0.20 +
            m_sql["score"] * 0.15 +
            m_readability["score"] * 0.10 +
            m_context["score"] * 0.10
        )
        composite = round(composite, 3)

        passed_all = (
            m_safety["passed"] and
            m_faithfulness["passed"] and
            m_relevance["passed"] and
            m_sql["passed"] and
            m_readability["passed"] and
            m_context["passed"]
        )

        return {
            "test_id": test_case["test_id"],
            "category": test_case["category"],
            "query": query,
            "target_crop": top_crop,
            "answer": answer,
            "latency_seconds": latency,
            "composite_score": composite,
            "passed": passed_all,
            "metrics": {
                "agronomic_safety": m_safety,
                "faithfulness": m_faithfulness,
                "answer_relevance": m_relevance,
                "sql_grounding": m_sql,
                "farmer_readability": m_readability,
                "context_precision": m_context,
            }
        }

    def evaluate_batch(
        self,
        category: Optional[str] = None,
        max_cases: Optional[int] = None
    ) -> dict[str, Any]:
        """Run batch evaluation across golden benchmark cases."""
        cases_to_run = self.test_cases
        if category:
            cases_to_run = [c for c in cases_to_run if c.get("category") == category]
        if max_cases:
            cases_to_run = cases_to_run[:max_cases]

        results = []
        start_time = time.time()

        for tc in cases_to_run:
            res = self.evaluate_case(tc)
            results.append(res)
            time.sleep(1.0)

        total_time = round(time.time() - start_time, 2)
        total_count = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        pass_rate = round(passed_count / total_count, 3) if total_count > 0 else 0.0

        # Mean metric scores
        metric_sums = {
            "agronomic_safety": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "sql_grounding": 0.0,
            "farmer_readability": 0.0,
            "context_precision": 0.0,
            "composite_score": 0.0
        }

        category_stats = {}
        for r in results:
            cat = r["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "composite_sum": 0.0}
            category_stats[cat]["total"] += 1
            if r["passed"]:
                category_stats[cat]["passed"] += 1
            category_stats[cat]["composite_sum"] += r["composite_score"]

            metric_sums["composite_score"] += r["composite_score"]
            for m_key in [
                "agronomic_safety", "faithfulness", "answer_relevance",
                "sql_grounding", "farmer_readability", "context_precision"
            ]:
                metric_sums[m_key] += r["metrics"][m_key]["score"]

        avg_metrics = {
            k: round(v / total_count, 3) if total_count > 0 else 0.0
            for k, v in metric_sums.items()
        }

        for cat, stat in category_stats.items():
            stat["pass_rate"] = round(stat["passed"] / stat["total"], 2)
            stat["avg_composite"] = round(stat["composite_sum"] / stat["total"], 3)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cases_evaluated": total_count,
            "passed_cases": passed_count,
            "overall_pass_rate": pass_rate,
            "average_metrics": avg_metrics,
            "category_performance": category_stats,
            "total_duration_seconds": total_time,
            "case_results": results
        }
        return report

    def save_report(self, report: dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save evaluation report to JSON file."""
        if not output_path:
            base_dir = Path(__file__).resolve().parent.parent
            output_path = str(base_dir / "eval" / "latest_eval_report.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return output_path
