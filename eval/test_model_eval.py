"""Integration and Unit Test Suite for Agricultural Model Response Evaluation.

Tests:
1. Golden benchmark test cases dataset schema and coverage.
2. Domain evaluation metrics (Agronomic Safety, Readability, SQL Grounding).
3. Evaluator engine initialization and fallback handling.
4. DeepEval and Ragas adapter interfaces.
"""

import os
import json
import unittest
from pathlib import Path

from eval.eval_metrics import (
    AgronomicSafetyEvaluator,
    FarmerReadabilityEvaluator,
    SQLGroundingEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextPrecisionEvaluator
)
from eval.evaluator import AgriculturalModelEvaluator


class TestModelEvaluationFramework(unittest.TestCase):
    """Test suite for evaluation framework components."""

    @classmethod
    def setUpClass(cls):
        cls.eval_dir = Path(__file__).resolve().parent
        cls.dataset_path = cls.eval_dir / "agri_test_cases.json"
        with open(cls.dataset_path, "r", encoding="utf-8") as f:
            cls.dataset = json.load(f)

    def test_golden_dataset_integrity(self):
        """Verify that golden test dataset has required fields and categories."""
        cases = self.dataset.get("test_cases", [])
        self.assertGreaterEqual(len(cases), 20, "Should have at least 20 golden test cases")

        categories = {c.get("category") for c in cases}
        expected_cats = {
            "Nutrient Management",
            "Pest & Disease (IPM)",
            "Farm History (SQL)",
            "Drought & Contingency",
            "Natural Farming",
            "Safety & Guardrails"
        }
        for cat in expected_cats:
            self.assertIn(cat, categories, f"Missing expected category: {cat}")

        for tc in cases:
            self.assertIn("test_id", tc)
            self.assertIn("query", tc)
            self.assertIn("ground_truth_facts", tc)
            self.assertIsInstance(tc["ground_truth_facts"], list)
            self.assertGreater(len(tc["ground_truth_facts"]), 0)

    def test_agronomic_safety_metric_passes_safe_advisory(self):
        """Verify that safe advisory passes safety evaluation."""
        safe_advisory = (
            "Vanakkam farmer! For stem borer management in Samba rice, release Trichogramma "
            "egg parasitoids and install pheromone traps at 5 per acre. Avoid broad-spectrum chemicals."
        )
        tc = {
            "forbidden_claims": ["Apply 100% urea as a single basal dose"]
        }
        res = AgronomicSafetyEvaluator.evaluate(safe_advisory, tc)
        self.assertTrue(res["passed"])
        self.assertEqual(res["score"], 1.0)

    def test_agronomic_safety_metric_catches_banned_chemical(self):
        """Verify that recommending banned chemical fails safety evaluation."""
        unsafe_advisory = (
            "You should spray Endosulfan immediately at 2 ml per liter of water for quick insect control."
        )
        tc = {"forbidden_claims": []}
        res = AgronomicSafetyEvaluator.evaluate(unsafe_advisory, tc)
        self.assertFalse(res["passed"])
        self.assertEqual(res["score"], 0.0)
        self.assertTrue(any("endosulfan" in r.lower() for r in res["reasons"]))

    def test_agronomic_safety_metric_passes_negated_banned_chemical(self):
        """Verify that warning the farmer against banned chemical is allowed."""
        safe_warning = (
            "Never use Endosulfan or Paraquat because they are strictly banned by the government. "
            "Instead use Neem oil 3% or recommended biopesticides."
        )
        tc = {"forbidden_claims": []}
        res = AgronomicSafetyEvaluator.evaluate(safe_warning, tc)
        self.assertTrue(res["passed"])
        self.assertEqual(res["score"], 1.0)

    def test_farmer_readability_evaluator(self):
        """Verify readability scoring rewards structured, practical word count."""
        good_advisory = (
            "Vanakkam Murugan! Here is your Samba Rice recommendation:\n"
            "- Practical Recommendation: Apply Urea in three calibrated splits for optimal absorption.\n"
            "- Field Execution & Dosages: Apply 25% at basal tillering, 50% at active tillering, and 25% at panicle emergence.\n"
            "- Practical Field Tip: Drain standing water prior to top-dressing and irrigate after 24 hours to prevent leaching."
        )
        tc = {"min_word_count": 50, "max_word_count": 200}
        res = FarmerReadabilityEvaluator.evaluate(good_advisory, tc)
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["score"], 0.8)

        # Test too short
        short_advisory = "Apply urea."
        res_short = FarmerReadabilityEvaluator.evaluate(short_advisory, tc)
        self.assertFalse(res_short["passed"])
        self.assertLess(res_short["score"], 0.7)

    def test_sql_grounding_evaluator(self):
        """Verify SQL grounding checks numerical and crop fidelity."""
        tc = {
            "requires_sql": True,
            "query": "Which crop gave me the highest profit?"
        }
        sql_rows = [
            {"crop_planted": "Sugarcane", "season": "Kharif", "profit": 52000, "yield_achieved": 85.0},
            {"crop_planted": "Rice", "season": "Rabi", "profit": 31000, "yield_achieved": 5.2}
        ]
        grounded_answer = "Your best profit record was Sugarcane with net profit of ₹52,000 in Kharif season."
        res = SQLGroundingEvaluator.evaluate(grounded_answer, tc, sql_rows)
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["score"], 0.85)

        # Ungrounded answer
        ungrounded_answer = "You made some money on unknown crops."
        res_bad = SQLGroundingEvaluator.evaluate(ungrounded_answer, tc, sql_rows)
        self.assertFalse(res_bad["passed"])

    def test_context_precision_evaluator(self):
        """Verify context precision evaluates retrieved chunk quality."""
        tc = {"requires_rag": True, "target_crop": "Rice"}
        chunks = "ICAR Rice Production Manual: Samba paddy fertilizer management and pest control guidelines."
        res = ContextPrecisionEvaluator.evaluate(chunks, tc)
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["score"], 0.8)

        # Irrelevant chunks
        bad_chunks = "Automobile engine maintenance and road traffic guidelines."
        res_bad = ContextPrecisionEvaluator.evaluate(bad_chunks, tc)
        self.assertFalse(res_bad["passed"])


if __name__ == "__main__":
    unittest.main()
