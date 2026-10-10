"""Command Line Interface for Agricultural Model Evaluation Benchmark.

Usage:
    python -m eval.run_eval --sample 3
    python -m eval.run_eval --category "Farm History (SQL)"
    python -m eval.run_eval --all
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Reconfigure stdout/stderr to utf-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from eval.evaluator import AgriculturalModelEvaluator


def print_banner():
    print("=" * 80)
    print(" [BENCHMARK] TAMIL NADU CROP ADVISORY AI MODEL EVALUATION BENCHMARK")
    print(" Evaluating: LLM Field Assistant, RAG Retriever, SQLite History & Guardrails")
    print("=" * 80)


def generate_markdown_report(report: dict, output_md: str) -> None:
    """Generate a clean markdown report from evaluation results."""
    md_lines = [
        "# 🌾 Agricultural AI Model Evaluation Benchmark Report",
        f"**Date:** {report.get('timestamp', datetime.now().isoformat())}  ",
        f"**Total Test Cases Evaluated:** {report.get('total_cases_evaluated')}  ",
        f"**Passed Cases:** {report.get('passed_cases')} ({report.get('overall_pass_rate', 0):.1%})  ",
        f"**Benchmark Runtime:** {report.get('total_duration_seconds')}s  \n",
        "## 1. Domain Metric Averages",
        "| Metric | Average Score (0-1.0) | Benchmark Status |",
        "| :--- | :---: | :---: |",
    ]

    metrics = report.get("average_metrics", {})
    status_map = lambda s: "✅ Pass" if s >= 0.80 else ("⚠️ Caution" if s >= 0.70 else "❌ Fail")

    for k, v in metrics.items():
        name = k.replace("_", " ").title()
        md_lines.append(f"| **{name}** | `{v:.3f}` | {status_map(v)} |")

    md_lines.extend([
        "\n## 2. Category Performance",
        "| Category | Total Cases | Passed Cases | Pass Rate | Average Composite Score |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for cat, stat in report.get("category_performance", {}).items():
        pass_pct = f"{stat.get('pass_rate', 0):.0%}"
        md_lines.append(f"| **{cat}** | {stat.get('total')} | {stat.get('passed')} | {pass_pct} | `{stat.get('avg_composite', 0):.3f}` |")

    md_lines.extend([
        "\n## 3. Individual Test Case Results",
        "| ID | Category | Target Crop | Latency | Composite Score | Result |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ])

    for c in report.get("case_results", []):
        res_str = "✅ Pass" if c.get("passed") else "❌ Fail"
        md_lines.append(
            f"| `{c.get('test_id')}` | {c.get('category')} | {c.get('target_crop')} | {c.get('latency_seconds')}s | `{c.get('composite_score'):.3f}` | {res_str} |"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    parser = argparse.ArgumentParser(description="Run Crop Advisory Model Evaluation Benchmark")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate first N sample cases")
    parser.add_argument("--category", type=str, default=None, help="Filter by specific category")
    parser.add_argument("--output", type=str, default=None, help="Custom JSON output path")
    parser.add_argument("--md", type=str, default=None, help="Export Markdown report path")
    args = parser.parse_args()

    print_banner()
    evaluator = AgriculturalModelEvaluator()

    total_avail = len(evaluator.test_cases)
    print(f"Loaded {total_avail} golden test cases from dataset.")
    if args.category:
        print(f"Filtering by category: {args.category}")
    if args.sample:
        print(f"Limiting to first {args.sample} sample cases.")

    print("\nRunning model evaluation against active target pipeline...")
    report = evaluator.evaluate_batch(category=args.category, max_cases=args.sample)

    # Print summary
    print("\n" + "=" * 80)
    print(" [RESULTS] EVALUATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Overall Pass Rate: {report['overall_pass_rate']:.1%} ({report['passed_cases']}/{report['total_cases_evaluated']} passed)")
    print(f"Total Duration:    {report['total_duration_seconds']} seconds\n")

    print("Domain Metric Breakdown:")
    for k, v in report["average_metrics"].items():
        name = k.replace("_", " ").title()
        badge = "[PASS]" if v >= 0.80 else ("[WARN]" if v >= 0.70 else "[FAIL]")
        print(f"  {badge} {name:<26}: {v:.3f} / 1.000")

    print("\nCategory Performance:")
    for cat, stat in report["category_performance"].items():
        print(f"  * {cat:<24}: {stat['passed']}/{stat['total']} passed ({stat['pass_rate']:.0%}), Avg Score: {stat['avg_composite']:.3f}")

    # Save output
    base_dir = Path(__file__).resolve().parent.parent
    json_path = args.output or str(base_dir / "eval" / "latest_eval_report.json")
    md_path = args.md or str(base_dir / "eval" / "latest_eval_report.md")

    evaluator.save_report(report, json_path)
    generate_markdown_report(report, md_path)

    print(f"\n[Saved JSON Report]     {json_path}")
    print(f"[Saved Markdown Report] {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
