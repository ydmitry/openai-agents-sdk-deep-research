#!/usr/bin/env python3
"""
Full Deep Research Pipeline Example
==================================
This script demonstrates running the complete three-step deep research pipeline:
1. Planning - Generate a research plan with sub-tasks
2. Search & Scraping - Search for and scrape relevant documents
3. Summarization - Generate a citation-based summary report

Usage:
    python examples/run_pipeline.py "Your research topic" --out-dir results

Models:
    Supports standard models like gpt-4.1 and also o4-mini with high reasoning effort.
"""

import os
import json
import asyncio
import argparse
from pathlib import Path

from deep_research.step1 import generate_research_plan
from deep_research.step2 import build_corpus
from deep_research.step3 import generate_report
from deep_research.utils import load_dotenv_files


async def run_pipeline_async(topic, out_dir, models=None, concurrency=4):
    """
    Run all three steps of the deep research pipeline.

    Args:
        topic: Research topic
        out_dir: Output directory for all files
        models: Dict of models to use for each step
        concurrency: Level of parallelism for steps 2 and 3
    """
    if models is None:
        models = {
            "planner": "o4-mini",
            "searcher": "gpt-4.1",
            "bullet": "gpt-4.1",
            "summary": "gpt-4.1",
            "report": "gpt-4.1",
            "critic": "gpt-4.1",
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plan_file = out_dir / "plan.json"
    corpus_file = out_dir / "corpus.jsonl"
    report_file = out_dir / "report.md"

    # Step 1: Generate research plan
    print(f"Step 1: Generating research plan for topic: {topic}")
    print(f"Using model: {models['planner']}")
    plan = await generate_research_plan(
        topic,
        model=models["planner"]
    )

    # Save plan to file
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan.__dict__, f, default=lambda o: o.__dict__, indent=2)

    print(f"Research plan saved to {plan_file}")
    print(f"Sub-tasks: {len(plan.sub_tasks)}")

    # Step 2: Search and scrape documents
    print("\nStep 2: Searching and scraping documents for each sub-task")
    print(f"Using model: {models['searcher']}")
    documents = await build_corpus(
        plan,
        model=models["searcher"],
        concurrency=concurrency
    )

    # Save corpus to file
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.to_json() + "\n")

    print(f"Research corpus saved to {corpus_file}")
    print(f"Documents: {len(documents)}")

    # Step 3: Generate summary report
    print("\nStep 3: Generating research report")
    print(f"Using models: bullet={models['bullet']}, summary={models['summary']}, report={models['report']}, critic={models['critic']}")
    report, fact_check = await generate_report(
        documents,
        plan.objective,
        bullet_model=models["bullet"],
        summary_model=models["summary"],
        report_model=models["report"],
        critic_model=models["critic"],
        concurrency=concurrency
    )

    # Save report to file
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# {report.title}\n\n")
        f.write(report.body_md)
        f.write("\n\n## References\n\n")
        for ref in report.references:
            f.write(f"{ref}\n")

    print(f"Research report saved to {report_file}")

    # Save critique if available
    if fact_check:
        critique_file = out_dir / "critique.md"
        with open(critique_file, "w", encoding="utf-8") as f:
            f.write(f"# Fact Check: {report.title}\n\n")
            f.write(f"## Overall Assessment\n\n{fact_check.overall_assessment}\n\n")

            if fact_check.unsupported_claims:
                f.write("## Unsupported Claims\n\n")
                for claim in fact_check.unsupported_claims:
                    f.write(f"- {claim}\n")
                f.write("\n")

            if fact_check.conflicting_claims:
                f.write("## Conflicting Claims\n\n")
                for claim in fact_check.conflicting_claims:
                    f.write(f"- {claim}\n")

        print(f"Critique saved to {critique_file}")

    print("\nDeep research pipeline completed successfully!")
    return {
        "plan": plan,
        "documents": documents,
        "report": report,
        "fact_check": fact_check
    }


def run_pipeline(topic, out_dir, models=None, concurrency=4):
    """Sync wrapper for the async pipeline function."""
    return asyncio.run(run_pipeline_async(topic, out_dir, models, concurrency))


def main():
    parser = argparse.ArgumentParser(description="Run the complete deep research pipeline")
    parser.add_argument("objective", help="Research objective or question")
    parser.add_argument("--out-dir", "-o", default="./output", help="Output directory for all artifacts")
    parser.add_argument("--model", "-m", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Parallel tasks (default: 4)")
    parser.add_argument("--skip-verification", action="store_true", help="Skip the final verification step")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv_files()

    models = {
        "planner": args.model,
        "searcher": args.model,
        "bullet": args.model,
        "summary": args.model,
        "report": args.model,
        "critic": args.model if not args.skip_verification else None,
    }

    run_pipeline(args.objective, args.out_dir, models, args.concurrency)


if __name__ == "__main__":
    main()
