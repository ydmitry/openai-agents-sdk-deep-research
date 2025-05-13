#!/usr/bin/env python3
"""
Example script demonstrating the full deep research pipeline (Steps 1 and 2).

This script shows how to:
1. Generate a research plan using step1
2. Use that plan to build a corpus of documents using step2

Usage:
    python research_pipeline.py "Your research topic here" --out-dir ./results

This will create:
- results/plan.json: The research plan
- results/corpus.jsonl: The corpus of documents
"""

import argparse
import json
import os
import sys
from pathlib import Path

from deep_research import generate_research_plan, build_corpus


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a deep research pipeline (Steps 1 and 2)")
    parser.add_argument("objective", help="Research objective or topic")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    parser.add_argument("--out-dir", default="./results", help="Output directory (default: ./results)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent tasks for step 2 (default: 4)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    plan_file = out_dir / "plan.json"
    corpus_file = out_dir / "corpus.jsonl"
    
    # Step 1: Generate research plan
    print(f"Generating research plan for: {args.objective}")
    plan = generate_research_plan(args.objective, model=args.model)
    
    # Save plan to file
    with plan_file.open("w", encoding="utf-8") as f:
        f.write(plan.to_json())
    
    print(f"Wrote research plan to {plan_file}")
    print(f"Plan has {len(plan.sub_tasks)} sub-tasks")
    
    # Step 2: Build corpus
    print(f"Building research corpus (this may take a while)...")
    docs = build_corpus(plan, model=args.model, concurrency=args.concurrency)
    
    # Save corpus to file
    with corpus_file.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.to_json() + "\n")
    
    print(f"Wrote {len(docs)} documents to {corpus_file}")
    
    # Summary of results
    docs_by_task = {}
    for doc in docs:
        task_id = doc.source_task_id
        if task_id not in docs_by_task:
            docs_by_task[task_id] = 0
        docs_by_task[task_id] += 1
    
    print("\nDocuments per sub-task:")
    for i, task in enumerate(plan.sub_tasks):
        count = docs_by_task.get(task.id, 0)
        print(f"  Task {task.id} (priority {task.priority}): {count} documents")
        print(f"    - {task.task}")
    
    print("\nResearch pipeline completed successfully!")


if __name__ == "__main__":
    main() 