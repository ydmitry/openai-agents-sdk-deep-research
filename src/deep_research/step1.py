"""
Step 1 of an AutoGPT‑style deep‑research pipeline implemented with the OpenAI Agents SDK.

This module defines the **Reasoning / Planning layer** responsible for
transforming a high‑level research objective into a prioritized, structured
list of sub‑tasks that downstream agents can execute (e.g. web search,
focused crawling, summarization).  It corresponds to the *Planner* role
in the classic Planner‑Executor architecture and follows the AutoGPT pattern
of Plan → Execute → Reflect loops.

Key features
------------
* Uses the OpenAI Agents SDK.
* Employs a *self‑reflection* loop so the agent can critique and refine its
  own sub‑task list before returning it.
* Outputs a JSON object that downstream components can consume directly.
* Keeps the reasoning trace for observability/tracing.

Environment
-----------
Requires:
    pip install openai-agents>=0.0.14
    export OPENAI_API_KEY="sk‑..."

Example
-------
>>> from deep_research.step1 import generate_research_plan
>>> plan = generate_research_plan("Assess the economic impact of generative AI on the 2025 EU labor market")
>>> print(plan)
{
  "objective": "Assess the economic impact of generative AI on the 2025 EU labor market",
  "sub_tasks": [
    {"id": 1, "task": "Identify key sectors in the EU labor market affected by gen‑AI", "priority": 1},
    {"id": 2, "task": "Gather 2024–2025 employment statistics from Eurostat & OECD", "priority": 2},
    ...
  ]
}
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional

from agents import Agent, Runner, RunConfig, ModelSettings

from deep_research.utils import load_dotenv_files, get_model_settings

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class SubTask:
    id: int
    task: str
    priority: int  # lower number == higher priority

@dataclass
class ResearchPlan:
    objective: str
    sub_tasks: List[SubTask]

    def to_json(self) -> str:
        return json.dumps({
            "objective": self.objective,
            "sub_tasks": [asdict(st) for st in self.sub_tasks],
        }, indent=2)

# -----------------------------
# Agent definition
# -----------------------------

SYSTEM_INSTRUCTIONS = """
You are ***AutoPlan***, the **Planner** agent in a multi‑step deep‑research pipeline.
Your goal is to translate the high‑level research *objective* provided by the user
into a concise, prioritized list of executable sub‑tasks.  Use the following
policy:

1. **Brainstorm**: Think step‑by‑step (Chain‑of‑Thought) to break the objective
   into granular research questions.  Cover breadth (major themes) before depth
   (specific metrics or sources).
2. **Prioritize**: Rank the questions by their importance in addressing the
   objective (1 = highest priority).  Limit to 8–12 tasks to keep downstream
   workloads manageable.
3. **Reflect**: Critically evaluate whether the tasks collectively and
   logically cover the objective.  If not, revise.
4. **Output**: Return *only* a valid JSON object of the form:

    {
        "objective": <string>,
        "sub_tasks": [
            {"id": 1, "task": <string>, "priority": <int>},
            ...
        ]
    }

• Do **NOT** include any additional keys.
• Do **NOT** wrap the JSON in markdown.
"""

# The reasoning layer does not need external tools; we supply none here.
planner_agent = Agent(
    name="Planner",
    instructions=SYSTEM_INSTRUCTIONS,
)

# -----------------------------
# Public API
# -----------------------------

async def _async_generate_plan(objective: str, model: str = "o4-mini", **kwargs) -> ResearchPlan:
    """Async helper that runs the agent and returns a structured ResearchPlan."""
    run_id = uuid.uuid4().hex[:6]

    # Extract model settings from kwargs
    temperature = kwargs.pop('temperature', 0.7)
    max_tokens = kwargs.pop('max_tokens', None)

    # Create model settings with conditional reasoning effort
    model_settings = get_model_settings(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Create run configuration with proper trace_id format
    run_config = RunConfig(
        model=model,
        model_settings=model_settings,
        # Disable tracing to avoid the trace_id format issue
        tracing_disabled=True,
        workflow_name="Research Plan Generator"
    )

    # Run the agent
    result = await Runner.run(
        planner_agent,
        objective,
        run_config=run_config,
        **kwargs,
    )

    # Get the response from the result
    # In version 0.0.14, access the final_output directly
    final_message = result.final_output

    # Parse JSON from the message content
    try:
        # Try to extract JSON from the message content
        # Look for JSON pattern in the response
        import re
        json_match = re.search(r'({.*})', final_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            parsed = json.loads(json_str)
        else:
            # If no JSON pattern found, try parsing the entire content
            parsed = json.loads(final_message)
    except json.JSONDecodeError:
        # In case of failure, return a descriptive error
        raise ValueError(f"Failed to parse JSON from agent response: {final_message}")

    sub_tasks = [SubTask(**st) for st in parsed["sub_tasks"]]
    return ResearchPlan(objective=parsed["objective"], sub_tasks=sub_tasks)


def generate_research_plan(objective: str, **kwargs) -> ResearchPlan:
    """Blocking wrapper around :pyfunc:`_async_generate_plan`.

    Args:
        objective: High-level research goal provided by the user.
        **kwargs: Extra arguments forwarded to the Runner (e.g., `temperature`).

    Returns:
        ResearchPlan dataclass comprising the objective and prioritized sub‑tasks.
    """
    # When called from an event loop, we need to just await the coroutine
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We are already in an event loop, so just create and return the coroutine
            return _async_generate_plan(objective, **kwargs)
    except RuntimeError:
        # No event loop running, so use asyncio.run to create one
        return asyncio.run(_async_generate_plan(objective, **kwargs))

# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    import argparse

    # Load environment variables from .env files
    loaded_files = load_dotenv_files()

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or export it directly.")
        print("Example content for .env file: OPENAI_API_KEY=sk-your-api-key")
        exit(1)

    parser = argparse.ArgumentParser(description="Generate an AutoGPT-style research plan (Step 1: Reasoning layer)")
    parser.add_argument("objective", help="High-level research objective, in quotes")
    parser.add_argument("--model", default="o4-mini", help="OpenAI model name (default: o4-mini)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation (default: 0.7)")
    args = parser.parse_args()

    plan = generate_research_plan(args.objective, model=args.model, temperature=args.temperature)
    print("\n===== RESEARCH PLAN =====\n")
    print(plan.to_json())
