#!/usr/bin/env python3
"""
Clarification Agent Factory

Factory function that creates a clarification agent to identify gaps and ambiguities
in user requests and generate relevant clarification questions.

Usage:
    from clarification_agent.agent import make_clarification_agent

    def my_collector(questions: str):
        print(f"Questions: {questions}\n")

    agent = make_clarification_agent(my_collector)
"""

import logging
from typing import Any
from collections.abc import Callable as ABCCallable

from agents import (
    Agent,
    ModelSettings,
    AgentHooks,
    RunContextWrapper,
)

# Import helper functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings

logger = logging.getLogger(__name__)


class ClarificationHooks(AgentHooks[Any]):
    """Lifecycle hooks that automatically collect clarification questions when the agent completes."""

    def __init__(self, collect_callback: ABCCallable[[str], None]):
        self.collect_callback = collect_callback

    async def on_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
    ) -> None:
        """Called before the agent is invoked."""
        try:
            # No special processing needed at start for clarification agent
            pass
        except Exception as e:
            logger.debug(f"Could not process on_start: {str(e)}")

    async def on_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output - automatically collect the questions."""
        try:
            questions = str(output) if output else ""

            # Call the collection callback with the questions
            self.collect_callback(questions)
            logger.info(f"Automatically collected clarification questions (length: {len(questions)} chars)")

        except Exception as e:
            # Log errors but don't interrupt the agent's response
            logger.error(f"Error in clarification collection hook: {str(e)}")


def make_clarification_agent(
    collect: ABCCallable[[str], None],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
) -> Agent[Any]:
    """
    Create a clarification agent that identifies gaps and ambiguities in user requests.

    Parameters
    ----------
    collect : (questions:str) -> None
        Callback that receives the clarification questions.
    model : str
        LLM model to use for the agent.
    temperature : float
        Temperature setting for the LLM.

    Returns
    -------
    Agent[Any]
        Ready-to-use Agents-SDK agent instance with automatic collection.
    """

    # Configure model settings using helper function
    model_settings = get_model_settings(
        model_name=model,
        temperature=temperature,
        max_tokens=2048,  # Sufficient for clarification questions
        parallel_tool_calls=False
    )

    # Create the hooks internally
    collection_hooks = ClarificationHooks(collect)

    return Agent[Any](
        name="Clarification Agent",
        instructions=(
            "You are a clarification agent. Analyze the user's request and identify what information is missing, ambiguous, or unclear.\n\n"
            "Generate specific clarification questions that would help make the request more precise and actionable. Focus on:\n"
            "- Ambiguous terms that need definition\n"
            "- Missing context or requirements\n" 
            "- Unclear scope or boundaries\n"
            "- Undefined constraints or preferences\n"
            "- Missing success criteria\n\n"
            "For each question, also suggest what a good answer might include to guide the user.\n\n"
            "Format your response as a numbered list:\n"
            "1. [Question] - A good answer should include: [guidance]\n"
            "2. [Question] - A good answer should include: [guidance]\n"
            "etc.\n\n"
            "Ask 3-7 focused questions that would most improve the clarity of the request."
        ),
        tools=[],  # No external tools needed - pure LLM analysis
        model=model,
        model_settings=model_settings,
        hooks=collection_hooks,  # Attach the collection hooks
    ) 