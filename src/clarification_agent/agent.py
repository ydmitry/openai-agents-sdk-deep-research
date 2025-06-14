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
from typing import Any, Optional
from collections.abc import Callable as ABCCallable

from agents import (
    Agent,
    AgentHooks,
    RunContextWrapper,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Import helper functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings
from storage.search_results import SearchResultsStorage

# Import sequential search agent for handoff
from sequential_search_agent import make_sequential_search_agent

logger = logging.getLogger(__name__)


class ClarificationHooks(AgentHooks[Any]):
    """Lifecycle hooks that display questions but do NOT collect them."""

    def __init__(self,
                 storage: Optional[SearchResultsStorage] = None,
                 display_callback: Optional[ABCCallable[[str], None]] = None):
        self.storage = storage  # For handoff to search agent only
        self.display_callback = display_callback  # For displaying questions to user

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
        """Called when the agent produces a final output - display questions but do NOT collect."""
        try:
            questions = str(output) if output else ""

            # Display questions to user if display callback provided
            if self.display_callback:
                self.display_callback(questions)
                logger.info(f"Displayed clarification questions (length: {len(questions)} chars)")

            # NEVER collect questions - only search results from handoff agents get collected
            logger.debug("Clarification agent output not collected (by design)")

        except Exception as e:
            # Log errors but don't interrupt the agent's response
            logger.error(f"Error in clarification display hook: {str(e)}")


def make_clarification_agent(
    storage: SearchResultsStorage,
    *,
    model: str = "gpt-4.1-mini",
    clarification_model: Optional[str] = None,
    temperature: float = 0.3,
    enable_handoff: bool = False,
    display_callback: Optional[ABCCallable[[str], None]] = None,
) -> Agent[Any]:
    """
    Create a clarification agent that identifies gaps and ambiguities in user requests.

    Parameters
    ----------
    storage : SearchResultsStorage
        Storage object that receives search results from handoff agents (NOT used for questions).
    model : str
        LLM model to use for the search agent in handoff scenarios.
    clarification_model : Optional[str]
        LLM model to use specifically for the clarification agent. If None, uses the model parameter.
    temperature : float
        Temperature setting for the LLM.
    enable_handoff : bool
        Whether to enable handoff to sequential search agent.
    display_callback : Optional[(questions:str) -> None]
        Optional callback for displaying questions to user (separate from collection).

    Returns
    -------
    Agent[Any]
        Ready-to-use Agents-SDK agent instance with display hooks.
    """

    # Use clarification_model if provided, otherwise fall back to model
    agent_model = clarification_model or model

    # Configure model settings using helper function
    model_settings = get_model_settings(
        model_name=agent_model,
        temperature=temperature,
        max_tokens=2048,  # Sufficient for clarification questions
        parallel_tool_calls=False
    )

    # Create the hooks with display callback
    clarification_hooks = ClarificationHooks(
        storage=storage,  # Passed to handoff agents only
        display_callback=display_callback  # For displaying questions
    )

    # Prepare handoffs if enabled
    handoffs = []
    if enable_handoff:
        # Create sequential search agent for handoff - THIS gets the storage
        # Use the original model parameter for search agent
        search_agent = make_sequential_search_agent(storage, model=model, temperature=temperature)
        handoffs = [search_agent]

    # Instructions based on whether handoff is enabled
    if enable_handoff:
        instructions = (
            f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
            "You are a clarification agent that helps users refine their requests before conducting research.\n\n"
            "WORKFLOW:\n"
            "1. When users provide vague or ambiguous requests, ask 3-7 clarifying questions\n"
            "2. When users provide comprehensive answers to your questions, handoff to the search agent\n"
            "3. When users ask for research, information gathering, or web search, handoff to the search agent\n"
            "4. When the request is clear and actionable, rewrite question to be more specific and handoff to the search agent\n\n"
            "HANDOFF CRITERIA - Handoff when:\n"
            "- User has answered most/all clarification questions with sufficient detail\n"
            "- User provides comprehensive context, requirements, and constraints\n"
            "- User explicitly requests research, analysis, or information gathering\n"
            "- Request contains enough specificity for effective searching\n"
            "- No major ambiguities remain that would hinder research\n\n"
            "- Original question was rewritten to be more specific\n\n"
            "CLARIFICATION FORMAT (when needed):\n"
            "1. [Question] - A good answer should include: [guidance]\n"
            "2. [Question] - A good answer should include: [guidance]\n"
            "etc.\n\n"
            "Always handoff when the request is sufficiently clarified for comprehensive research."
        )
    else:
        instructions = (
            "You are a clarification agent. Analyze the user's request and identify what information is missing, ambiguous, or unclear.\n\n"
            "Generate specific clarification questions that would help make the request more precise and actionable. Focus on:\n"
            "For each question, also suggest what a good answer might include to guide the user.\n\n"
            "Format your response as a numbered list:\n"
            "1. [Question] - A good answer should include: [guidance]\n"
            "2. [Question] - A good answer should include: [guidance]\n"
            "etc.\n\n"
            "Ask 3-7 focused questions that would most improve the clarity of the request."
        )

    return Agent[Any](
        name="Clarification Agent" + (" with Handoff" if enable_handoff else ""),
        instructions=instructions,
        model=agent_model,
        model_settings=model_settings,
        hooks=clarification_hooks,  # Attach the display hooks
        handoffs=handoffs,  # Enable handoff to search agent if requested
    )
