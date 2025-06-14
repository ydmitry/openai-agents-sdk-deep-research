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
    ModelSettings,
    AgentHooks,
    RunContextWrapper,
    WebSearchTool,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Import helper functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings

# Import sequential search agent for handoff
from sequential_search_agent import make_sequential_search_agent

logger = logging.getLogger(__name__)


class ClarificationHooks(AgentHooks[Any]):
    """Lifecycle hooks that display questions but do NOT collect them."""

    def __init__(self, 
                 collect_callback: Optional[ABCCallable[[str], None]] = None,
                 display_callback: Optional[ABCCallable[[str], None]] = None):
        self.collect_callback = collect_callback  # For handoff to search agent only
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
    collect: ABCCallable[[str], None],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    enable_handoff: bool = False,
    display_callback: Optional[ABCCallable[[str], None]] = None,
) -> Agent[Any]:
    """
    Create a clarification agent that identifies gaps and ambiguities in user requests.

    Parameters
    ----------
    collect : (questions:str) -> None
        Callback that receives search results from handoff agents (NOT used for questions).
    model : str
        LLM model to use for the agent.
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

    # Configure model settings using helper function
    model_settings = get_model_settings(
        model_name=model,
        temperature=temperature,
        max_tokens=2048,  # Sufficient for clarification questions
        parallel_tool_calls=False
    )

    # Create the hooks with display callback
    clarification_hooks = ClarificationHooks(
        collect_callback=collect,  # Passed to handoff agents only
        display_callback=display_callback  # For displaying questions
    )

    # Prepare handoffs if enabled
    handoffs = []
    if enable_handoff:
        # Create sequential search agent for handoff - THIS gets the collector
        search_agent = make_sequential_search_agent(collect, model=model, temperature=temperature)
        handoffs = [search_agent]

    # Instructions based on whether handoff is enabled
    if enable_handoff:
        instructions = (
            f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
            "You are a clarification agent that helps users refine their requests before conducting research.\n\n"
            "WORKFLOW:\n"
            "1. IMPORTANT! Detect terms and use web_search to understand them first\n"
            "2. When users provide vague or ambiguous requests, ask 3-7 clarifying questions\n"
            "3. When users provide comprehensive answers to your questions, handoff to the search agent\n"
            "4. When users ask for research, information gathering, or web search, handoff to the search agent\n"
            "5. When the request is clear and actionable, handoff to the search agent\n\n"
            "SEARCH STRATEGY:\n"
            "- Use web_search for unknown terms, technologies, frameworks, or concepts mentioned by the user\n"
            "- Search for context to better understand the domain or industry they're working in\n"
            "- This helps you ask more informed and relevant clarification questions\n\n"
            "HANDOFF CRITERIA - Handoff when:\n"
            "- User has answered most/all clarification questions with sufficient detail\n"
            "- User provides comprehensive context, requirements, and constraints\n"
            "- User explicitly requests research, analysis, or information gathering\n"
            "- Request contains enough specificity for effective searching\n"
            "- No major ambiguities remain that would hinder research\n\n"
            "CLARIFICATION FORMAT (when needed):\n"
            "1. [Question] - A good answer should include: [guidance]\n"
            "2. [Question] - A good answer should include: [guidance]\n"
            "etc.\n\n"
            "Always handoff when the request is sufficiently clarified for comprehensive research."
        )
    else:
        instructions = (
            "You are a clarification agent. Analyze the user's request and identify what information is missing, ambiguous, or unclear.\n\n"
            "WORKFLOW:\n"
            "1. IMPORTANT! Detect terms and use web_search to understand them first\n"
            "2. Generate specific clarification questions based on your understanding\n\n"
            "SEARCH STRATEGY:\n"
            "- Use web_search for unknown terms, technologies, frameworks, or concepts mentioned by the user\n"
            "- Search for context to better understand the domain or industry they're working in\n"
            "- This helps you ask more informed and relevant clarification questions\n\n"
            "Generate specific clarification questions that would help make the request more precise and actionable. Focus on:\n"
            "- Ambiguous terms that need definition (search first if unfamiliar)\n"
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
        )

    return Agent[Any](
        name="Clarification Agent" + (" with Handoff" if enable_handoff else ""),
        instructions=instructions,
        tools=[WebSearchTool(search_context_size="low")],  # Add search tool for unknown terms
        model=model,
        model_settings=model_settings,
        hooks=clarification_hooks,  # Attach the display hooks
        handoffs=handoffs,  # Enable handoff to search agent if requested
    )
