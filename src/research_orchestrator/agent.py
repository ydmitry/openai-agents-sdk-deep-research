#!/usr/bin/env python3
"""
Research Orchestrator Agent Factory

Factory function for creating the research orchestrator agent.
"""

import logging
from typing import Any

from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings
from storage.search_results import SearchResultsStorage

from clarification_agent.agent import make_clarification_agent
from sequential_search_agent.agent import make_sequential_search_agent


logger = logging.getLogger(__name__)

def make_research_orchestrator(
    storage: SearchResultsStorage,
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
) -> Agent[Any]:
    """
    Creates the research orchestrator agent.

    This agent is responsible for receiving the initial user query and handing
    it off to the appropriate specialized agent.

    Parameters
    ----------
    storage : SearchResultsStorage
        Storage object for search results.
    model : str
        LLM model to use.
    temperature : float
        Temperature setting for the LLM.

    Returns
    -------
    Agent[Any]
        An instance of the research orchestrator agent.
    """
    model_settings = get_model_settings(
        model_name=model,
        temperature=temperature,
        max_tokens=2048,
        parallel_tool_calls=False
    )

    clarification_agent = make_clarification_agent(storage, model=model, temperature=temperature, enable_handoff=True)
    sequential_search_agent = make_sequential_search_agent(storage, model=model, temperature=temperature)

    handoffs = [
        clarification_agent,
        sequential_search_agent,
    ]

    instructions = (
        "You are a research orchestrator. Your primary role is to analyze the user's request "
        "and delegate the task to the most appropriate specialized agent.\n\n"
        "WORKFLOW:\n"
        "1. **On the first user request of a conversation, you MUST handoff to the Clarification Agent.** "
        "This agent will greet the user and handle the initial query.\n"
        "2. **For subsequent requests, analyze the user's intent:**\n"
        "   - If the request is vague, ambiguous, or needs more detail, handoff to the **Clarification Agent**.\n"
        "   - For research tasks that are clear enough, handoff to the **Sequential Search Agent**.\n\n"
        "Always handoff to one of the specialized agents. Do not attempt to answer the user directly."
    )

    return Agent[Any](
        name="Research Orchestrator",
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        handoffs=handoffs,
    ) 