#!/usr/bin/env python3
"""
WebSearch Agent Factory

Factory function that creates a search agent with collection capabilities.
The agent performs web searches and forwards results to a user-supplied callback.

Usage:
    from websearch_agent.agent import make_search_agent

    def my_collector(answer: str):
        print(f"Answer: {answer}\n")

    agent = make_search_agent(my_collector)
"""

import logging
from typing import Any
from collections.abc import Callable as ABCCallable

from agents import (
    Agent,
    WebSearchTool,
    ModelSettings,
    AgentHooks,
    RunContextWrapper,
)

# Import helper functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings
from storage.search_results import SearchResultsStorage

logger = logging.getLogger(__name__)


class SearchCollectionHooks(AgentHooks[Any]):
    """Lifecycle hooks that automatically collect results when the agent completes."""

    def __init__(self, storage: SearchResultsStorage):
        self.storage = storage

    async def on_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
    ) -> None:
        """Called before the agent is invoked - capture the initial query from conversation."""
        # Try to extract the query from the conversation history
        try:
            # The input is typically the last user message
            # We'll get this from the agent's conversation or context if available
            # For now, we'll leave it empty and handle in on_end
            pass
        except Exception as e:
            logger.debug(f"Could not extract query in on_start: {str(e)}")

    async def on_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output - automatically collect the result."""
        try:
            answer = str(output) if output else ""

            # Store the result using storage object
            self.storage.collect_search_results(answer)
            logger.info(f"Automatically stored result (answer length: {len(answer)} chars)")

        except Exception as e:
            # Log errors but don't interrupt the agent's response
            logger.error(f"Error in collection hook: {str(e)}")


def make_search_agent(
    storage: SearchResultsStorage,
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> Agent[Any]:
    """
    Create a search agent that automatically stores answers via lifecycle hooks.

    Parameters
    ----------
    storage : SearchResultsStorage
        Storage object that receives each answer.
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
        max_tokens=4096,  # Standardized to match sequential agent
        parallel_tool_calls=False
    )

    # Create the hooks internally
    collection_hooks = SearchCollectionHooks(storage)

    return Agent[Any](
        name="Search & Collect Agent",
        instructions=(
            "You are a helpful search agent. For every user request:\n"
            "1. Use the web_search tool to gather comprehensive, up-to-date information.\n"
            "2. Analyze and synthesize the search results.\n"
            "3. Provide a thorough, well-structured answer based on the search findings.\n"
            "Make sure your response is informative, accurate, and directly addresses the user's query.\n"
            "IMPORTANT! Duplicate user's input in your response in first paragraph."
        ),
        tools=[
            WebSearchTool(search_context_size="high"),   # Only need the search tool
        ],
        model=model,
        model_settings=model_settings,
        hooks=collection_hooks,  # Attach the collection hooks
    )
