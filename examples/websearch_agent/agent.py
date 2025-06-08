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

logger = logging.getLogger(__name__)


class SearchCollectionHooks(AgentHooks[Any]):
    """Lifecycle hooks that automatically collect results when the agent completes."""
    
    def __init__(self, collect_callback: ABCCallable[[str], None]):
        self.collect_callback = collect_callback
    
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
            
            # Call the collection callback with just the answer
            self.collect_callback(answer)
            logger.info(f"Automatically collected result (answer length: {len(answer)} chars)")
            
        except Exception as e:
            # Log errors but don't interrupt the agent's response
            logger.error(f"Error in collection hook: {str(e)}")


def make_search_agent(
    collect: ABCCallable[[str], None],
    *,
    model: str = "gpt-4.1",
    temperature: float = 0.2,
) -> Agent[Any]:
    """
    Create a search agent that automatically collects answers via lifecycle hooks.
    
    Parameters
    ----------
    collect : (answer:str) -> None
        Callback that receives each answer.
    model : str
        LLM model to use for the agent.
    temperature : float
        Temperature setting for the LLM.
        
    Returns
    -------
    Agent[Any]
        Ready-to-use Agents-SDK agent instance with automatic collection.
    """

    # Configure model settings
    model_settings_kwargs = {"max_tokens": 2048}
    if not (model.startswith("o3") or model.startswith("o1") or model.startswith("o4")):
        model_settings_kwargs["temperature"] = temperature

    if model.startswith("o4"):
        model_settings_kwargs["reasoning_effort"] = "high"

    try:
        model_settings = ModelSettings(**model_settings_kwargs)
    except TypeError:
        # Remove unsupported parameters if needed
        if "reasoning_effort" in model_settings_kwargs:
            model_settings_kwargs.pop("reasoning_effort")
        model_settings = ModelSettings(**model_settings_kwargs)

    # Create the hooks internally
    collection_hooks = SearchCollectionHooks(collect)

    return Agent[Any](
        name="Search & Collect Agent",
        instructions=(
            "You are a helpful search agent. For every user request:\n"
            "1. Use the web_search tool to gather comprehensive, up-to-date information.\n"
            "2. Analyze and synthesize the search results.\n"
            "3. Provide a thorough, well-structured answer based on the search findings.\n"
            "Make sure your response is informative, accurate, and directly addresses the user's query."
        ),
        tools=[
            WebSearchTool(search_context_size="high"),   # Only need the search tool
        ],
        model=model,
        model_settings=model_settings,
        hooks=collection_hooks,  # Attach the collection hooks
    ) 