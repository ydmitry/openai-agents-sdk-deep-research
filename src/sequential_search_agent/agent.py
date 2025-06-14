#!/usr/bin/env python3
"""
Sequential Search Agent Factory

Factory function that creates a sequential search agent with collection capabilities.
The agent uses websearch agents as tools to perform complex, multi-step research.

Usage:
    from sequential_search_agent.agent import make_sequential_search_agent

    def my_collector(answer: str):
        print(f"Answer: {answer}\n")

    agent = make_sequential_search_agent(my_collector)
"""

import logging
import asyncio
from typing import Any, Dict, List
from collections.abc import Callable as ABCCallable

from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
)

# Import helper functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_model_settings

# Import the websearch agent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from websearch_agent.agent import make_search_agent

logger = logging.getLogger(__name__)


def create_websearch_tool(collect_callback: ABCCallable[[str], None], model: str = "gpt-4.1-mini", temperature: float = 0.2):
    """
    Create a websearch function tool that wraps the websearch agent.
    """
    # Create the websearch agent with the same collect callback
    # This allows the websearch agent to call the collect callback directly when it completes
    websearch_agent = make_search_agent(
        collect_callback,
        model=model,
        temperature=temperature
    )

    @function_tool
    async def websearch(query: str) -> str:
        """
        Perform web searches using an intelligent search agent. Use this to gather information from the internet on any topic.

        Args:
            query: The search query to execute

        Returns:
            The search results as a string
        """
        try:
            logger.info(f"WebSearch tool executing search: {query}")

            # Run the websearch agent (it will call the collect callback when done)
            result = await Runner.run(websearch_agent, query)

            # Return the search results for the sequential agent to use
            search_result = result.final_output if result.final_output else ""
            logger.info(f"WebSearch tool completed search (result length: {len(search_result)} chars)")
            print(f"WebSearch: {query}\n")

            return search_result

        except Exception as e:
            error_msg = f"Error in WebSearch tool: {str(e)}"
            logger.error(error_msg)
            return f"Search failed: {error_msg}"

    return websearch





def make_sequential_search_agent(
    collect: ABCCallable[[str], None],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> Agent[Any]:
    """
    Create a sequential search agent that uses websearch agents as tools.

    Parameters
    ----------
    collect : (answer:str) -> None
        Callback that receives the final comprehensive answer.
    model : str
        LLM model to use for the agent.
    temperature : float
        Temperature setting for the LLM.

    Returns
    -------
    Agent[Any]
        Ready-to-use Agents-SDK agent instance with sequential search capabilities.
    """

    # Configure model settings using helper function
    model_settings = get_model_settings(
        model_name=model,
        temperature=temperature,
        max_tokens=4096,
        parallel_tool_calls=False
    )

    # Create the websearch tool that wraps the websearch agent
    # The websearch agent will call the collect callback directly when it completes
    websearch_tool = create_websearch_tool(collect, model=model, temperature=temperature)

    return Agent[Any](
        name="Sequential Search & Research Agent",
        instructions=(
            "You are an intelligent sequential search and research agent. Your role is to perform comprehensive research by conducting multiple strategic web searches when needed.\n\n"

            "For every user request:\n"
            "1. **Analyze the Query**: Determine if the request requires a single search or multiple sequential searches to gather comprehensive information.\n"
            "2. **Plan Your Research**: break down the query into logical search components or related topics.\n"
            "3. **Execute Searches**: Use the websearch tool to gather information. Always perform multiple searches to:\n"
            "   - Get comprehensive coverage of a broad topic\n"
            "   - Compare different perspectives or sources\n"
            "   - Gather background information first, then dive into specifics\n"
            "   - Follow up on interesting findings from initial searches\n"
            "4. **Synthesize Results**: Combine information from all searches into a comprehensive, well-structured answer.\n"
            "5. **Provide Context**: Include relevant context, comparisons, and analysis based on your research.\n\n"

            "Examples of multiple searches:\n"
            "- 'Compare X vs Y' → Search for X, then search for Y, then compare\n"
            "- 'Pros and cons of X' → Search for benefits, then search for drawbacks\n"
            "- 'History and current status of X' → Search for historical context, then current developments\n\n"

            "Always aim to provide thorough, accurate, and well-researched answers. Use multiple searches strategically to ensure comprehensive coverage."
        ),
        tools=[websearch_tool],
        model=model,
        model_settings=model_settings,
    )
