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

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Optional
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


# ──────────────────────────────────────────────────────────────────────────
# Collection Callback Examples
# ──────────────────────────────────────────────────────────────────────────

def create_json_file_collector(output_file: str) -> Callable[[str], None]:
    """
    Create a collector that appends answers to a JSON file.
    
    Parameters
    ----------
    output_file : str
        Path to the output JSON file.
        
    Returns
    -------
    Callable
        Collection function that writes to the specified file.
    """
    def collector(answer: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "answer": answer
        }
        
        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or create new list
        try:
            if Path(output_file).exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
        
        # Append new record
        data.append(record)
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Collected result to {output_file}")
    
    return collector


def create_text_file_collector(output_file: str) -> Callable[[str], None]:
    """
    Create a collector that appends answers to a text file.
    
    Parameters
    ----------
    output_file : str
        Path to the output text file.
        
    Returns
    -------
    Callable
        Collection function that writes to the specified file.
    """
    def collector(answer: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"ANSWER:\n{answer}\n")
            f.write(f"{'='*80}\n\n")
        
        logger.info(f"Collected result to {output_file}")
    
    return collector


def create_memory_collector() -> tuple[Callable[[str], None], Callable[[], list[dict]]]:
    """
    Create a collector that stores answers in memory.
    
    Returns
    -------
    tuple
        (collection_function, retrieval_function)
        - collection_function: Callable that stores results
        - retrieval_function: Callable that returns stored results
    """
    storage = []
    
    def collector(answer: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "answer": answer
        }
        storage.append(record)
        logger.info(f"Collected result in memory (total: {len(storage)} records)")
    
    def retriever() -> list[dict]:
        return storage.copy()
    
    return collector, retriever


# ──────────────────────────────────────────────────────────────────────────
# Quick Demo
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    
    # Add the src directory to Python path for environment loading
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Load environment variables
    try:
        from deep_research.utils import load_dotenv_files
        load_dotenv_files()
    except ImportError:
        print("Note: Could not load environment files. Make sure API keys are set.")
    
    # Create memory collector for demo
    collector, get_results = create_memory_collector()
    
    async def demo() -> None:
        from agents import Runner
        
        print("Creating search agent with lifecycle-based collection...")
        agent = make_search_agent(collector)
        
        query = "What are the main exports of Ukraine in 2024?"
        print(f"Running search query: {query}")
        
        # No context needed - just run the agent directly
        result = await Runner.run(agent, query)
        
        print("\nAGENT RESPONSE:")
        print(result.final_output)
        
        print("\nCOLLECTED RESULTS:")
        results = get_results()
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("WebSearch Agent Factory Demo (Lifecycle-based Collection)")
    print("="*60)
    asyncio.run(demo()) 