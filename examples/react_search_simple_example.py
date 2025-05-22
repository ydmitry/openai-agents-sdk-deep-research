#!/usr/bin/env python3
"""
Example script demonstrating the ReAct (Reasoning + Action) search pattern.

This script shows how to:
1. Create a search agent that follows the ReAct pattern
2. Implement chain-of-thought reasoning between search actions
3. Set up tools for web search functionality
4. Process and display comprehensive research results with reasoning traces
5. Optionally enable Langfuse tracing for observability

Usage:
    python react_search_example.py "What are the latest advancements in quantum computing?"
    python react_search_example.py --interactive
    python react_search_example.py "How do large language models work?" --enable-langfuse

The interactive mode lets you ask multiple questions in a row.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple
from pydantic import BaseModel, Field

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research.utils import load_dotenv_files

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration."""
    # Create logs directory if using file logging and directory doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Return the logger
    return root_logger

# Module logger
logger = logging.getLogger(__name__)

# Define reasoning step model to track chain of thought
class ReasoningStep(BaseModel):
    thought: str = Field(description="The agent's thought process")
    action: Optional[str] = Field(None, description="The action taken based on the thought")
    observation: Optional[str] = Field(None, description="The observation from the action")

def load_react_search_agent(
    enable_langfuse: bool = False,
    service_name: str = "react_search_agent",
) -> Tuple[Any, Callable[[str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that follows the ReAct pattern
        2. An async `run_agent(query)` coroutine to execute queries with reasoning chain

    The implementation uses ONLY the `agents` package (no direct
    `openai` import). All model access is delegated to the Agents SDK.
    """
    try:
        # Import the agents package (not OpenAI directly)
        from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Optional Langfuse tracing
    # ──────────────────────────────────────────────────────────────────────────
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse

            if setup_langfuse():
                logger.info("Langfuse tracing enabled for react search agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning(
                "Langfuse integration not found; install `pydantic-ai[logfire]` "
                "if you need tracing."
            )

    # List of reasoning steps for tracking chain of thought
    reasoning_steps = []

    # ──────────────────────────────────────────────────────────────────────────
    # Custom function tools to implement ReAct pattern
    # ──────────────────────────────────────────────────────────────────────────
    @function_tool
    async def add_reasoning_step(thought: str, action: Optional[str] = None, observation: Optional[str] = None) -> str:
        """
        Add a step to the reasoning chain.

        Args:
            thought: The agent's thought process
            action: The action taken based on the thought
            observation: The observation from the action

        Returns:
            Confirmation message
        """
        reasoning_steps.append(ReasoningStep(
            thought=thought,
            action=action,
            observation=observation
        ))
        return f"Added reasoning step #{len(reasoning_steps)}"

    @function_tool
    async def get_reasoning_chain() -> List[ReasoningStep]:
        """
        Get the current reasoning chain.

        Returns:
            The list of reasoning steps
        """
        return reasoning_steps

    @function_tool
    async def web_search(query: str) -> str:
        """
        Wrapper for the WebSearchTool: perform a web search and return the results as a JSON list.
        """
        # Create a helper search agent that returns URLs as JSON
        search_agent = Agent(
            name="SearchExecutor",
            instructions="""
            Perform a single web search for the input query and return a JSON list of up to 5 high-quality URLs.
            Respond with **only** the JSON array, e.g. ["https://example.com", ...].
            """,
            tools=[WebSearchTool(search_context_size="high")],
            model="gpt-4.1",
        )
        # Run the helper agent with a limited number of turns
        result = await Runner.run(search_agent, query, max_turns=3)
        return result.final_output

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent with the web_search wrapper tool
    # ──────────────────────────────────────────────────────────────────────────
    agent = Agent(
        name="ReAct Search Assistant",
        instructions="""
# System
You're an AI assistant that follows the ReAct (Reason + Act) pattern to perform a research task.
Follow this loop: Thought, Action, Observation.
- Thought: Describe your reasoning about the current state and next steps.
- Action: Use web_search to find information.
Add `Finish.` at the end when the research task is fully discovered.

IMPORTANT! Do at least 5 web searches to get a comprehensive answer.

────────────────────  FEW-SHOT EXAMPLES  ────────────────────

# EXAMPLE 1
USER: What are the latest advancements in quantum computing?
ASSISTANT: Thought: I need to find the latest advancements in quantum computing.
ASSISTANT: (action: web_search, query="latest advancements in quantum computing")
TOOL: Recent breakthroughs in quantum computing include improvements in qubit stability and quantum supremacy demonstrations by major tech companies.
ASSISTANT: Thought: I need more specific information about qubit technology advancements.
ASSISTANT: (action: web_search, query="recent qubit technology breakthroughs")
TOOL: Researchers have developed new methods for reducing quantum decoherence and error correction, with IBM and Google leading in superconducting qubit architecture.
ASSISTANT: Thought: I should find information about quantum algorithms development.
ASSISTANT: (action: web_search, query="quantum algorithm advancements 2023")
TOOL: New quantum algorithms for optimization problems have shown exponential speedups compared to classical alternatives.
ASSISTANT: Recent advancements in quantum computing include significant improvements in qubit stability with error rates reduced by a factor of 10 in the past year, quantum supremacy demonstrations processing tasks in minutes that would take classical supercomputers thousands of years, and new quantum algorithms showing exponential speedups for optimization problems in logistics and material science.
Finish.

# EXAMPLE 2
USER: What are the latest findings on climate change?
ASSISTANT: Thought: I need to research the latest findings on climate change.
ASSISTANT: (action: web_search, query="latest climate change findings scientific reports")
TOOL: Recent IPCC report shows accelerating warming trends and identifies previously unknown feedback mechanisms.
ASSISTANT: Thought: I should look for specific data on sea level rise projections.
ASSISTANT: (action: web_search, query="sea level rise projections latest research")
TOOL: New models project sea level rise between 0.5-1.3 meters by 2100 depending on emission scenarios.
ASSISTANT: Thought: I need information about extreme weather event correlation with climate change.
ASSISTANT: (action: web_search, query="extreme weather attribution studies climate change")
TOOL: Recent attribution studies show climate change has made certain extreme weather events 2-5 times more likely.
ASSISTANT: Latest climate change findings include: accelerating warming trends at 0.18°C per decade (up from previous 0.15°C estimates), projected sea level rise of 0.5-1.3 meters by 2100, attribution studies demonstrating climate change has made extreme weather events 2-5 times more likely, and new research identifying near-term tipping points in Arctic systems with global implications.
Finish.

──────────────────── END OF EXAMPLES ────────────────────────

When you receive a new question, do NOT repeat the examples. Apply the same pattern:
Thought → (action: web_search, query="...") → Process the information → New thought → etc.

# Guidelines

- Be thorough in your reasoning
- Break down complex queries into specific searchable components
- Critically evaluate the information you find
- Consider multiple perspectives and sources
- Always cite sources in your final answer
- Perform at least 5 web searches before providing your final answer
- Add "Finish." at the end of your response

Your final response should include:
1. A comprehensive answer to the original query
2. Citations for all sources used
        """,
        tools=[web_search],
        model="gpt-4.1",
        model_settings=ModelSettings(
            temperature=0.2,
            max_tokens=4000,
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper coroutine to execute a single query
    # ──────────────────────────────────────────────────────────────────────────
    async def run_agent(query: str):
        """
        Execute the agent for a single query with ReAct reasoning chain.
        Handles optional Langfuse tracing if enabled.
        """
        # Clear the reasoning steps from previous runs
        reasoning_steps.clear()

        logger.info(f"Running ReAct search query: {query}")

        async def _invoke() -> str:
            # Run the agent with higher max_turns to allow for multiple search-reasoning cycles
            result = await Runner.run(agent, query, max_turns=10)

            # Format the full response with the reasoning chain
            full_response = f"# Research Query: {query}\n\n"
            full_response += "## Chain of Reasoning:\n\n"

            for i, step in enumerate(reasoning_steps, 1):
                full_response += f"### Step {i}:\n"
                full_response += f"**Thought**: {step.thought}\n\n"

                if step.action:
                    full_response += f"**Action**: {step.action}\n\n"

                if step.observation:
                    full_response += f"**Observation**: {step.observation}\n\n"

            full_response += f"## Final Answer:\n\n{result.final_output}\n"

            logger.debug(f"Response with reasoning chain length: {len(full_response)}")
            return full_response

        # With tracing
        if enable_langfuse:
            try:
                from deep_research.langfuse_integration import create_trace

                session_id = f"react_search_{hash(query) % 10_000}"
                with create_trace(
                    name="ReAct-Search-Query",
                    session_id=session_id,
                    tags=["react_search", "chain_of_thought"],
                    environment=os.getenv("ENVIRONMENT", "development"),
                ) as span:
                    if span is not None:
                        span.set_attribute("input.value", query)
                        span.set_attribute("reasoning_steps.count", 0)  # will update later

                    output = await _invoke()

                    if span is not None:
                        span.set_attribute("output.value", output)
                        span.set_attribute("reasoning_steps.count", len(reasoning_steps))

                    return output
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse tracing failure: %s – proceeding normally", exc)

        # Without tracing
        return await _invoke()

    logger.info("ReAct search agent initialized")
    return agent, run_agent

async def run_query(query: str, enable_langfuse: bool = False):
    """Run a single query through the ReAct search agent."""
    logger.info(f"Processing single query: {query}")

    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the ReAct search agent with optional Langfuse tracing
    logger.debug(f"Loading ReAct search agent (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    _, run_agent = load_react_search_agent(enable_langfuse=enable_langfuse)

    # Run the query and return the result
    logger.info("Sending query to agent")
    result = await run_agent(query)
    logger.info("Query completed successfully")
    return result

async def interactive_mode(enable_langfuse: bool = False):
    """Run the ReAct search agent in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")

    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the ReAct search agent (only once for the session) with optional Langfuse tracing
    logger.debug("Loading ReAct search agent for interactive session")
    _, run_agent = load_react_search_agent(enable_langfuse=enable_langfuse)

    print("ReAct Search Agent (type 'exit' to quit)")

    query_count = 0
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break

        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Running ReAct search with chain-of-thought reasoning...")
        try:
            response = await run_agent(query)
            logger.info(f"Successfully completed interactive query #{query_count}")
            print("\nAgent response:")
            print(response)
        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")

    logger.info(f"Interactive session ended after {query_count} queries")

async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run ReAct search agent with chain-of-thought reasoning")
    parser.add_argument("query", nargs="?", help="The query to research")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--enable-langfuse", action="store_true",
                        help="Enable Langfuse tracing for observability")
    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive):
        # Auto-create log file for debug mode or interactive sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/react_search_{timestamp}.log"

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting ReAct search example with log level: {args.log_level}")

    if args.enable_langfuse:
        logger.info("Langfuse tracing is enabled")

    if args.interactive:
        await interactive_mode(enable_langfuse=args.enable_langfuse)
    elif args.query:
        logger.info(f"Running in single query mode")
        result = await run_query(args.query, enable_langfuse=args.enable_langfuse)
        print(result)
    else:
        logger.warning("No query provided and not in interactive mode")
        parser.print_help()

    logger.info("ReAct search example completed")

def main():
    """
    Main entry point that handles setting up the asyncio loop
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
