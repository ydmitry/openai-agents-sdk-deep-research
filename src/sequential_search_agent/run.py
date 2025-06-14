#!/usr/bin/env python3
"""
Sequential Search Agent Runner

Standalone script for running sequential search queries with result collection.
Uses the make_sequential_search_agent factory to create agents with various collection methods.

Usage:
    python examples/sequential_search_agent/run.py "Your complex research query"
    python examples/sequential_search_agent/run.py "Compare AI models 2024" --output results.json
    python examples/sequential_search_agent/run.py "History and future of renewable energy" --format text --output research_log.txt
    python examples/sequential_search_agent/run.py "Pros and cons of electric vehicles" --model gpt-4o --temperature 0.3
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.helpers import load_dotenv_files
from src.sequential_search_agent.agent import make_sequential_search_agent
from src.storage.search_results import (
    create_memory_storage,
    create_textfile_storage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Collection Callback Functions (replaced by storage system)
# ──────────────────────────────────────────────────────────────────────────


async def run_sequential_search_query(
    query: str,
    output_file: Optional[str] = None,
    output_format: str = "json",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    log_level: str = "INFO"
) -> str:
    """
    Run a sequential search query with the sequential search agent.

    Args:
        query: The search query to execute
        output_file: Optional output file path
        output_format: Output format ("json" or "text")
        model: Model to use for the agent
        temperature: Temperature setting
        log_level: Logging level

    Returns:
        The agent's response
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Load environment variables
    env_files = load_dotenv_files()
    if env_files:
        logger.info(f"Loaded environment from: {', '.join(env_files)}")

    # Create appropriate storage
    if output_file:
        storage = create_textfile_storage(output_file)
        collector = storage.collect_search_results
        logger.info(f"Results will be saved to: {output_file}")
    else:
        storage = create_memory_storage()
        collector = storage.collect_search_results
        logger.info("Using memory storage (results will not be saved)")

    # Create the sequential search agent
    logger.info(f"Creating sequential search agent with model: {model}")
    agent = make_sequential_search_agent(collector, model=model, temperature=temperature)

    # Run the agent
    logger.info(f"Executing sequential search query: {query}")

    try:
        from agents import Runner
        result = await Runner.run(agent, query)

        logger.info("Sequential search completed successfully")
        return result.final_output

    except Exception as e:
        logger.error(f"Error running sequential search agent: {str(e)}")
        raise


def main():
    """Main entry point for the sequential search agent runner."""
    parser = argparse.ArgumentParser(
        description="Sequential Search Agent - Run complex research queries with multiple strategic searches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Compare renewable energy vs fossil fuels"
  %(prog)s "History and future of AI development" --output research.json
  %(prog)s "Pros and cons of remote work" --format text --output analysis.txt
  %(prog)s "Market analysis of electric vehicles" --model gpt-4o --temperature 0.1
        """
    )

    parser.add_argument(
        "query",
        help="The research query to execute (can be complex, requiring multiple searches)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path for saving results"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-mini",
        help="Model to use (default: gpt-4.1-mini)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.2,
        help="Temperature setting (default: 0.2)"
    )

    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Run the sequential search query
    try:
        result = asyncio.run(run_sequential_search_query(
            query=args.query,
            output_file=args.output,
            output_format=args.format,
            model=args.model,
            temperature=args.temperature,
            log_level=args.log_level
        ))

        # Print the result
        print("\n" + "="*80)
        print("SEQUENTIAL SEARCH RESULTS")
        print("="*80)
        print(result)
        print("="*80)

        if args.output:
            print(f"\nResults also saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
