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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Collection Callback Functions
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
    import json

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

    # Create appropriate collector
    if output_file:
        if output_format.lower() == "text":
            collector = create_text_file_collector(output_file)
        else:
            collector = create_json_file_collector(output_file)
        logger.info(f"Results will be saved to: {output_file}")
    else:
        collector, _ = create_memory_collector()
        logger.info("Using memory collector (results will not be saved)")

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
