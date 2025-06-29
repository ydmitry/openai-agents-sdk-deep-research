#!/usr/bin/env python3
"""
Research Orchestrator Runner

This script provides a chat interface to the Research Orchestrator agent,
which manages the entire research workflow.

Usage:
    python src/research_orchestrator/run.py
    python src/research_orchestrator/run.py --model gpt-4o
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.helpers import load_dotenv_files
from src.research_orchestrator.agent import make_research_orchestrator
from src.storage.search_results import (
    create_memory_storage,
    create_postgres_storage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def chat_mode(model: str = "gpt-4.1-mini", temperature: float = 0.3, log_level: str = "INFO", postgres: bool = False):
    """
    Run the research orchestrator in chat mode.
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    logger.info(f"Starting research orchestrator chat mode with model: {model}")

    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Import Runner here to ensure it's available
    try:
        from agents import Runner
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # Create appropriate storage for search results
    session_id = None
    if postgres:
        storage, session_id = create_postgres_storage()
        logger.info(f"Using PostgreSQL storage for search results with session_id: {session_id}")
    else:
        # Create a memory storage for search results
        storage = create_memory_storage()

    # Initialize the research orchestrator agent
    logger.debug("Initializing research orchestrator agent for chat session")
    agent = make_research_orchestrator(
        storage,
        model=model,
        temperature=temperature,
    )

    print("\nWelcome to the Research Assistant!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Let's start with your research request.")

    conversation_history = None  # Initialize conversation history

    # Main chat loop
    while True:
        try:
            user_request = input("\n> ")
            if user_request.lower() in ["exit", "quit"]:
                print("Thank you for using the Research Assistant. Goodbye!")
                break

            if not user_request:
                continue

            # For the first message, start a new conversation
            if conversation_history is None:
                result = await Runner.run(agent, user_request)
            else:
                # For subsequent messages, append to the conversation history
                new_input = conversation_history + [{"role": "user", "content": user_request}]
                result = await Runner.run(agent, new_input)
            
            # Store conversation history for next turn
            conversation_history = result.to_input_list()

            # Display the final output to the user
            final_output = result.final_output
            if final_output:
                print(f"\nAssistant:\n{final_output}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nConversation interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during the chat session: {e}", exc_info=True)
            print("Sorry, an error occurred. Please try again.")


def main():
    """Main entry point for the research orchestrator runner."""
    parser = argparse.ArgumentParser(
        description="Run the Research Orchestrator agent in chat mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model to use for the orchestrator and specialized agents."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature setting for the LLM."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Use PostgreSQL for storing search results instead of in-memory."
    )

    args = parser.parse_args()

    try:
        asyncio.run(chat_mode(
            model=args.model,
            temperature=args.temperature,
            log_level=args.log_level,
            postgres=args.postgres
        ))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 