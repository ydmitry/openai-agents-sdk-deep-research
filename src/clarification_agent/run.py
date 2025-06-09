#!/usr/bin/env python3
"""
Clarification Agent Runner

Standalone script for analyzing user requests and generating clarification questions.
Uses the make_clarification_agent factory to create agents with various collection methods.

Usage:
    python src/clarification_agent/run.py "Build me a web app"
    python src/clarification_agent/run.py "Create ML model" --output questions.txt
    python src/clarification_agent/run.py "Design database" --model gpt-4o --temperature 0.2
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
from src.clarification_agent.agent import make_clarification_agent

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
    Create a collector that appends clarification questions to a JSON file.

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

    def collector(questions: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "questions": questions
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

        logger.info(f"Collected questions to {output_file}")

    return collector


def create_text_file_collector(output_file: str) -> Callable[[str], None]:
    """
    Create a collector that appends clarification questions to a text file.

    Parameters
    ----------
    output_file : str
        Path to the output text file.

    Returns
    -------
    Callable
        Collection function that writes to the specified file.
    """
    def collector(questions: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CLARIFICATION QUESTIONS - {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{questions}\n")
            f.write(f"{'='*80}\n\n")

        logger.info(f"Collected questions to {output_file}")

    return collector


def create_memory_collector() -> tuple[Callable[[str], None], Callable[[], list[dict]]]:
    """
    Create a collector that stores clarification questions in memory.

    Returns
    -------
    tuple
        (collection_function, retrieval_function)
        - collection_function: Callable that stores results
        - retrieval_function: Callable that returns stored results
    """
    storage = []

    def collector(questions: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "questions": questions
        }
        storage.append(record)
        logger.info(f"Collected questions in memory (total: {len(storage)} records)")

    def retriever() -> list[dict]:
        return storage.copy()

    return collector, retriever


def create_console_collector() -> Callable[[str], None]:
    """
    Create a collector that prints clarification questions to console.

    Returns
    -------
    Callable
        Collection function that prints to stdout.
    """
    def collector(questions: str) -> None:
        print(f"\n{'='*60}")
        print("CLARIFICATION QUESTIONS:")
        print('='*60)
        print(questions)
        print('='*60)

    return collector


async def run_clarification_analysis(
    user_request: str,
    output_file: Optional[str] = None,
    output_format: str = "json",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    log_level: str = "INFO"
) -> str:
    """
    Analyze a user request and generate clarification questions.

    Args:
        user_request: The request to analyze for clarification
        output_file: Optional output file path
        output_format: Output format ("json" or "text")
        model: Model to use for the agent
        temperature: Temperature setting
        log_level: Logging level

    Returns:
        The clarification questions as a string
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
        logger.info(f"Questions will be saved to: {output_file}")
    else:
        collector = create_console_collector()
        logger.info("Using console collector (questions will be displayed)")

    # Create the clarification agent
    logger.info(f"Creating clarification agent with model: {model}")
    agent = make_clarification_agent(collector, model=model, temperature=temperature)

    # Run the agent
    logger.info(f"Analyzing user request: {user_request}")

    try:
        from agents import Runner
        result = await Runner.run(agent, user_request)

        logger.info("Clarification analysis completed successfully")
        return result.final_output

    except Exception as e:
        logger.error(f"Error running clarification agent: {str(e)}")
        raise


async def chat_mode(model: str = "gpt-4.1-mini", temperature: float = 0.3):
    """
    Run the clarification agent in chat mode, providing a conversational interface.

    This creates a simple chat-like experience where the user can describe requests
    and receive clarification questions to help refine their requirements.
    The conversation history is maintained to provide context between messages.
    """
    logger.info(f"Starting clarification chat mode with model: {model}")

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

    # Create a simple collector for chat mode that just returns the result
    chat_collector, retriever = create_memory_collector()

    # Initialize clarification agent for chat session
    logger.debug("Initializing clarification agent for chat session")
    agent = make_clarification_agent(chat_collector, model=model, temperature=temperature)

    print("\n================================")
    print("❓ Clarification Agent Chat")
    print("================================")
    print("Describe what you want to build or accomplish, and I'll ask clarifying questions.")
    print("Type 'exit' or 'quit' to end the chat.")
    print("================================\n")

    message_count = 0
    conversation_history = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nChat session ended. Goodbye!")
            logger.info("User ended chat session")
            break

        if not user_input.strip():
            print("Please describe your request or type 'exit' to quit.")
            continue

        message_count += 1
        logger.info(f"Processing chat message #{message_count}: {user_input}")

        print("\nAnalyzing your request for clarification...")
        try:
            # For the first message, start a new conversation
            if conversation_history is None:
                result = await Runner.run(agent, user_input, max_turns=5)
            else:
                # For subsequent messages, append to the conversation history
                new_input = conversation_history + [{"role": "user", "content": user_input}]
                result = await Runner.run(agent, new_input, max_turns=5)

            # Store conversation history for next turn
            conversation_history = result.to_input_list()

            logger.info(f"Successfully generated clarification questions for message #{message_count}")

            # Display only the final output
            print("\nClarification Questions:")
            print(result.final_output)
            print("\n--------------------------------")

        except Exception as e:
            logger.error(f"Error processing message #{message_count}: {str(e)}")
            print(f"Sorry, I encountered an error: {e}")

    logger.info(f"Chat session ended after {message_count} messages")


def main():
    """Main entry point for the clarification agent runner."""
    parser = argparse.ArgumentParser(
        description="Clarification Agent - Generate questions to clarify ambiguous requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Build me a web app"
  %(prog)s "Create a machine learning model" --output questions.txt
  %(prog)s "Design a database" --format text --output clarifications.txt
  %(prog)s "Develop mobile app" --model gpt-4o --temperature 0.2
  %(prog)s --chat  # Start interactive chat mode
        """
    )

    parser.add_argument(
        "request",
        nargs='?',
        help="The user request to analyze for clarification (optional in chat mode)"
    )
    
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start interactive chat mode"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path for saving questions"
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
        default=0.3,
        help="Temperature setting (default: 0.3)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Check if chat mode is requested
    if args.chat:
        try:
            asyncio.run(chat_mode(
                model=args.model,
                temperature=args.temperature
            ))
        except KeyboardInterrupt:
            logger.info("Chat session interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Chat session failed: {str(e)}")
            sys.exit(1)
    else:
        # Single analysis mode - require request argument
        if not args.request:
            parser.error("Request argument is required when not in chat mode. Use --chat for interactive mode.")
        
        # Run the clarification analysis
        try:
            asyncio.run(run_clarification_analysis(
                args.request,
                args.output,
                args.format,
                args.model,
                args.temperature,
                args.log_level
            ))
        except KeyboardInterrupt:
            logger.info("Clarification analysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Clarification analysis failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main() 