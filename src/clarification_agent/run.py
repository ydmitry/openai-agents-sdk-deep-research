#!/usr/bin/env python3
"""
Clarification Agent Runner

Standalone script for analyzing user requests and generating clarification questions.
Questions are displayed to users, while search results (from handoff) are collected.

Usage:
    python src/clarification_agent/run.py "Build me a web app"
    python src/clarification_agent/run.py "Create ML model" --output results.txt
    python src/clarification_agent/run.py "Design database" --model gpt-4o --temperature 0.2
"""

import argparse
import asyncio
import logging
import sys
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.helpers import load_dotenv_files
from src.clarification_agent.agent import make_clarification_agent
from src.storage.search_results import (
    create_memory_storage,
    create_textfile_storage,
    create_postgres_storage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Display Callback Functions (for showing questions to users)
# ──────────────────────────────────────────────────────────────────────────




def create_file_display(output_file: str, format_type: str = "text") -> Callable[[str], None]:
    """
    Create a display callback that writes questions to a file for user review.

    Parameters
    ----------
    output_file : str
        Path to the output file.
    format_type : str
        Format type ("text" or "json")

    Returns
    -------
    Callable
        Display function that writes questions to the specified file.
    """
    def display(questions: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == "json":
            import json
            display_record = {
                "timestamp": timestamp,
                "questions": questions
            }
            
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
            data.append(display_record)
            
            # Write back to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # Text format
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"CLARIFICATION QUESTIONS - {timestamp}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{questions}\n")
                f.write(f"{'='*80}\n\n")
                
        logger.info(f"Displayed questions to {output_file}")

    return display


# ──────────────────────────────────────────────────────────────────────────
# Collection Callback Functions (replaced by storage system)
# ──────────────────────────────────────────────────────────────────────────


async def run_clarification_analysis(
    user_request: str,
    output_file: Optional[str] = None,
    output_format: str = "json",
    model: str = "gpt-4.1-mini",
    clarification_model: Optional[str] = None,
    temperature: float = 0.3,
    log_level: str = "INFO",
    enable_handoff: bool = False,
    use_postgres: bool = False
) -> str:
    """
    Analyze a user request and generate clarification questions.

    Args:
        user_request: The request to analyze for clarification
        output_file: Optional output file path
        output_format: Output format ("json" or "text")
        model: Model to use for the search agent in handoff scenarios
        clarification_model: Model to use specifically for the clarification agent
        temperature: Temperature setting
        log_level: Logging level
        enable_handoff: Whether to enable handoff to sequential search agent
        use_postgres: Whether to use PostgreSQL collector for storing search results

    Returns:
        The clarification questions as a string
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Load environment variables
    env_files = load_dotenv_files()
    if env_files:
        logger.info(f"Loaded environment from: {', '.join(env_files)}")

    # Create appropriate storage for search results (from handoff agent)
    session_id = None
    if use_postgres:
        storage, session_id = create_postgres_storage()
        logger.info(f"Using PostgreSQL storage for search results with session_id: {session_id}")
    elif output_file:
        storage = create_textfile_storage(output_file)
        logger.info(f"Search results will be saved to: {output_file}")
    else:
        # For console output, create a simple console collector callback
        class ConsoleStorage:
            def collect_search_results(self, results: str) -> None:
                print(f"\n{'='*60}")
                print("SEARCH RESULTS:")
                print('='*60)
                print(results)
                print('='*60)
            
            def get_citations(self) -> list[str]:
                return []  # Console storage doesn't persist citations
        
        storage = ConsoleStorage()
        logger.info("Using console storage for search results")

    # Create appropriate display callback for questions (file only, console shows naturally)
    if output_file:
        display_callback = create_file_display(output_file, output_format)
        logger.info(f"Questions will be displayed to: {output_file}")
    else:
        display_callback = None  # Let console output show naturally
        logger.info("Questions will be displayed on console via normal agent output")

    # Create the clarification agent
    clarification_model_info = f" (clarification: {clarification_model})" if clarification_model else ""
    logger.info(f"Creating clarification agent with model: {model}{clarification_model_info}, handoff: {enable_handoff}")
    agent = make_clarification_agent(
        storage, 
        model=model,
        clarification_model=clarification_model,
        temperature=temperature, 
        enable_handoff=enable_handoff,
        display_callback=display_callback
    )

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


async def chat_mode(model: str = "gpt-4.1-mini", clarification_model: Optional[str] = None, temperature: float = 0.3, enable_handoff: bool = True, log_level: str = "INFO", use_postgres: bool = False):
    """
    Run the clarification agent in chat mode, providing a conversational interface.

    This creates a simple chat-like experience where the user can describe requests
    and receive clarification questions to help refine their requirements.
    With handoff enabled, the agent can automatically transition to search mode.
    The conversation history is maintained to provide context between messages.
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    clarification_model_info = f" (clarification: {clarification_model})" if clarification_model else ""
    logger.info(f"Starting clarification chat mode with model: {model}{clarification_model_info}, handoff: {enable_handoff}")

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

    # Create appropriate storage for search results (from handoff agent)
    session_id = None
    if use_postgres:
        storage, session_id = create_postgres_storage()
        logger.info(f"Using PostgreSQL storage for search results with session_id: {session_id}")
    else:
        # Create a memory storage for search results (from handoff)
        storage = create_memory_storage()

    # Create display callback for questions in chat mode (let console show naturally)
    chat_display = None

    # Initialize clarification agent for chat session with optional handoff
    logger.debug(f"Initializing clarification agent for chat session (handoff: {enable_handoff})")
    agent = make_clarification_agent(
        storage, 
        model=model,
        clarification_model=clarification_model,
        temperature=temperature, 
        enable_handoff=enable_handoff,
        display_callback=chat_display
    )

    # Display appropriate header based on handoff capability and storage
    if enable_handoff:
        print("\n================================")
        print("❓🔍 Clarification → Search Chat")
        print("================================")
        print("Describe your request. I'll ask questions to clarify, then search for comprehensive answers.")
        if use_postgres and session_id:
            print(f"🗄️  Session ID: {session_id}")
            print("Search results will be stored in PostgreSQL with embeddings.")
        print("Type 'exit' or 'quit' to end the chat.")
        print("================================\n")
    else:
        print("\n================================")
        print("❓ Clarification Agent Chat")
        print("================================")
        print("Describe what you want to build or accomplish, and I'll ask clarifying questions.")
        if use_postgres and session_id:
            print(f"🗄️  Session ID: {session_id}")
            print("Search results will be stored in PostgreSQL with embeddings.")
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

        # Show appropriate processing message
        if enable_handoff:
            print("\nProcessing (clarification → search if ready)...")
        else:
            print("\nAnalyzing your request for clarification...")
        
        try:
            # For the first message, start a new conversation
            if conversation_history is None:
                result = await Runner.run(agent, user_input, max_turns=20)
            else:
                # For subsequent messages, append to the conversation history
                new_input = conversation_history + [{"role": "user", "content": user_input}]
                result = await Runner.run(agent, new_input, max_turns=20)

            # Store conversation history for next turn
            conversation_history = result.to_input_list()

            logger.info(f"Successfully processed message #{message_count}")

            # Display the result (could be clarification questions or search results)
            print(f"\nAssistant: {result.final_output}")
            print("\n" + "-"*50)

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
  %(prog)s "Create a machine learning model" --output results.txt
  %(prog)s "Design a database" --format text --output results.txt
  %(prog)s "Develop mobile app" --model gpt-4o --temperature 0.2
  %(prog)s "Analyze user needs" --clarification-model gpt-4o --model gpt-4.1-mini  # Different models for clarification vs search
  %(prog)s "Analyze user needs" --postgres  # Store search results in PostgreSQL with embeddings
  %(prog)s --chat  # Start interactive chat mode
  %(prog)s --chat --clarification-model gpt-4o --postgres  # Chat mode with specific clarification model and PostgreSQL storage
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
        help="Output file path for displaying questions and collecting search results"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Use PostgreSQL collector to store search results with embeddings"
    )

    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-mini",
        help="Model to use for search agent (default: gpt-4.1-mini)"
    )

    parser.add_argument(
        "--clarification-model", "-cm",
        help="Model to use specifically for clarification agent (default: uses --model value)"
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
    
    parser.add_argument(
        "--enable-handoff",
        action="store_true",
        default=True,
        help="Enable handoff to sequential search agent (default: True)"
    )
    
    parser.add_argument(
        "--disable-handoff",
        action="store_true",
        help="Disable handoff to sequential search agent"
    )

    args = parser.parse_args()

    # Determine handoff setting
    enable_handoff = args.enable_handoff and not args.disable_handoff

    # Check if chat mode is requested
    if args.chat:
        try:
            asyncio.run(chat_mode(
                model=args.model,
                clarification_model=args.clarification_model,
                temperature=args.temperature,
                enable_handoff=enable_handoff,
                log_level=args.log_level,
                use_postgres=args.postgres
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
                args.clarification_model,
                args.temperature,
                args.log_level,
                enable_handoff,
                args.postgres
            ))
        except KeyboardInterrupt:
            logger.info("Clarification analysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Clarification analysis failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main() 