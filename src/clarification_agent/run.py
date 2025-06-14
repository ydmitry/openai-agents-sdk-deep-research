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

# Additional imports for postgres collector
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
# Collection Callback Functions (for storing search results only)
# ──────────────────────────────────────────────────────────────────────────

def create_json_file_collector(output_file: str) -> Callable[[str], None]:
    """
    Create a collector that appends search results to a JSON file.

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

    def collector(search_results: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "search_results": search_results
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

        logger.info(f"Collected search results to {output_file}")

    return collector


def create_text_file_collector(output_file: str) -> Callable[[str], None]:
    """
    Create a collector that appends search results to a text file.

    Parameters
    ----------
    output_file : str
        Path to the output text file.

    Returns
    -------
    Callable
        Collection function that writes to the specified file.
    """
    def collector(search_results: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"SEARCH RESULTS - {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{search_results}\n")
            f.write(f"{'='*80}\n\n")

        logger.info(f"Collected search results to {output_file}")

    return collector


def create_memory_collector() -> tuple[Callable[[str], None], Callable[[], list[dict]]]:
    """
    Create a collector that stores search results in memory.

    Returns
    -------
    tuple
        (collection_function, retrieval_function)
        - collection_function: Callable that stores results
        - retrieval_function: Callable that returns stored results
    """
    storage = []

    def collector(search_results: str) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "search_results": search_results
        }
        storage.append(record)
        logger.info(f"Collected search results in memory (total: {len(storage)} records)")

    def retriever() -> list[dict]:
        return storage.copy()

    return collector, retriever


def create_console_collector() -> Callable[[str], None]:
    """
    Create a collector that prints search results to console.

    Returns
    -------
    Callable
        Collection function that prints to stdout.
    """
    def collector(search_results: str) -> None:
        print(f"\n{'='*60}")
        print("SEARCH RESULTS:")
        print('='*60)
        print(search_results)
        print('='*60)

    return collector


def create_postgres_collector() -> tuple[Callable[[str], None], str]:
    """
    Create a collector that stores search results in PostgreSQL with embeddings.
    
    Automatically generates a session UUID and stores search results with 
    OpenAI embeddings in the search_results table.
    
    NOTE: This collector is only used for search results from handoff agents,
    NOT for clarification questions.

    Returns
    -------
    tuple
        (collection_function, session_id)
        - collection_function: Callable that stores search results in PostgreSQL
        - session_id: The generated session UUID for this collector
    
    Raises
    ------
    ImportError
        If required dependencies (psycopg2) are not available
    """
    # Check dependencies
    if not PSYCOPG2_AVAILABLE:
        raise ImportError("psycopg2 is required for PostgreSQL collector. Install with: pip install psycopg2-binary")
    
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI library not available - embeddings will be skipped")
    # Database configuration (same as demo_pgvector.py)
    DB_CONFIG = {
        'host': '127.0.0.1',
        'port': 5433,
        'database': 'my-deep-research',
        'user': 'postgres',
        'password': 'secret'
    }
    
    # Generate session UUID
    session_id = str(uuid.uuid4())
    logger.info(f"Created postgres collector with session_id: {session_id}")
    
    # Initialize OpenAI client (reuse from environment)
    openai_client = None
    if OPENAI_AVAILABLE:
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for embedding generation")
            else:
                logger.warning("OPENAI_API_KEY not found - embeddings will be skipped")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e} - embeddings will be skipped")
    else:
        logger.warning("OpenAI library not available - embeddings will be skipped")
    
    def get_embedding(text: str) -> Optional[list[float]]:
        """Get embedding for text using OpenAI API."""
        if not openai_client:
            return None
            
        try:
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for text (length: {len(text)}): {e}")
            return None
    
    def connect_to_db():
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def collector(search_results: str) -> None:
        """Store search results in PostgreSQL with embeddings."""
        try:
            # Connect to database
            conn = connect_to_db()
            if not conn:
                logger.error("Failed to connect to database - search results not stored")
                return
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get embedding for the search results
                    embedding = get_embedding(search_results)
                    
                    # Insert search results (with or without embedding)
                    cur.execute("""
                        INSERT INTO search_results (session_id, text, embedding)
                        VALUES (%s, %s, %s)
                        RETURNING id, created_at;
                    """, (session_id, search_results, embedding))
                    
                    result = cur.fetchone()
                    conn.commit()
                    
                    embedding_status = "with embedding" if embedding else "without embedding"
                    logger.info(
                        f"Stored search results to database (ID: {result['id']}, "
                        f"Session: {session_id[:8]}..., {embedding_status})"
                    )
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error storing search results to database: {e}")
    
    return collector, session_id


async def run_clarification_analysis(
    user_request: str,
    output_file: Optional[str] = None,
    output_format: str = "json",
    model: str = "gpt-4.1-mini",
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
        model: Model to use for the agent
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

    # Create appropriate collector for search results (from handoff agent)
    session_id = None
    if use_postgres:
        collector, session_id = create_postgres_collector()
        logger.info(f"Using PostgreSQL collector for search results with session_id: {session_id}")
    elif output_file:
        if output_format.lower() == "text":
            collector = create_text_file_collector(output_file)
        else:
            collector = create_json_file_collector(output_file)
        logger.info(f"Search results will be saved to: {output_file}")
    else:
        collector = create_console_collector()
        logger.info("Using console collector for search results")

    # Create appropriate display callback for questions (file only, console shows naturally)
    if output_file:
        display_callback = create_file_display(output_file, output_format)
        logger.info(f"Questions will be displayed to: {output_file}")
    else:
        display_callback = None  # Let console output show naturally
        logger.info("Questions will be displayed on console via normal agent output")

    # Create the clarification agent
    logger.info(f"Creating clarification agent with model: {model}, handoff: {enable_handoff}")
    agent = make_clarification_agent(
        collector, 
        model=model, 
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


async def chat_mode(model: str = "gpt-4.1-mini", temperature: float = 0.3, enable_handoff: bool = True, log_level: str = "INFO", use_postgres: bool = False):
    """
    Run the clarification agent in chat mode, providing a conversational interface.

    This creates a simple chat-like experience where the user can describe requests
    and receive clarification questions to help refine their requirements.
    With handoff enabled, the agent can automatically transition to search mode.
    The conversation history is maintained to provide context between messages.
    """
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    logger.info(f"Starting clarification chat mode with model: {model}, handoff: {enable_handoff}")

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

    # Create appropriate collector for search results (from handoff agent)
    session_id = None
    if use_postgres:
        chat_collector, session_id = create_postgres_collector()
        logger.info(f"Using PostgreSQL collector for search results with session_id: {session_id}")
    else:
        # Create a memory collector for search results (from handoff)
        chat_collector, _ = create_memory_collector()

    # Create display callback for questions in chat mode (let console show naturally)
    chat_display = None

    # Initialize clarification agent for chat session with optional handoff
    logger.debug(f"Initializing clarification agent for chat session (handoff: {enable_handoff})")
    agent = make_clarification_agent(
        chat_collector, 
        model=model, 
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
  %(prog)s "Analyze user needs" --postgres  # Store search results in PostgreSQL with embeddings
  %(prog)s --chat  # Start interactive chat mode
  %(prog)s --chat --postgres  # Chat mode with PostgreSQL storage for search results
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