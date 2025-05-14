#!/usr/bin/env python3
"""
Example script demonstrating the web search agent functionality.

This script shows how to:
1. Initialize the web search agent from the utils module
2. Ask questions that benefit from web search capabilities
3. Process and display the agent's responses
4. Optionally enable Langfuse tracing for observability

Usage:
    python web_search_example.py "What is the current status of Mars missions?"
    python web_search_example.py --interactive
    python web_search_example.py "How does GPT-4 work?" --enable-langfuse

The interactive mode lets you ask multiple questions in a row.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research.utils import load_web_search_agent, load_dotenv_files

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

async def run_query(query: str, enable_langfuse: bool = False):
    """Run a single query through the web search agent."""
    logger.info(f"Processing single query: {query}")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the web search agent with optional Langfuse tracing
    logger.debug(f"Loading web search agent (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    _, run_agent = load_web_search_agent(enable_langfuse=enable_langfuse)
    
    # Run the query and return the result
    logger.info("Sending query to agent")
    result = await run_agent(query)
    logger.info("Query completed successfully")
    return result


async def interactive_mode(enable_langfuse: bool = False):
    """Run the web search agent in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the web search agent (only once for the session) with optional Langfuse tracing
    logger.debug("Loading web search agent for interactive session")
    _, run_agent = load_web_search_agent(enable_langfuse=enable_langfuse)
    
    print("Web Search Agent (type 'exit' to quit)")
    
    query_count = 0
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Searching the web and generating response...")
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
    parser = argparse.ArgumentParser(description="Query the web search agent")
    parser.add_argument("query", nargs="?", help="The query to search for")
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
        log_file = f"logs/web_search_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting web search example with log level: {args.log_level}")
    
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
    
    logger.info("Web search example completed")


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