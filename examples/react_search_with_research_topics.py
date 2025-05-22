#!/usr/bin/env python3
"""
Example script demonstrating the ReAct (Reasoning + Action) search pattern with research topics generation.

This script extends the basic ReAct search example by:
1. Creating a search agent that follows the ReAct pattern
2. Implementing chain-of-thought reasoning between search actions
3. Setting up tools for web search functionality
4. Adding a research topics generator that is available through agent handoff
5. Enabling bi-directional handoffs between agents to continue exploration of topics
6. Processing and displaying comprehensive research results with reasoning traces
7. Optionally enabling Langfuse tracing for observability

Usage:
    python react_search_with_research_topics.py "What are the latest advancements in quantum computing?"
    python react_search_with_research_topics.py --interactive
    python react_search_with_research_topics.py "How do large language models work?" --enable-langfuse

The interactive mode lets you ask multiple questions in a row.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# Import the react search agent (which now includes handoff to research topics agent)
from deep_research.agents.react_search_agent import load_react_search_agent
from deep_research.agents.research_topics_agent import load_research_topics_agent

async def run_query_with_bidirectional_handoff(query: str, enable_langfuse: bool = False):
    """Run a single query through the ReAct search agent with bi-directional handoffs."""
    logger.info(f"Processing query with bi-directional handoffs: {query}")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # First, set up research topics agent with search handoff capability
    logger.debug("Setting up research topics agent with search handoff")
    research_topics_agent, _ = load_research_topics_agent(
        enable_langfuse=enable_langfuse,
        with_search_handoff=True
    )
    
    # Then, set up the ReAct search agent with research topics handoff
    logger.debug("Setting up ReAct search agent with research topics handoff")
    _, run_search_agent = load_react_search_agent(
        enable_langfuse=enable_langfuse,
        with_research_topics_handoff=True
    )
    
    # Run the search query with handoff capability
    logger.info("Sending query to search agent with handoff capability")
    results = await run_search_agent(query)
    logger.info("Query completed successfully")
    
    return results

async def interactive_mode(enable_langfuse: bool = False):
    """Run the ReAct search agent with bi-directional handoffs in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Set up the research topics agent with search handoff
    logger.debug("Setting up research topics agent with search handoff")
    research_topics_agent, _ = load_research_topics_agent(
        enable_langfuse=enable_langfuse,
        with_search_handoff=True
    )
    
    # Set up the ReAct search agent with research topics handoff
    logger.debug("Setting up ReAct search agent with bi-directional handoffs")
    _, run_search_agent = load_react_search_agent(
        enable_langfuse=enable_langfuse,
        with_research_topics_handoff=True
    )
    
    print("ReAct Search Agent with Bi-directional Handoffs (type 'exit' to quit)")
    print("This agent can search for information, suggest related topics, and search those topics")
    
    query_count = 0
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Running ReAct search with chain-of-thought reasoning and bi-directional handoffs...")
        try:
            # Run search with handoff capability
            results = await run_search_agent(query)
            logger.info(f"Successfully completed query #{query_count}")
            
            print("\nAgent response:")
            print(results)
        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")
    
    logger.info(f"Interactive session ended after {query_count} queries")

async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run ReAct search agent with bi-directional handoffs")
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
        log_file = f"logs/react_search_topics_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting ReAct search with bi-directional handoffs example with log level: {args.log_level}")
    
    if args.enable_langfuse:
        logger.info("Langfuse tracing is enabled")
    
    if args.interactive:
        await interactive_mode(enable_langfuse=args.enable_langfuse)
    elif args.query:
        logger.info(f"Running in single query mode")
        result = await run_query_with_bidirectional_handoff(args.query, enable_langfuse=args.enable_langfuse)
        print(result)
    else:
        logger.warning("No query provided and not in interactive mode")
        parser.print_help()
    
    logger.info("ReAct search with bi-directional handoffs example completed")

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