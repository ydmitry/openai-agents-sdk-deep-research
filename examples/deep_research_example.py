#!/usr/bin/env python3
"""
Example script demonstrating the deep research agent functionality.

This script shows how to:
1. Initialize a deep research agent with sequential thinking capabilities
2. Ask complex research questions requiring in-depth analysis
3. Process and display comprehensive research results
4. Optionally enable Langfuse tracing for observability

Usage:
    python deep_research_example.py "What are the environmental impacts of lithium mining?"
    python deep_research_example.py --interactive
    python deep_research_example.py "How does quantum computing affect cryptography?" --enable-langfuse
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

from deep_research.utils import load_dotenv_files, get_model_settings

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

def load_deep_research_agent(enable_langfuse: bool = False, service_name: str = "deep_research_agent"):
    """
    Factory function to create and return a deep research agent with sequential thinking.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name to use for Langfuse tracing
        
    Returns:
        tuple: (agent, run_agent_function) - The configured agent instance and a function to run queries.
    """
    import openai
    from agents import Agent, Runner, WebSearchTool, ModelSettings

    logger.info("Initializing deep research agent")

    # Set up Langfuse tracing if enabled
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse
            
            if setup_langfuse():
                logger.info("Langfuse tracing enabled for deep research agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning("Could not import langfuse_integration module. Langfuse tracing will not be enabled.")
            logger.warning("Make sure you have installed 'pydantic-ai[logfire]' package.")

    # Get API client
    async_client = openai.AsyncOpenAI()

    # Create an agent with deep research capabilities using sequential thinking
    agent = Agent(
        name="Deep Research Assistant",
        instructions="""
        You are an advanced deep research assistant that provides comprehensive, nuanced analysis 
        on complex topics. You combine web search with structured sequential thinking to deliver 
        in-depth, well-reasoned research.
        
        Follow these guidelines:
        1. For complex questions, use sequential thinking to break down the research process into steps.
           - First, identify the key aspects of the question that need investigation
           - Search for reliable sources on each aspect
           - Analyze and synthesize findings from multiple sources
           - Draw well-reasoned conclusions based on the evidence
        2. Search for multiple perspectives and scholarly sources on the topic.
        3. Evaluate the credibility of sources, distinguishing between factual information and opinions.
        4. Synthesize information across multiple sources to form a comprehensive view.
        5. Acknowledge limitations and gaps in available information.
        6. Structure your response with clear sections (background, analysis, implications).
        7. Always cite your sources thoroughly.
        8. Where appropriate, note areas where expert consensus exists or where opinions diverge.
        """,
        tools=[
            # Web search with high context for detailed results
            WebSearchTool(
                search_context_size='high'
            ),
            # Sequential thinking is implemented through detailed agent instructions
            # and the model's inherent reasoning capabilities, not as a separate tool
        ],
        # Use a powerful model suited for deep research
        model="gpt-4.1",
        # Configure model settings for deep analysis
        model_settings=get_model_settings(
            model_name="gpt-4.1",
            temperature=0.2,  # Lower temperature for more focused responses
            max_tokens=4000,  # Allow for comprehensive responses
        ),
    )

    logger.debug("Deep research agent created with WebSearchTool and GPT-4.1 model")

    async def run_agent(query: str):
        """Run the agent with the given query."""
        logger.info(f"Running deep research query: {query}")
        try:
            # If Langfuse is enabled, wrap the agent run with a trace context
            if enable_langfuse:
                try:
                    from deep_research.langfuse_integration import create_trace
                    
                    # Generate a session ID based on the first few chars of the query
                    session_id = f"deep_research_{hash(query) % 10000}"
                    
                    with create_trace(
                        name="Deep-Research-Query",
                        session_id=session_id,
                        tags=["deep_research"],
                        environment=os.environ.get("ENVIRONMENT", "development")
                    ) as span:
                        # Set input for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("input.value", query)
                            except Exception as e:
                                logger.warning(f"Could not set input attribute on span: {e}")
                        
                        # Use a higher max_turns value for deep research to allow for more extensive analysis
                        result = await Runner.run(
                            agent,
                            query,
                            max_turns=10,  # Allow more turns for thorough research
                        )
                        
                        # Set output for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("output.value", result.final_output)
                            except Exception as e:
                                logger.warning(f"Could not set output attribute on span: {e}")
                        
                        logger.info("Deep research completed successfully")
                        logger.debug(f"Response length: {len(result.final_output)}")
                        return result.final_output
                except ImportError as e:
                    logger.warning(f"Could not import create_trace: {e}")
                    logger.warning("Running without tracing.")
                except Exception as e:
                    logger.error(f"Error with Langfuse tracing: {e}")
                    logger.info("Continuing without tracing...")
            
            # Run normally without tracing if Langfuse failed or is disabled
            result = await Runner.run(
                agent,
                query,
                max_turns=10,  # Allow more turns for thorough research
            )
            logger.info("Deep research completed successfully")
            logger.debug(f"Response length: {len(result.final_output)}")
            return result.final_output
                
        except Exception as e:
            logger.error(f"Error running deep research agent: {str(e)}")
            raise

    logger.info("Deep research agent initialized and ready for queries")
    return agent, run_agent


async def run_query(query: str, enable_langfuse: bool = False):
    """Run a single query through the deep research agent."""
    logger.info(f"Processing single query: {query}")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the deep research agent with optional Langfuse tracing
    logger.debug(f"Loading deep research agent (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    # Run the query and return the result
    logger.info("Sending query to agent")
    print("Running deep research... This may take some time for thorough analysis.")
    result = await run_agent(query)
    logger.info("Query completed successfully")
    return result


async def interactive_mode(enable_langfuse: bool = False):
    """Run the deep research agent in interactive mode."""
    logger.info(f"Starting interactive mode (Langfuse tracing: {'enabled' if enable_langfuse else 'disabled'})")
    
    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")
    
    # Load the deep research agent (only once for the session) with optional Langfuse tracing
    logger.debug("Loading deep research agent for interactive session")
    _, run_agent = load_deep_research_agent(enable_langfuse=enable_langfuse)
    
    print("Deep Research Agent (type 'exit' to quit)")
    print("Note: Deep research takes more time than simple web searches for comprehensive analysis.")
    
    query_count = 0
    while True:
        query = input("\nEnter your research question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Conducting deep research... (this may take a few minutes for thorough analysis)")
        try:
            response = await run_agent(query)
            logger.info(f"Successfully completed interactive query #{query_count}")
            print("\nResearch results:")
            print(response)
        except Exception as e:
            logger.error(f"Error processing query #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")
    
    logger.info(f"Interactive session ended after {query_count} queries")


async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conduct deep research on complex topics")
    parser.add_argument("query", nargs="?", help="The research question to investigate")
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
        log_file = f"logs/deep_research_{timestamp}.log"
    
    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting deep research example with log level: {args.log_level}")
    
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
    
    logger.info("Deep research example completed")


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