#!/usr/bin/env python3
"""
Example script demonstrating the deep research agent functionality.

This script shows how to:
1. Initialize a deep research system with multiple specialized agents
2. Ask complex research questions requiring in-depth analysis
3. Use a triage agent to coordinate between search and analysis agents
4. Enable iterative refinement of research through multiple search-analyze cycles
5. Process and display comprehensive research results
6. Optionally enable Langfuse tracing for observability

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
    Factory function to create and return a deep research system with multiple specialized agents.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name to use for Langfuse tracing
        
    Returns:
        tuple: (agent, run_agent_function) - The configured triage agent and a function to run queries.
    """
    import openai
    from agents import Agent, Runner, WebSearchTool, ModelSettings
    from agents import handoff

    logger.info("Initializing deep research agent system")

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

    # Define shared model settings for more consistency
    powerful_model_settings = get_model_settings(
        model_name="gpt-4.1",
        temperature=0.2,  # Lower temperature for more focused responses
        max_tokens=4000,  # Allow for comprehensive responses
    )
    
    lighter_model_settings = get_model_settings(
        model_name="gpt-4.1",
        temperature=0.1,  # Even lower temperature for coordination
        max_tokens=1000,  # Less tokens needed for coordination
    )

    # Create specialized search agent
    search_agent = Agent(
        name="Research Search Agent",
        instructions="""
        You are a specialized search agent focused on finding comprehensive information 
        for deep research queries.
        
        Follow these guidelines:
        1. Identify key aspects of the research question that need investigation
        2. Formulate effective search queries to find relevant information
        3. Search for diverse and reliable sources on each aspect of the topic
        4. Prioritize academic and authoritative sources
        5. Gather information from multiple perspectives
        6. Collect both factual information and different viewpoints
        7. Document sources thoroughly for each piece of information
        8. Be thorough in your search to ensure comprehensive coverage
        
        Remember, your primary role is to GATHER information, not analyze it.
        Always provide full context and sources for the analysis agent to work with.
        """,
        tools=[
            # Web search with high context for detailed results
            WebSearchTool(
                search_context_size='high'
            ),
        ],
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Search agent created with WebSearchTool")

    # Create specialized analysis agent
    analysis_agent = Agent(
        name="Research Analysis Agent",
        instructions="""
        You are a specialized analysis agent that synthesizes and evaluates research information.
        
        Follow these guidelines:
        1. Carefully analyze the information provided by the search agent
        2. Synthesize findings across multiple sources to form a comprehensive view
        3. Evaluate the credibility and relevance of each source
        4. Identify patterns, contradictions, and gaps in the available information
        5. Draw well-reasoned conclusions based on the evidence
        6. Structure your analysis with clear sections (background, analysis, implications)
        7. Note areas where expert consensus exists or where opinions diverge
        8. Highlight limitations and uncertainties in the available information
        9. Identify areas that would benefit from additional research
        
        Your role is to ANALYZE and SYNTHESIZE information, not to search for more content.
        Provide a thorough analysis with well-supported conclusions and identify any gaps
        that might require additional search.
        """,
        model="gpt-4.1",
        model_settings=powerful_model_settings,
    )
    
    logger.debug("Analysis agent created")

    # Create triage agent that coordinates between search and analysis
    triage_agent = Agent(
        name="Deep Research Coordinator",
        instructions="""
        You are a research coordinator that orchestrates the deep research process by 
        managing specialized search and analysis agents.
        
        Follow these guidelines:
        1. Start by handing off to the search agent to gather initial information
        2. Then, hand off to the analysis agent to evaluate and synthesize the findings
        3. Review the analysis to determine if additional research is needed:
           - Are there important aspects of the question not yet addressed?
           - Are there contradictions that need resolution?
           - Are there knowledge gaps that require more information?
        4. If more research is needed, formulate specific follow-up questions and hand off
           again to the search agent with these targeted queries
        5. Continue this iterative process until you have a comprehensive understanding
        6. When sufficient research has been conducted, prepare a final synthesis that:
           - Provides a thorough answer to the original question
           - Presents multiple perspectives where relevant
           - Acknowledges limitations of the research
           - Cites sources appropriately
        
        For complex topics, you should perform multiple iterations of search and analysis.
        Simple topics may require only one or two iterations.
        
        Your goal is to produce the most comprehensive and well-reasoned research possible.
        """,
        model="gpt-4.1",
        model_settings=lighter_model_settings,
        handoffs=[
            handoff(search_agent),
            handoff(analysis_agent),
        ],
    )
    
    logger.debug("Triage agent created with handoffs to search and analysis agents")

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
                        # and multiple iterations of search-analyze cycles
                        result = await Runner.run(
                            triage_agent,
                            query,
                            max_turns=20,  # Increased max_turns to allow for multiple iterations
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
                triage_agent,
                query,
                max_turns=20,  # Increased max_turns to allow for multiple iterations
            )
            logger.info("Deep research completed successfully")
            logger.debug(f"Response length: {len(result.final_output)}")
            return result.final_output
                
        except Exception as e:
            logger.error(f"Error running deep research agent: {str(e)}")
            raise

    logger.info("Deep research agent system initialized and ready for queries")
    return triage_agent, run_agent


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
    print("Running deep research... This may take some time for thorough analysis and multiple search iterations.")
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
    print("Note: Deep research takes more time than simple searches for comprehensive analysis with multiple iterations.")
    
    query_count = 0
    while True:
        query = input("\nEnter your research question: ")
        if query.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break
        
        query_count += 1
        logger.info(f"Processing interactive query #{query_count}: {query}")
        print("Conducting deep research... (this may take several minutes for thorough analysis and multiple search iterations)")
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