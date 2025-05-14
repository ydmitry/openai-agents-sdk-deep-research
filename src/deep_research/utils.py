"""
Utility functions for the deep_research package.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any
from agents import ModelSettings

# Configure logging
logger = logging.getLogger(__name__)

def find_dotenv_files(start_dir: Optional[str] = None, max_levels_up: int = 3) -> List[str]:
    """
    Find .env files in the current directory and parent directories.

    Args:
        start_dir: Directory to start searching from. Defaults to current directory.
        max_levels_up: Maximum number of parent directories to search.

    Returns:
        List of paths to .env files, ordered from highest level (furthest parent) to lowest.
    """
    if start_dir is None:
        start_dir = os.getcwd()

    dotenv_files = []
    current_dir = Path(start_dir).absolute()

    # Search in current directory and up to max_levels_up parent directories
    for _ in range(max_levels_up + 1):
        env_file = current_dir / '.env'
        if env_file.exists():
            dotenv_files.append(str(env_file))

        # Move up one directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # We've reached the root directory
            break
        current_dir = parent_dir

    # Return files in order from highest directory to lowest
    return list(reversed(dotenv_files))

def load_dotenv_files(start_dir: Optional[str] = None, max_levels_up: int = 3) -> List[str]:
    """
    Load environment variables from .env files in current and parent directories.

    Variables in lower directories (closer to start_dir) take precedence over
    those in higher directories.

    Args:
        start_dir: Directory to start searching from. Defaults to current directory.
        max_levels_up: Maximum number of parent directories to search.

    Returns:
        List of paths to .env files that were successfully loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv package is not installed. Cannot load .env files.")
        print("Install it with: pip install python-dotenv")
        return []

    loaded_files = []
    for dotenv_path in find_dotenv_files(start_dir, max_levels_up):
        # Load the .env file
        if load_dotenv(dotenv_path, override=True):  # override=True allows lower dirs to take precedence
            loaded_files.append(dotenv_path)
            print(f"Loaded environment variables from: {dotenv_path}")

    return loaded_files

def get_model_settings(model_name, temperature=0.7, max_tokens=None, parallel_tool_calls=False):
    """
    Returns model settings with appropriate reasoning effort based on model type.
    For o4-mini, applies high reasoning effort.
    For other models like gpt-4o, uses default settings.

    Args:
        model_name: Name of the model to use
        temperature: Temperature for model generation (ignored for o4-mini)
        max_tokens: Maximum tokens to generate
        parallel_tool_calls: Whether to enable parallel tool calls

    Returns:
        ModelSettings object configured appropriately for the model
    """
    # Create different settings objects based on model type
    if model_name == "o4-mini" or model_name == "o3":
        # o4-mini doesn't support temperature parameter
        settings = ModelSettings(
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls
        )
        # Apply high reasoning effort
        settings.reasoning_effort = "high"
    else:
        # For other models like gpt-4o, use all parameters
        settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls
        )

    return settings

def load_web_search_agent(enable_langfuse: bool = False, service_name: str = "web_search_agent") -> Tuple[Any, Callable]:
    """
    Factory function to create and return a web search agent.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name to use for Langfuse tracing
        
    Returns:
        tuple: (agent, run_agent_function) - The configured agent instance and a function to run queries.
    """
    import asyncio
    import openai
    from agents import Agent, Runner, WebSearchTool, ModelSettings

    logger.info("Initializing web search agent")

    # Set up Langfuse tracing if enabled
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse
            
            if setup_langfuse():
                logger.info("Langfuse tracing enabled for web search agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning("Could not import langfuse_integration module. Langfuse tracing will not be enabled.")
            logger.warning("Make sure you have installed 'pydantic-ai[logfire]' package.")

    # Get API client
    async_client = openai.AsyncOpenAI()

    # Create an agent with web search capabilities
    agent = Agent(
        name="Web Search Assistant",
        instructions="""
        You are a research assistant that provides accurate, up-to-date information by searching the web.
        Follow these guidelines:
        1. When asked about current events or recent information, always use the web search tool.
        2. Cite your sources by mentioning the websites you found information from.
        3. If you don't know the answer or can't find reliable information, be honest about it.
        4. Provide balanced perspectives on controversial topics.
        5. When appropriate, include relevant statistics or data to support your answers.
        """,
        tools=[
            # Updated parameters based on the WebSearchTool API
            WebSearchTool(
                search_context_size='high'  # Use high context size to get more detailed results
            )
        ],
        # Set the model directly
        model="gpt-4.1",
        # Use ModelSettings for temperature and other parameters
        model_settings=ModelSettings(
            temperature=0.2,  # Lower temperature for more focused responses
        ),
    )

    logger.debug("Web search agent created with WebSearchTool and GPT-4.1 model")

    async def run_agent(query: str):
        """Run the agent with the given query."""
        logger.info(f"Running web search query: {query}")
        try:
            # If Langfuse is enabled, wrap the agent run with a trace context
            if enable_langfuse:
                try:
                    from deep_research.langfuse_integration import create_trace
                    
                    # Generate a session ID based on the first few chars of the query
                    session_id = f"web_search_{hash(query) % 10000}"
                    
                    with create_trace(
                        name="Web-Search-Query",
                        session_id=session_id,
                        tags=["web_search"],
                        environment=os.environ.get("ENVIRONMENT", "development")
                    ) as span:
                        # Set input for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("input.value", query)
                            except Exception as e:
                                logger.warning(f"Could not set input attribute on span: {e}")
                        
                        result = await Runner.run(
                            agent,
                            query,
                            max_turns=5,  # Limit the maximum number of turns
                        )
                        
                        # Set output for the trace if span is not None
                        if span is not None:
                            try:
                                span.set_attribute("output.value", result.final_output)
                            except Exception as e:
                                logger.warning(f"Could not set output attribute on span: {e}")
                        
                        logger.info("Web search completed successfully")
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
                max_turns=5,  # Limit the maximum number of turns
            )
            logger.info("Web search completed successfully")
            logger.debug(f"Response length: {len(result.final_output)}")
            return result.final_output
                
        except Exception as e:
            logger.error(f"Error running web search agent: {str(e)}")
            raise

    logger.info("Web search agent initialized and ready for queries")
    return agent, run_agent
