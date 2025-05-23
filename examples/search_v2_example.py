#!/usr/bin/env python3
"""
Example script demonstrating sequential use of tools with Agents SDK.

This script shows how to:
1. Create an agent that uses tools in a defined sequence
2. Define function tools with simple implementations
3. Process inputs through a chain of tools
4. Implement both single-query and interactive modes
5. Utilize WebSearchTool for real web searches
6. Provide a chat interface for asking questions

Usage:
    python examples/search_v2_example.py "Your input here"
    python examples/search_v2_example.py --interactive
    python examples/search_v2_example.py --chat
    python examples/search_v2_example.py "Process this text" --model "gpt-4o"
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

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

def load_sequential_agent(
    model: str = "gpt-4.1",
    temperature: float = 0.2,
) -> Tuple[Any, Callable[[str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that uses tools sequentially
        2. An async `run_agent(input)` coroutine to execute the agent with input
    """
    try:
        # Import the agents package
        from agents import Agent, Runner, ModelSettings, function_tool, WebSearchTool
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # Store process results for tracking
    process_results = {
        "web_search_input": "",
        "web_search_output": ""
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Define the sequential tools
    # ──────────────────────────────────────────────────────────────────────────
    @function_tool
    async def web_search_tool(query: str) -> str:
        """
        Web search tool that creates an agent with WebSearchTool to search the actual web.

        Args:
            query: The search query to process

        Returns:
            Web search results
        """
        process_results["web_search_input"] = query
        logger.info(f"Web search tool received query: '{query}'")

        try:
            # Create a search agent that uses WebSearchTool
            search_agent = Agent(
                name="Web Search Agent",
                instructions=f"""
                You are a web search agent. Your task is to search the internet for relevant information
                about the following query and provide comprehensive results:

                "{query}"

                Use the web_search tool to find the most relevant and up-to-date information.
                Summarize your findings in a clear, concise manner with the most important facts and details.
                Include citations or sources when possible.
                """,
                tools=[WebSearchTool(search_context_size="high")],
                model=model,
                model_settings=ModelSettings(
                    temperature=temperature,
                    max_tokens=1000,
                ),
            )

            # Run the search agent
            search_result = await Runner.run(search_agent, "", max_turns=3)
            output = f"WEB SEARCH RESULTS: {search_result.final_output}"

        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}")
            output = f"WEB SEARCH ERROR: Failed to get results for query '{query}'. Error: {str(e)}"

        process_results["web_search_output"] = output
        logger.info(f"Web search tool produced output of length: {len(output)}")
        return output

    @function_tool
    async def get_process_results() -> dict:
        """
        Get the results from all processing steps.

        Returns:
            A dictionary containing inputs and outputs from all tools
        """
        return process_results

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    agent = Agent(
        name="Sequential Search Agent",
        instructions="""
        You are an agent that performs tasks using a sequence of tools.

        For each input, follow these steps:

        1. FIRST STEP: If this is the first message use web_search_tool with the original input to search the internet for real-time information.

        2. SECOND STEP: Brainstorm the next unique search query based on the output from previous steps. This search query should be different from the previous search queries. Result should augment the previous results.

        3. THIRD STEP: Use web_search_tool with the search query from previous step to search the internet for real-time information.

        4. LOOP: Continue 1-3 steps two more times in same sequence.

        5. FINAL STEP: Provide a comprehensive summary of all the search results to the user.

        You can use get_process_results to review all inputs and outputs if needed.

        Always follow this exact sequence for processing any input.

        Important guidelines:
        - Be thorough in your reasoning
        - Break down complex queries into specific searchable components
        - Critically evaluate the information you find
        - Consider multiple perspectives and sources
        - Always cite sources in your final answer

        Your final response should include:
        1. A comprehensive answer to the original query
        2. Citations for all sources used
        """,
        tools=[
            web_search_tool,
            get_process_results
        ],
        model=model,
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=1000,
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper coroutine to execute a single query
    # ──────────────────────────────────────────────────────────────────────────
    async def run_agent(input_text: str):
        """
        Execute the agent for processing a single input.
        """
        # Reset the process results from previous runs
        process_results.update({
            "web_search_input": "",
            "web_search_output": ""
        })

        logger.info(f"Running sequential agent with input: {input_text}")

        # Run the agent
        result = await Runner.run(agent, input_text, max_turns=20)

        # Format response to show processing steps
        full_response = f"# Sequential Processing Result\n\n"
        full_response += f"## Initial Input\n{input_text}\n\n"

        # full_response += f"## Processing Steps\n"
        # full_response += f"1. **Web Search Tool**\n"
        # full_response += f"   - Query: {process_results['web_search_input']}\n"
        # full_response += f"   - Results: {process_results['web_search_output']}\n\n"

        full_response += f"## Agent Response\n{result.final_output}\n"

        logger.debug(f"Full response with processing details: {len(full_response)} chars")
        return full_response

    logger.info(f"Sequential tools agent initialized with model: {model}")
    return agent, run_agent

async def run_query(input_text: str, model: str = "gpt-4.1", temperature: float = 0.2):
    """Run a single input through the sequential tools agent."""
    logger.info(f"Processing single input: {input_text}")

    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the sequential tools agent
    logger.debug(f"Loading sequential tools agent with model: {model}")
    _, run_agent = load_sequential_agent(model=model, temperature=temperature)

    # Run the query and return the result
    logger.info("Sending input to agent")
    result = await run_agent(input_text)
    logger.info("Processing completed successfully")
    return result

async def interactive_mode(model: str = "gpt-4.1", temperature: float = 0.2):
    """Run the sequential tools agent in interactive mode."""
    logger.info(f"Starting interactive mode with model: {model}")

    # Load environment variables (including API keys)
    log_env_files = load_dotenv_files()
    if log_env_files:
        logger.info(f"Loaded environment from: {', '.join(log_env_files)}")

    # Load the sequential tools agent (only once for the session)
    logger.debug("Loading sequential tools agent for interactive session")
    _, run_agent = load_sequential_agent(model=model, temperature=temperature)

    print("Sequential Tools Agent (type 'exit' to quit)")
    print("This agent processes inputs through sequential web search operations.")

    query_count = 0
    while True:
        input_text = input("\nEnter your input: ")
        if input_text.lower() in ['exit', 'quit']:
            logger.info("User requested exit from interactive mode")
            break

        query_count += 1
        logger.info(f"Processing interactive input #{query_count}: {input_text}")
        print("Running sequential processing...")
        try:
            response = await run_agent(input_text)
            logger.info(f"Successfully completed interactive query #{query_count}")
            print("\nAgent response:")
            print(response)
        except Exception as e:
            logger.error(f"Error processing input #{query_count}: {str(e)}")
            print(f"Error occurred: {e}")

    logger.info(f"Interactive session ended after {query_count} queries")

async def chat_mode(model: str = "gpt-4.1", temperature: float = 0.2):
    """
    Run the search agent in chat mode, providing a conversational interface.

    This creates a simple chat-like experience where the user can ask questions
    and receive direct responses from the agent without seeing the processing details.
    The conversation history is maintained to provide context between messages.
    """
    logger.info(f"Starting chat mode with model: {model}")

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

    # Load the agent directly (not the run_agent function)
    logger.debug("Initializing search agent for chat session")
    agent, _ = load_sequential_agent(model=model, temperature=temperature)

    print("\n================================")
    print("💬 Search Agent Chat")
    print("================================")
    print("Ask any question and get answers with real-time web search.")
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
            print("Please enter a question or type 'exit' to quit.")
            continue

        message_count += 1
        logger.info(f"Processing chat message #{message_count}: {user_input}")

        print("\nSearching for information...")
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

            logger.info(f"Successfully generated response for message #{message_count}")

            # Display only the final output
            print("\nAssistant:")
            print(result.final_output)
            print("\n--------------------------------")

        except Exception as e:
            logger.error(f"Error processing message #{message_count}: {str(e)}")
            print(f"Sorry, I encountered an error: {e}")

    logger.info(f"Chat session ended after {message_count} messages")

async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run an agent with sequential tools")
    parser.add_argument("input", nargs="?", help="The input to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chat", "-c", action="store_true", help="Run in chat mode with a conversational interface")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use (default: gpt-4.1)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature setting (default: 0.2)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive or args.chat):
        # Auto-create log file for debug mode or interactive/chat sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/sequential_tools_{timestamp}.log"

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting sequential tools example with model: {args.model}, temperature: {args.temperature}")

    if args.chat:
        await chat_mode(model=args.model, temperature=args.temperature)
    elif args.interactive:
        await interactive_mode(model=args.model, temperature=args.temperature)
    elif args.input:
        logger.info(f"Running in single input mode")
        result = await run_query(args.input, model=args.model, temperature=args.temperature)
        print(result)
    else:
        logger.warning("No input provided and not in interactive mode")
        parser.print_help()

    logger.info("Sequential tools example completed")

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
