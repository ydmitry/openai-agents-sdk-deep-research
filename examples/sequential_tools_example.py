#!/usr/bin/env python3
"""
Example script demonstrating sequential use of tools with Agents SDK.

This script shows how to:
1. Create an agent that uses tools in a defined sequence
2. Define function tools with simple implementations
3. Process inputs through a chain of tools
4. Implement both single-query and interactive modes

Usage:
    python sequential_tools_example.py "Your input here"
    python sequential_tools_example.py --interactive
    python sequential_tools_example.py "Process this text" --model "gpt-4o"
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
        from agents import Agent, Runner, ModelSettings, function_tool
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        sys.exit(1)

    # Store process results for tracking
    process_results = {
        "tool_one_input": "",
        "tool_one_output": "",
        "tool_two_input": "",
        "tool_two_output": ""
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Define the sequential tools
    # ──────────────────────────────────────────────────────────────────────────
    @function_tool
    async def tool_one(input: str) -> str:
        """
        First tool in the sequence. Processes the input text.

        Args:
            input: The text to be processed by the first tool

        Returns:
            Processed output to be passed to the second tool
        """
        # Simple example logic - convert to uppercase and add a prefix
        process_results["tool_one_input"] = input
        output = f"PROCESSED: {input.upper()}"
        process_results["tool_one_output"] = output

        logger.info(f"Tool One processed input: '{input}' → '{output}'")
        return output

    @function_tool
    async def tool_two(input: str) -> str:
        """
        Second tool in the sequence. Further processes the output from tool_one.

        Args:
            input: The text from tool_one to be further processed

        Returns:
            Final processed output
        """
        # Simple example logic - extract words and add a suffix
        process_results["tool_two_input"] = input
        words = input.split()
        output = f"FINALIZED: {' '.join(words)} [Completed at step 2]"
        process_results["tool_two_output"] = output

        logger.info(f"Tool Two processed input: '{input}' → '{output}'")
        return output

    @function_tool
    async def get_process_results() -> dict:
        """
        Get the results from both processing steps.

        Returns:
            A dictionary containing inputs and outputs from both tools
        """
        return process_results

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    agent = Agent(
        name="Sequential Tools Agent",
        instructions="""
        You are an agent that performs tasks using a sequence of tools.

        For each input, follow these steps:

        1. FIRST STEP: Use tool_one to process the initial input.

        2. SECOND STEP: Take the output from tool_one and use it as input for tool_two.

        3. THIRD STEP: Use tool_one to process the output from previous step.

        4. LOOP: Continue 1-3 steps five more times in same sequence.

        5. FINAL STEP: Provide the result from tool_two to the user, along with a brief explanation
           of how the input was processed through both tools.

        You can use get_process_results to review all inputs and outputs if needed.

        Always follow this exact sequence for processing any input.
        """,
        tools=[
            tool_one,
            tool_two,
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
            "tool_one_input": "",
            "tool_one_output": "",
            "tool_two_input": "",
            "tool_two_output": ""
        })

        logger.info(f"Running sequential agent with input: {input_text}")

        # Run the agent
        result = await Runner.run(agent, input_text, max_turns=20)

        # Format response to show processing steps
        full_response = f"# Sequential Processing Result\n\n"
        full_response += f"## Initial Input\n{input_text}\n\n"

        full_response += f"## Processing Steps\n"
        full_response += f"1. **Tool One**\n"
        full_response += f"   - Input: {process_results['tool_one_input']}\n"
        full_response += f"   - Output: {process_results['tool_one_output']}\n\n"

        full_response += f"2. **Tool Two**\n"
        full_response += f"   - Input: {process_results['tool_two_input']}\n"
        full_response += f"   - Output: {process_results['tool_two_output']}\n\n"

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
    print("This agent processes inputs through two sequential tools.")

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

    logger.info(f"Interactive session ended after {query_count} inputs")

async def main_async():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run an agent with sequential tools")
    parser.add_argument("input", nargs="?", help="The input to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use (default: gpt-4.1)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature setting (default: 0.2)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file
    if not log_file and (args.log_level == "DEBUG" or args.interactive):
        # Auto-create log file for debug mode or interactive sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/sequential_tools_{timestamp}.log"

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.info(f"Starting sequential tools example with model: {args.model}, temperature: {args.temperature}")

    if args.interactive:
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
