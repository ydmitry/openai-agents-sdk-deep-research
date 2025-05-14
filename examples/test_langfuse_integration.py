#!/usr/bin/env python3
"""
Test script to verify Langfuse integration with OpenAI Agents SDK.

This script:
1. Sets up Langfuse tracing
2. Creates a simple agent
3. Runs a test query with tracing enabled
4. Outputs the trace URL if successful

Requirements:
- OPENAI_API_KEY environment variable must be set
- LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables must be set

Usage:
    python test_langfuse_integration.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research.langfuse_integration import setup_langfuse, create_trace
from deep_research.utils import load_dotenv_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_langfuse_integration():
    """Test the Langfuse integration with a simple agent."""
    # Load environment variables (including API keys)
    loaded_files = load_dotenv_files()
    if loaded_files:
        logger.info(f"Loaded environment from: {', '.join(loaded_files)}")
    
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment")
        return False
    
    # Set up Langfuse
    if not setup_langfuse():
        logger.error("Failed to set up Langfuse integration")
        return False
    
    logger.info("Langfuse integration set up successfully")
    
    # Create a simple agent
    try:
        from agents import Agent, Runner
        
        agent = Agent(
            name="Test Agent",
            instructions="You are a test agent. Just respond with 'Hello from the test agent!'",
            model="gpt-4.1",
        )
        
        # Run a test query with tracing
        logger.info("Running test query with tracing...")
        
        try:
            with create_trace(
                name="Test-Agent-Trace",
                user_id="test-user",
                session_id="test-session",
                tags=["test", "integration"],
                environment="test"
            ) as span:
                # Set input if span is not None
                if span is not None:
                    try:
                        span.set_attribute("input.value", "Hello, agent!")
                    except Exception as e:
                        logger.warning(f"Could not set input attribute: {e}")
                
                # Run the agent
                result = await Runner.run(agent, "Hello, agent!")
                
                # Set output if span is not None
                if span is not None:
                    try:
                        span.set_attribute("output.value", result.final_output)
                    except Exception as e:
                        logger.warning(f"Could not set output attribute: {e}")
                
                logger.info(f"Agent response: {result.final_output}")
                
                logger.info("Test completed successfully!")
                logger.info("Check your Langfuse dashboard to see the trace")
                
                # If we got here, tracing was successful
                return True
        except Exception as e:
            logger.error(f"Error during tracing: {e}")
            logger.warning("Test will continue without tracing...")
            
            # Try running without tracing
            result = await Runner.run(agent, "Hello, agent!")
            logger.info(f"Agent response (without tracing): {result.final_output}")
            logger.warning("Test completed, but tracing failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
        return False

def main():
    """Main entry point for the script."""
    try:
        success = asyncio.run(test_langfuse_integration())
        if success:
            print("\n✅ Langfuse integration test passed!")
            print("Check your Langfuse dashboard to see the trace.")
        else:
            print("\n❌ Langfuse integration test failed.")
            print("See the logs above for details.")
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 