"""
Test script to examine RunResult structure.
"""
import asyncio
import os
from agents import Agent, Runner

# Load environment variables from .env files
try:
    from dotenv import load_dotenv
    from deep_research.utils import find_dotenv_files
    
    # Load all .env files found
    for env_file in find_dotenv_files():
        load_dotenv(env_file)
        print(f"Loaded environment variables from: {env_file}")
except ImportError:
    print("Could not import dotenv or utils module")

# Let's also directly examine the RunResult class
def print_class_info():
    """Print info about the RunResult class."""
    try:
        from agents.run import RunResult
        print("\nRunResult class info:")
        print(f"Module: {RunResult.__module__}")
        print(f"Class: {RunResult.__name__}")
        print(f"Bases: {RunResult.__bases__}")
        
        # Get annotations
        try:
            import inspect
            print(f"\nAnnotations: {inspect.get_annotations(RunResult)}")
        except Exception as e:
            print(f"Could not get annotations: {e}")
            
        # Get __init__ parameters
        try:
            sig = inspect.signature(RunResult.__init__)
            print(f"\nInit signature: {sig}")
        except Exception as e:
            print(f"Could not get signature: {e}")
    except ImportError:
        print("Could not import RunResult class")

print_class_info()

async def test():
    """Test runner and print RunResult structure."""
    # Check if API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set")
        return None
        
    agent = Agent(name="Test", instructions="Hello, respond with 'Hello World'")
    result = await Runner.run(agent, "test")
    
    print("\nAttributes of RunResult:")
    print(dir(result))
    
    print("\nType of RunResult:")
    print(type(result))
    
    print("\nString representation:")
    print(str(result))
    
    print("\nRepr representation:")
    print(repr(result))
    
    # Try to access potential attributes
    try:
        print("\nResult.final_output:")
        print(result.final_output)
    except AttributeError:
        print("No attribute 'final_output'")

    try:
        print("\nResult.output:")
        print(result.output)  
    except AttributeError:
        print("No attribute 'output'")
        
    try:
        print("\nResult.response:")
        print(result.response)
    except AttributeError:
        print("No attribute 'response'")
    
    try:
        print("\nResult.outputs:")
        print(result.outputs)
    except AttributeError:
        print("No attribute 'outputs'")
        
    try:
        print("\nResult.completion:")
        print(result.completion)
    except AttributeError:
        print("No attribute 'completion'")

    # Check if result has a __dict__ and print its contents
    print("\nContents of __dict__:")
    print(result.__dict__)
        
    return result

if __name__ == "__main__":
    asyncio.run(test()) 