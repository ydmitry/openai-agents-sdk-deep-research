"""
Sequential Search Agent Package

A sequential search agent that uses websearch agents as tools for complex, multi-step research.

This package provides:
- make_sequential_search_agent: Factory function for creating sequential search agents
- Collection helper functions: Available in the runner script
- Standalone runner script for command-line usage

Example usage:
    from sequential_search_agent.agent import make_sequential_search_agent
    
    def my_collector(answer: str):
        print(f"Collected: {answer}")
    
    agent = make_sequential_search_agent(my_collector)
"""

from .agent import make_sequential_search_agent

__version__ = "1.0.0"
__all__ = [
    "make_sequential_search_agent",
] 