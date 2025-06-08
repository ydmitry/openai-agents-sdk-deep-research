"""
WebSearch Agent Package

A standalone web search agent with collection capabilities.

This package provides:
- make_search_agent: Factory function for creating search agents
- Collection helper functions: Available in the runner script
- Standalone runner script for command-line usage

Example usage:
    from websearch_agent.agent import make_search_agent
    
    def my_collector(answer: str):
        print(f"Collected: {answer}")
    
    agent = make_search_agent(my_collector)
"""

from .agent import make_search_agent

__version__ = "1.0.0"
__all__ = [
    "make_search_agent",
] 