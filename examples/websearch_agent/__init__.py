"""
WebSearch Agent Package

A standalone web search agent with collection capabilities.

This package provides:
- make_search_agent: Factory function for creating search agents
- Collection methods: JSON file, text file, and memory collectors
- Standalone runner script for command-line usage

Example usage:
    from websearch_agent.agent import make_search_agent, create_json_file_collector
    
    collector = create_json_file_collector("results.json")
    agent = make_search_agent(collector)
"""

from .agent import (
    make_search_agent,
    create_json_file_collector,
    create_text_file_collector,
    create_memory_collector,
)

__version__ = "1.0.0"
__all__ = [
    "make_search_agent",
    "create_json_file_collector", 
    "create_text_file_collector",
    "create_memory_collector",
] 