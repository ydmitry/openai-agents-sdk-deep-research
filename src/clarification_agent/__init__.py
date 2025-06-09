"""
Clarification Agent Package

A standalone clarification agent that analyzes user requests and generates
clarification questions to identify gaps and ambiguities.

Example usage:
    from clarification_agent import make_clarification_agent
    
    def handle_questions(questions: str):
        print(f"Questions: {questions}")
    
    agent = make_clarification_agent(handle_questions)
"""

from .agent import make_clarification_agent

__version__ = "1.0.0"
__all__ = [
    "make_clarification_agent",
] 