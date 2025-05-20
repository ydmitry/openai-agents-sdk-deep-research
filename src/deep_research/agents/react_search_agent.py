#!/usr/bin/env python3
"""
Implementation of the ReAct (Reasoning + Action) search agent.

This module provides the ReAct search agent with chain-of-thought reasoning
capabilities for comprehensive research queries.
"""

import logging
import os
from typing import Any, Callable, List, Optional, Tuple
from pydantic import BaseModel, Field

# Module logger
logger = logging.getLogger(__name__)

# Define reasoning step model to track chain of thought
class ReasoningStep(BaseModel):
    thought: str = Field(description="The agent's thought process")
    action: Optional[str] = Field(None, description="The action taken based on the thought")
    observation: Optional[str] = Field(None, description="The observation from the action")

def load_react_search_agent(
    enable_langfuse: bool = False,
    service_name: str = "react_search_agent",
) -> Tuple[Any, Callable[[str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that follows the ReAct pattern
        2. An async `run_agent(query)` coroutine to execute queries with reasoning chain

    The implementation uses ONLY the `agents` package (no direct
    `openai` import). All model access is delegated to the Agents SDK.
    """
    try:
        # Import the agents package (not OpenAI directly)
        from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool
    except ImportError:
        logger.error("Agents SDK not installed. Please ensure it's available in your environment.")
        raise ImportError("Agents SDK not installed")

    # ──────────────────────────────────────────────────────────────────────────
    # Optional Langfuse tracing
    # ──────────────────────────────────────────────────────────────────────────
    if enable_langfuse:
        try:
            from deep_research.langfuse_integration import setup_langfuse

            if setup_langfuse():
                logger.info("Langfuse tracing enabled for react search agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning(
                "Langfuse integration not found; install `pydantic-ai[logfire]` "
                "if you need tracing."
            )

    # List of reasoning steps for tracking chain of thought
    reasoning_steps = []
    
    # ──────────────────────────────────────────────────────────────────────────
    # Custom function tools to implement ReAct pattern
    # ──────────────────────────────────────────────────────────────────────────
    @function_tool
    async def add_reasoning_step(thought: str, action: Optional[str] = None, observation: Optional[str] = None) -> str:
        """
        Add a step to the reasoning chain.
        
        Args:
            thought: The agent's thought process
            action: The action taken based on the thought
            observation: The observation from the action
            
        Returns:
            Confirmation message
        """
        reasoning_steps.append(ReasoningStep(
            thought=thought,
            action=action,
            observation=observation
        ))
        return f"Added reasoning step #{len(reasoning_steps)}"
    
    @function_tool
    async def get_reasoning_chain() -> List[ReasoningStep]:
        """
        Get the current reasoning chain.
        
        Returns:
            The list of reasoning steps
        """
        return reasoning_steps

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    agent = Agent(
        name="ReAct Search Assistant",
        instructions="""
        You are a ReAct (Reasoning + Action) search agent that uses a chain-of-thought process to find information.
        
        For each query, you must follow these steps:
        
        1. THINK: First, reason about what you need to know and how to approach the search. Use the add_reasoning_step tool to record your thinking.
        
        2. ACT: Based on your reasoning, use the web_search tool to look for specific information. Always be precise with your search queries.
        
        3. OBSERVE: Analyze the search results carefully. Use the add_reasoning_step tool to record your observations.
        
        4. THINK AGAIN: Based on what you've learned, reason about what additional information you need or what conclusions you can draw. Use the add_reasoning_step tool.
        
        5. Repeat steps 2-4 until you have gathered enough information to provide a comprehensive answer.
        
        At each step, explicitly record your reasoning using the add_reasoning_step tool before taking any action.
        
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
            WebSearchTool(search_context_size="high"),
            add_reasoning_step,
            get_reasoning_chain
        ],
        model="gpt-4.1",  # the Agents SDK handles the provider under the hood
        model_settings=ModelSettings(
            temperature=0.2,  # lower temperature → focussed answers
            max_tokens=4000,  # allow for comprehensive responses
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper coroutine to execute a single query
    # ──────────────────────────────────────────────────────────────────────────
    async def run_agent(query: str):
        """
        Execute the agent for a single query with ReAct reasoning chain.
        Handles optional Langfuse tracing if enabled.
        """
        # Clear the reasoning steps from previous runs
        reasoning_steps.clear()
        
        logger.info(f"Running ReAct search query: {query}")

        async def _invoke() -> str:
            # Run the agent with higher max_turns to allow for multiple search-reasoning cycles
            result = await Runner.run(agent, query, max_turns=10)
            
            # Format the full response with the reasoning chain
            full_response = f"# Research Query: {query}\n\n"
            full_response += "## Chain of Reasoning:\n\n"
            
            for i, step in enumerate(reasoning_steps, 1):
                full_response += f"### Step {i}:\n"
                full_response += f"**Thought**: {step.thought}\n\n"
                
                if step.action:
                    full_response += f"**Action**: {step.action}\n\n"
                    
                if step.observation:
                    full_response += f"**Observation**: {step.observation}\n\n"
                    
            full_response += f"## Final Answer:\n\n{result.final_output}\n"
            
            logger.debug(f"Response with reasoning chain length: {len(full_response)}")
            return full_response

        # With tracing
        if enable_langfuse:
            try:
                from deep_research.langfuse_integration import create_trace

                session_id = f"react_search_{hash(query) % 10_000}"
                with create_trace(
                    name="ReAct-Search-Query",
                    session_id=session_id,
                    tags=["react_search", "chain_of_thought"],
                    environment=os.getenv("ENVIRONMENT", "development"),
                ) as span:
                    if span is not None:
                        span.set_attribute("input.value", query)
                        span.set_attribute("reasoning_steps.count", 0)  # will update later

                    output = await _invoke()

                    if span is not None:
                        span.set_attribute("output.value", output)
                        span.set_attribute("reasoning_steps.count", len(reasoning_steps))

                    return output
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse tracing failure: %s – proceeding normally", exc)

        # Without tracing
        return await _invoke()

    logger.info("ReAct search agent initialized")
    return agent, run_agent 