#!/usr/bin/env python3
"""
Implementation of the Research Topics Generator agent.

This module provides an agent that analyzes search results and identifies
related topics for broader research.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

# Module logger
logger = logging.getLogger(__name__)

def load_research_topics_agent(
    enable_langfuse: bool = False,
    service_name: str = "research_topics_agent",
) -> Tuple[Any, Callable[[str, str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that generates research topics
        2. An async `run_agent(query, search_results)` coroutine to execute topic generation

    The implementation uses ONLY the `agents` package (no direct
    `openai` import). All model access is delegated to the Agents SDK.
    """
    try:
        # Import the agents package (not OpenAI directly)
        from agents import Agent, Runner, ModelSettings, function_tool
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
                logger.info("Langfuse tracing enabled for research topics agent")
            else:
                logger.warning("Failed to set up Langfuse tracing")
        except ImportError:
            logger.warning(
                "Langfuse integration not found; install `pydantic-ai[logfire]` "
                "if you need tracing."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    topics_agent = Agent(
        name="Research Topics Generator",
        instructions="""
        You are a Research Topics Generator that analyzes search results and identifies related topics for broader research.
        
        Given:
        1. A original research query
        2. The search results and reasoning chain from a ReAct search agent
        
        Your task is to:
        1. Identify 3-5 related topics that would be valuable for broader research
        2. For each topic, provide a brief explanation of why it's relevant
        3. Suggest a specific search query that would be effective for researching each topic
        
        Format your response as a structured list of topics with explanations and suggested queries.
        Focus on identifying gaps in the current research or areas that would provide valuable context.
        
        For each topic, use this format:
        
        1. **Topic Name**
           - **Relevance**: Brief explanation of why this topic is important
           - **Suggested Query**: "specific search query for this topic"
        
        Make sure to:
        - Choose diverse topics that cover different aspects of the subject
        - Identify potential gaps in the original research
        - Suggest specific, well-formulated search queries
        - Explain clearly why each topic adds value to the research
        """,
        model="gpt-4.1",  # the Agents SDK handles the provider under the hood
        model_settings=ModelSettings(
            temperature=0.4,  # slightly higher temperature for creative topic generation
            max_tokens=2000,  # allow for comprehensive response with multiple topics
        ),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper coroutine to execute topic generation
    # ──────────────────────────────────────────────────────────────────────────
    async def run_agent(query: str, search_results: str) -> str:
        """
        Execute the agent to generate research topics based on search results.
        Handles optional Langfuse tracing if enabled.
        
        Args:
            query: The original research query
            search_results: The results from the ReAct search agent
            
        Returns:
            A formatted string with 3-5 research topics
        """
        logger.info(f"Generating research topics for query: {query}")
        
        async def _invoke() -> str:
            # Prepare the input to the topics agent
            input_message = f"""
            Original Research Query: {query}
            
            Search Results and Reasoning:
            {search_results}
            
            Based on the above information, please generate 3-5 related research topics that would provide 
            valuable additional context or address gaps in the current research.
            """
            
            # Run the agent
            try:
                result = await Runner.run(topics_agent, input_message, max_turns=3)
                topics = result.final_output
                logger.info(f"Successfully generated research topics")
                return topics
            except Exception as e:
                logger.error(f"Error generating research topics: {str(e)}")
                return "Error generating research topics. Please try again."

        # With tracing
        if enable_langfuse:
            try:
                from deep_research.langfuse_integration import create_trace

                session_id = f"topics_generator_{hash(query) % 10_000}"
                with create_trace(
                    name="Research-Topics-Generation",
                    session_id=session_id,
                    tags=["research_topics", "gap_analysis"],
                    environment=os.getenv("ENVIRONMENT", "development"),
                ) as span:
                    if span is not None:
                        span.set_attribute("input.query", query)
                        span.set_attribute("input.search_results_length", len(search_results))

                    output = await _invoke()

                    if span is not None:
                        span.set_attribute("output.value", output)
                        span.set_attribute("output.topics_count", output.count("\n**") if "**" in output else 0)

                    return output
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse tracing failure: %s – proceeding normally", exc)

        # Without tracing
        return await _invoke()

    logger.info("Research topics agent initialized")
    return topics_agent, run_agent 