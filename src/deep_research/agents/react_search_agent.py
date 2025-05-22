#!/usr/bin/env python3
"""
Implementation of the ReAct (Reasoning + Action) search agent.

This module provides the ReAct search agent with chain-of-thought reasoning
capabilities for comprehensive research queries, and includes handoff to
the research topics agent.
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
    with_research_topics_handoff: bool = False
) -> Tuple[Any, Callable[[str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that follows the ReAct pattern
        2. An async `run_agent(query)` coroutine to execute queries with reasoning chain

    The implementation uses ONLY the `agents` package (no direct
    `openai` import). All model access is delegated to the Agents SDK.

    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name for tracing
        with_research_topics_handoff: Whether to enable handoff to research topics agent
    """
    try:
        # Import the agents package (not OpenAI directly)
        from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool, handoff, RunContextWrapper
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
    # Get research topics agent for handoff if enabled
    # ──────────────────────────────────────────────────────────────────────────
    handoffs = []
    if with_research_topics_handoff:
        try:
            from deep_research.agents.research_topics_agent import load_research_topics_agent

            # Load the research topics agent
            research_topics_agent, run_topics_agent = load_research_topics_agent(enable_langfuse=enable_langfuse)

            # Define the on_handoff callback function
            async def on_handoff(ctx: RunContextWrapper[None]):
                """
                Handle the handoff from the search agent to the research topics agent.

                Args:
                    ctx: The run context wrapper with the full conversation history
                """
                logger.info("Handling research topics handoff")

                # Extract the original query from the conversation history
                query = ""
                # Extract query from conversation history
                history = ctx.input_list
                for message in reversed(history):
                    if message.get("role") == "user":
                        query = message.get("content", "")
                        logger.info(f"Extracted query from conversation history: {query}")
                        break

                if not query:
                    logger.warning("No query found in conversation history")
                    return "Unable to generate research topics without a valid query."

                # Run the topics agent with the query
                return await run_topics_agent(query)

            # Create a handoff object with a custom name, input type, and on_handoff function
            research_topics_handoff = handoff(
                agent=research_topics_agent,
                tool_name_override="generate_research_topics",
                tool_description_override="Generate related research topics that would provide additional context or address gaps in the current research",
            )

            handoffs.append(research_topics_handoff)
            logger.info("Research topics agent loaded for handoff with custom configuration")
        except ImportError:
            logger.error("Failed to import research_topics_agent. Handoff will not be available.")

    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    agent = Agent(
        name="ReAct Search Assistant",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are "ReAct Research Orchestrator", a multi-stage research assistant that runs an **iterative loop**:

─────────────────────────────────
HIGH-LEVEL WORKFLOW
─────────────────────────────────
1.  **SELECT TOPIC**
    • If no topic has been researched yet, the topic = the user's original query.
    • Otherwise, pick the next topic returned by the *generate_research_topics* tool that has **NOT** been fully researched yet.
    • Keep a running list called `researched_topics` to avoid repeats.
    • Record a THINK step explaining *why* you chose this topic (`add_reasoning_step`).

2.  **REACT SEARCH CYCLE FOR TOPIC**
    Repeat until confident you can summarise the topic:
    a. THINK → record via `add_reasoning_step`.
    b. ACT  → run **web_search** with a precise query.
    c. OBSERVE → scan results, record key findings (`add_reasoning_step`).
    d. THINK-AGAIN → decide next action (`add_reasoning_step`).

3.  **SUMMARISE TOPIC**
    • Provide a concise but thorough summary **with inline citations**.
    • Append this summary to the growing report.
    • Mark the topic as researched (`researched_topics.append(topic)`).

4.  **DISCOVER NEXT TOPICS**
    • Immediately call the *generate_research_topics* tool (research-topics handoff).
      Parameters:
      ```json
      {{
        "original_query": "<the user's original question>",
        "reason_for_handoff": "discover follow-up topics or knowledge gaps after finishing <topic>"
      }}
      ```
    • The tool returns a JSON list of potential topics (may be empty).
    • Record an OBSERVE step detailing the returned list.

5.  **DECISION**
    • If the list contains an unresearched topic, go to step 1 with that topic.
    • If the list is empty **or** all topics were already covered, **exit the loop**.

6.  **FINAL ANSWER**
    Return:
      1. A structured, sectioned report – one section per researched topic, each with citations.
      2. A bibliography of all sources.
      3. The complete chain-of-thought (optional, if user requests).

─────────────────────────────────
RULES
─────────────────────────────────
• ALWAYS use `add_reasoning_step` **before every tool call**.
• NEVER invent sources; cite only URLs actually found.
• Keep `temperature` low – be factual and concise.
• After every topic summary you **must** call *generate_research_topics* exactly once.
• Stop only when no new topics remain.

Begin when ready.""",
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
        handoffs=handoffs,
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
