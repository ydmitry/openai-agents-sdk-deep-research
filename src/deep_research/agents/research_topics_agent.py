#!/usr/bin/env python3
"""
Implementation of the Research Topics Generator agent.

This module provides an agent that analyzes search results and identifies
related topics for broader research.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
from pydantic import BaseModel

# Module logger
logger = logging.getLogger(__name__)

# Input model for the handoff
class ResearchHandoffInput(BaseModel):
    original_query: str = ""
    reason_for_handoff: Optional[str] = None

def load_research_topics_agent(
    enable_langfuse: bool = False,
    service_name: str = "research_topics_agent",
    with_search_handoff: bool = False
) -> Tuple[Any, Callable[[str, str], Any]]:
    """
    Factory function that returns:
        1. The configured Agents-SDK `Agent` that generates research topics
        2. An async `run_agent(query, search_results)` coroutine to execute topic generation

    The implementation uses ONLY the `agents` package (no direct
    `openai` import). All model access is delegated to the Agents SDK.
    
    Args:
        enable_langfuse: Whether to enable Langfuse tracing
        service_name: Service name for tracing
        with_search_handoff: Whether to include handoff back to search agent
    """
    try:
        # Import the agents package (not OpenAI directly)
        from agents import Agent, Runner, ModelSettings, function_tool, RunContextWrapper, handoff
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
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
    # Get search agent for handoff if enabled
    # ──────────────────────────────────────────────────────────────────────────
    handoffs = []
    if with_search_handoff:
        try:
            from deep_research.agents.react_search_agent import load_react_search_agent
            
            # Load the search agent (avoid circular reference by not enabling its handoff to us)
            search_agent, run_search_agent = load_react_search_agent(
                enable_langfuse=enable_langfuse,
                with_research_topics_handoff=False
            )
            
            # Define the on_handoff callback function for search
            async def on_search_handoff(ctx: RunContextWrapper[None]):
                """
                Handle the handoff from the topics agent to search agent.
                
                Args:
                    ctx: The run context wrapper with the full conversation history
                """
                logger.info("Handling handoff from topics to search")
                
                # Extract search information from conversation history
                # We'll look for JSON data in the last assistant message that contains topic information
                history = ctx.input_list
                topic_name = "related topic"
                query = ""
                
                # Look for the most recent assistant message that might contain topic information
                for message in reversed(history):
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        # Try to find JSON array in the message
                        import re
                        import json
                        
                        # Look for JSON array pattern
                        json_match = re.search(r'\[\s*\{\s*"topic"\s*:', content)
                        if json_match:
                            try:
                                # Find the opening bracket
                                start_idx = content.find('[', json_match.start())
                                # Find the matching closing bracket
                                bracket_count = 0
                                for i in range(start_idx, len(content)):
                                    if content[i] == '[':
                                        bracket_count += 1
                                    elif content[i] == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            end_idx = i + 1
                                            break
                                
                                json_str = content[start_idx:end_idx]
                                topics = json.loads(json_str)
                                if topics and isinstance(topics, list) and len(topics) > 0:
                                    first_topic = topics[0]
                                    topic_name = first_topic.get("topic", "related topic")
                                    query = first_topic.get("search_query", "")
                                    logger.info(f"Extracted topic '{topic_name}' with query: {query}")
                                    break
                            except (json.JSONDecodeError, ValueError, IndexError) as e:
                                logger.warning(f"Failed to parse JSON from message: {e}")
                        
                        # If we couldn't extract from JSON, look for a topic_query mention
                        if not query:
                            topic_query_match = re.search(r'"topic_query"\s*:\s*"([^"]+)"', content)
                            if topic_query_match:
                                query = topic_query_match.group(1)
                                # Try to extract topic name too
                                topic_name_match = re.search(r'"topic_name"\s*:\s*"([^"]+)"', content)
                                if topic_name_match:
                                    topic_name = topic_name_match.group(1)
                                logger.info(f"Extracted query from topic_query field: {query}")
                                break
                
                if not query:
                    # Fallback: try to find the most recent user query
                    for message in reversed(history):
                        if message.get("role") == "user":
                            query = message.get("content", "")
                            logger.info(f"Falling back to user query: {query}")
                            break
                
                if not query:
                    logger.warning("No topic query could be extracted from conversation history")
                    return "Unable to search without a specific query. Please provide a topic query."
                
                logger.info(f"Searching for topic: {topic_name} with query: {query}")
                
                # Run the search agent with the topic query
                search_results = await run_search_agent(query)
                return search_results
            
            # Create a handoff object for search
            search_handoff = handoff(
                agent               = search_agent,
                tool_name_override  = "search_topic",              # ← the name the LLM must invoke
                tool_description_override =
                    "Run a ReAct web-search cycle for one follow-up topic",
            )
            
            handoffs.append(search_handoff)
            logger.info("Search agent loaded for handoff from topics agent")
        except ImportError:
            logger.error("Failed to import react_search_agent. Handoff will not be available.")
        except Exception as e:
            logger.error(f"Error setting up search handoff: {str(e)}")

    # Add output guardrail for safety
    from agents import output_guardrail, GuardrailFunctionOutput
    import re

    @output_guardrail
    async def must_call_search(ctx, agent, output: str) -> GuardrailFunctionOutput:
        tool_called = re.search(r'"tool"\s*:\s*"search_topic"', output) is not None
        if not tool_called:
            logger.warning("Research Topics Generator did not call search_topic tool as expected.")
        return GuardrailFunctionOutput(
            output_info={"tool_called": tool_called},
            tripwire_triggered=False,  # Changed to False to prevent exception
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Build the agent
    # ──────────────────────────────────────────────────────────────────────────
    topics_agent = Agent(
        name="Research Topics Generator",
        handoff_description="Generates related research topics based on search results",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are "Research Topics Generator".

WORKFLOW
────────
1. Read the background information (original query and any search results).
2. Produce **exactly 3–5** follow-up research topics in the JSON ARRAY format shown below.
3. **Immediately after** the JSON, INVOKE the tool `search_topic`
   for the **first** topic you generated, passing:

   {{
     "topic_query"      : "<search_query from that topic>",
     "topic_name"       : "<topic>",
     "reason_for_search": "begin automatic deep-dive on the next topic"
   }}

4. End your turn.  
   *No confirmation question, no extra prose.*

JSON FORMAT
───────────
```json
[
  {{
    "topic"       : "Topic Name 1",
    "relevance"   : "Why this topic matters",
    "search_query": "Well-formed web search query"
  }},
  ...
]
```

RULES
─────
• The JSON **must** be the FIRST thing in your assistant message.  
• The `search_topic` tool call **must** follow immediately after the JSON.  
• Do **not** ask the user whether to continue; you always continue automatically.
""",
        model="gpt-4.1",  # the Agents SDK handles the provider under the hood
        model_settings=ModelSettings(
            temperature=0.4,  # slightly higher temperature for creative topic generation
            max_tokens=2000,  # allow for comprehensive response with multiple topics
        ),
        handoffs=handoffs,
        output_guardrails=[must_call_search],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper coroutine to execute topic generation
    # ──────────────────────────────────────────────────────────────────────────
    async def run_agent(query: str, search_results: Optional[str] = None) -> str:
        """
        Execute the agent to generate research topics based on search results.
        Handles optional Langfuse tracing if enabled.
        
        Args:
            query: The original research query
            search_results: The results from the ReAct search agent (optional when called via handoff)
            
        Returns:
            A formatted string with 3-5 research topics
        """
        logger.info(f"Generating research topics for query: {query}")
        
        async def _invoke() -> str:
            # Prepare the input to the topics agent
            if search_results:
                input_message = f"""
                Original Research Query: {query}
                
                Search Results and Reasoning:
                {search_results}
                
                Based on the above information, please generate 3-5 related research topics that would provide 
                valuable additional context or address gaps in the current research.
                """
            else:
                # When called via handoff, we might not have search results
                input_message = f"""
                Original Research Query: {query}
                
                Based on this query, please generate 3-5 related research topics that would provide 
                valuable additional context or broaden understanding of this subject.
                """
            
            # Run the agent
            try:
                result = await Runner.run(topics_agent, input_message, max_turns=5)
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
                        if search_results:
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
    
    # Return the agent and the run_agent function
    # The actual handoff will be configured in the react_search_agent.py file
    return topics_agent, run_agent 