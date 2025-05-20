# ReAct Search with Research Topics Generator

This implementation extends the basic ReAct search pattern with an additional agent that generates related research topics.

## What is ReAct with Research Topics?

This implementation builds on the standard ReAct (Reasoning + Action) pattern by adding:

1. All the core ReAct search functionality:
   - Chain-of-thought reasoning (the "Re" part)
   - Action-taking abilities via web search (the "Act" part)
   - Structured reasoning process that interleaves thinking and searching

2. A dedicated Research Topics Generator that:
   - Analyzes the results of the initial search
   - Identifies gaps in the research
   - Suggests 3-5 additional topics for broader research
   - Provides specific search queries for each suggested topic

This dual-agent approach enables more comprehensive research by not only answering the original query but also suggesting related avenues for further exploration.

## Implementation Details

The implementation in `react_search_with_research_topics.py` showcases:

### 1. Agents SDK Integration

```python
from deep_research.agents.react_search_agent import load_react_search_agent
from agents import Agent, ModelSettings, function_tool
```

The example uses the core ReAct search agent and creates an additional Research Topics Generator agent.

### 2. Research Topics Generator Agent

The implementation creates a dedicated agent for topic generation:

```python
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
    """,
    model="gpt-4.1",
    model_settings=ModelSettings(
        temperature=0.4,  # Slightly higher temperature for creative topic generation
        max_tokens=2000,
    ),
)
```

### 3. Handoff Between Agents

The implementation demonstrates how to:
1. Run the ReAct search agent to get comprehensive search results
2. Pass those results to the Research Topics Generator
3. Combine both outputs into a unified response

```python
# Run the search query
search_results = await run_search_agent(query)

# Generate research topics based on search results
topics = await run_topics_agent(query, search_results)

# Combine the results
combined_results = f"{search_results}\n\n## Related Research Topics:\n\n{topics}"
```

### 4. Interactive Mode

The implementation supports an interactive mode that:
1. Keeps both agents loaded throughout the session
2. Processes queries one after another
3. Shows the combined output of both agents for each query

### 5. Langfuse Tracing Integration

The implementation includes optional Langfuse tracing for both agents:

```python
if enable_langfuse:
    try:
        from deep_research.langfuse_integration import setup_langfuse
        
        if setup_langfuse():
            logger.info("Langfuse tracing enabled for research topics agent")
```

## Running the Example

You can run the example in several ways:

1. **Single query mode**:
   ```
   python react_search_with_research_topics.py "What are the latest advancements in quantum computing?"
   ```

2. **Interactive mode**:
   ```
   python react_search_with_research_topics.py --interactive
   ```

3. **With Langfuse tracing** (for observability):
   ```
   python react_search_with_research_topics.py "How do large language models work?" --enable-langfuse
   ```

## Output Format

The output provides both the structured search results and suggested research topics:

```
# Research Query: What are the latest advancements in quantum computing?

## Chain of Reasoning:
[...full reasoning chain from ReAct search agent...]

## Final Answer:
[Comprehensive answer with citations]

## Related Research Topics:

1. **Quantum Error Correction Techniques**
   - **Relevance**: Critical for making quantum computers practical by reducing errors
   - **Suggested Query**: "latest quantum error correction techniques 2024 surface codes"

2. **Quantum Computing Business Applications**
   - **Relevance**: Shows how quantum technology is being applied in industry
   - **Suggested Query**: "commercial applications of quantum computing financial optimization pharmaceutical"

[Additional topics...]
```

## Benefits of the Dual-Agent Approach

The ReAct search with topics generator approach offers several advantages:

1. **Research Breadth**: Identifies related areas that might be missed in a focused search
2. **Gap Identification**: Helps identify areas not covered in the initial research
3. **Learning Opportunities**: Suggests new directions for further exploration
4. **Time Efficiency**: Provides ready-made queries for continued research
5. **Context Building**: Helps researchers understand how the topic fits into broader knowledge areas
6. **Research Planning**: Assists in mapping out a more comprehensive research strategy

## Factory Function Pattern

Both agents use a factory function pattern that returns the agent and a run function:

```python
def create_research_topics_agent(enable_langfuse: bool = False):
    # Agent setup...
    
    async def run_topics_agent(query: str, search_results: str):
        # Agent execution logic...
        
    return topics_agent, run_topics_agent
```

This pattern allows for clean separation between agent configuration and execution logic.

## Dependencies

- Agents SDK
- Pydantic for data modeling
- Optionally: Langfuse for observability 