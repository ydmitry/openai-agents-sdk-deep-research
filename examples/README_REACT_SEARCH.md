# ReAct Search Implementation with Agents SDK

This repository includes an implementation of the ReAct (Reasoning + Action) search pattern using the custom Agents SDK. The implementation demonstrates how to create an agent that follows a structured reasoning process while performing web searches.

## What is ReAct?

ReAct stands for "Reasoning + Action" and is a prompting pattern that combines:

1. Chain-of-thought reasoning (the "Re" part)
2. Action-taking abilities (the "Act" part)

This pattern helps LLMs break down complex problems by interleaving reasoning steps with action steps in a more structured way. For search tasks, this means:

- **Reasoning about what to search for before performing the search**
- **Analyzing search results thoughtfully before deciding on next steps**
- **Building up knowledge iteratively through multiple searches**
- **Tracing the full chain of thought for transparency and debugging**

## Implementation Details

The implementation in `react_search_example.py` showcases:

### 1. Agents SDK Integration

```python
from agents import Agent, Runner, WebSearchTool, ModelSettings, function_tool
```

The example uses the Agents SDK, which provides a clean interface for creating agents with custom tools, completely independent from direct OpenAI API calls.

### 2. Custom Tools

The implementation includes three main tools:

- **WebSearchTool**: Performs web searches using the Agents SDK's built-in web search capability
- **add_reasoning_step**: Custom function tool that records reasoning steps in the chain-of-thought process
- **get_reasoning_chain**: Custom function tool that retrieves the current reasoning chain

```python
agent = Agent(
    name="ReAct Search Assistant",
    instructions="...",
    tools=[
        WebSearchTool(search_context_size="high"),
        add_reasoning_step,
        get_reasoning_chain
    ],
    model="gpt-4.1",
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=4000,
    ),
)
```

### 3. System Prompt for ReAct Pattern

The agent instructions explicitly guide the agent to follow the ReAct process:

```
1. THINK: First, reason about what you need to know and how to approach the search.
2. ACT: Based on your reasoning, use the web_search tool to look for specific information.
3. OBSERVE: Analyze the search results carefully.
4. THINK AGAIN: Determine what additional information you need based on previous results.
5. Repeat until you have enough information to provide a comprehensive answer.
```

### 4. Reasoning Tracking

The implementation tracks each reasoning step using a dedicated Pydantic model:

```python
class ReasoningStep(BaseModel):
    thought: str = Field(description="The agent's thought process")
    action: Optional[str] = Field(None, description="The action taken based on the thought")
    observation: Optional[str] = Field(None, description="The observation from the action")
```

This ensures a structured record of the agent's entire reasoning process.

### 5. Langfuse Tracing Integration

The implementation includes optional Langfuse tracing for observability:

```python
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
            # Tracing logic...
```

## Running the Example

You can run the example in several ways:

1. **Single query mode**:
   ```
   python react_search_example.py "What are the latest advancements in quantum computing?"
   ```

2. **Interactive mode**:
   ```
   python react_search_example.py --interactive
   ```

3. **With Langfuse tracing** (for observability):
   ```
   python react_search_example.py "How do large language models work?" --enable-langfuse
   ```

## Output Format

The output provides a structured view of the reasoning process:

```
# Research Query: What are the latest advancements in quantum computing?

## Chain of Reasoning:

### Step 1:
**Thought**: I need to find information about the most recent advancements in quantum computing...

**Action**: Search for "latest advancements in quantum computing 2024"

**Observation**: Found results about quantum error correction, quantum advantage demonstrations...

### Step 2:
**Thought**: Based on these results, I should look deeper into quantum error correction...

...

## Final Answer:

[Comprehensive answer with citations and key findings]
```

## Benefits Over Standard Search

The ReAct search pattern offers several advantages:

1. **Transparency**: The complete reasoning chain is visible, making it clear how the agent arrived at conclusions
2. **Thoroughness**: By breaking down the search process, the agent can explore multiple aspects of complex topics
3. **Iterative refinement**: Each search builds upon previous findings
4. **Critical evaluation**: The reasoning steps encourage analysis of information quality and relevance
5. **Better attribution**: Sources are explicitly tracked at each step

## Factory Function Pattern

The implementation uses a factory function pattern that returns both the agent and a run function:

```python
def load_react_search_agent(
    enable_langfuse: bool = False,
    service_name: str = "react_search_agent",
) -> Tuple[Any, Callable[[str], Any]]:
    # Agent setup...
    
    async def run_agent(query: str):
        # Agent execution logic...
        
    return agent, run_agent
```

This pattern allows for clean separation between agent configuration and execution logic.

## Dependencies

- Agents SDK (not direct OpenAI API)
- Pydantic for data modeling
- Optionally: Langfuse for observability 