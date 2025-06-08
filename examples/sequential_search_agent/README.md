# Sequential Search Agent

A sophisticated sequential search agent that uses websearch agents as tools to perform complex, multi-step research. Built on the OpenAI Agents SDK.

## Features

- **Multi-Step Research**: Automatically breaks down complex queries into strategic sequential searches
- **WebSearch Agent Integration**: Uses the websearch agent as an internal tool for robust search capabilities
- **Intelligent Planning**: Analyzes queries to determine optimal search strategies
- **Collection Callbacks**: Flexible result collection with the same callback system as websearch_agent
- **Command-line Interface**: Ready-to-use runner script with comprehensive options
- **Comprehensive Analysis**: Synthesizes results from multiple searches into cohesive answers

## Architecture

The sequential search agent creates a higher-level orchestration layer:

```
Sequential Search Agent
    └── WebSearch Agent Tool (internal)
        └── WebSearch Agent
            └── WebSearchTool (actual web search)
```

### Callback Flow

The collect callback flows through the architecture:
```
run.py → make_sequential_search_agent() → websearch tool → make_search_agent()
```

The sequential agent acts as a pure orchestrator without hooks. Each websearch agent calls the collect callback directly when it completes, so you receive results from each individual search operation.

## Quick Start

### Command Line Usage

```bash
# Complex research query
python examples/sequential_search_agent/run.py "Compare renewable energy vs fossil fuels in 2024"

# Save comprehensive analysis to JSON
python examples/sequential_search_agent/run.py "History and future of artificial intelligence" --output ai_research.json

# Multi-perspective analysis
python examples/sequential_search_agent/run.py "Pros and cons of remote work" --format text --output remote_work_analysis.txt

# Market research with specific model
python examples/sequential_search_agent/run.py "Electric vehicle market trends and projections" --model gpt-4o --temperature 0.1
```

### Programmatic Usage

```python
from examples.sequential_search_agent.agent import make_sequential_search_agent
from agents import Runner

# Create a collector
def my_collector(answer: str):
    print(f"Research Complete: {answer}")

# Create the sequential search agent
agent = make_sequential_search_agent(my_collector)

# Run complex research
result = await Runner.run(agent, "Compare the environmental impact of different energy sources")
print(result.final_output)
```

## Use Cases

The sequential search agent excels at:

### Comparison Research
- "Compare X vs Y" → Searches for X, then Y, then synthesizes comparison
- "Pros and cons of Z" → Searches for benefits, then drawbacks, then balances

### Historical + Current Analysis
- "History and current status of X" → Historical context, then current developments
- "Evolution of Y technology" → Past developments, current state, future trends

### Multi-Faceted Research
- "Market analysis of X" → Market size, trends, competitors, forecasts
- "Impact of Y on Z" → Direct effects, indirect consequences, long-term implications

### Deep Dive Investigations
- "What's happening with [complex topic]?" → Background, recent news, expert analysis
- "Comprehensive overview of X" → Multiple perspectives and detailed coverage

## How It Works

### 1. Query Analysis
The agent analyzes incoming queries to determine if they require:
- Single focused search
- Multiple sequential searches
- Comparative analysis
- Historical progression research

### 2. Search Strategy Planning
Based on the analysis, it plans:
- Number of searches needed
- Sequence of search topics
- How to synthesize results

### 3. Sequential Execution
- Executes planned searches using the WebSearchAgentTool
- Each search builds on previous results
- Maintains context between searches

### 4. Synthesis & Analysis
- Combines information from all searches
- Provides comprehensive, well-structured responses
- Includes relevant context and analysis

## Collection Methods

Uses the same collection system as websearch_agent:

### JSON File Collection
```python
from examples.sequential_search_agent.run import create_json_file_collector

collector = create_json_file_collector("research_results.json")
```

### Text File Collection
```python
from examples.sequential_search_agent.run import create_text_file_collector

collector = create_text_file_collector("research_log.txt")
```

### Memory Collection
```python
from examples.sequential_search_agent.run import create_memory_collector

collector, retriever = create_memory_collector()
```

## Command Line Options

```
usage: run.py [-h] [--output OUTPUT] [--format {json,text}] [--model MODEL] 
              [--temperature TEMPERATURE] [--log-level {DEBUG,INFO,WARNING,ERROR}]
              query

positional arguments:
  query                 The research query to execute (can be complex)

optional arguments:
  --output, -o          Output file to save research results
  --format, -f          Output file format (default: json)
  --model, -m           Model to use (default: gpt-4.1)
  --temperature, -t     Temperature setting (default: 0.2)
  --log-level           Set logging level (default: INFO)
```

## Examples

### Energy Research
```bash
python examples/sequential_search_agent/run.py \
    "Compare the cost, efficiency, and environmental impact of solar, wind, and nuclear energy in 2024" \
    --output energy_comparison.json
```

### Technology Analysis
```bash
python examples/sequential_search_agent/run.py \
    "Evolution of machine learning from 2020 to 2024: key developments and future trends" \
    --format text --output ml_evolution.txt
```

### Market Research
```bash
python examples/sequential_search_agent/run.py \
    "Cryptocurrency market analysis: current state, regulations, and institutional adoption" \
    --model gpt-4o --temperature 0.1
```

### Policy Analysis
```bash
python examples/sequential_search_agent/run.py \
    "Impact of remote work policies on productivity, employee satisfaction, and company culture" \
    --log-level DEBUG --output remote_work_study.json
```

## Integration with WebSearch Agent

The sequential search agent enhances the websearch agent by:

1. **Strategy Layer**: Adds intelligent research planning
2. **Multi-Search Coordination**: Orchestrates multiple related searches
3. **Result Synthesis**: Combines multiple search results into comprehensive analyses
4. **Callback Propagation**: Maintains the same collection interface

## File Structure

```
examples/sequential_search_agent/
├── __init__.py          # Package initialization
├── agent.py             # Sequential agent factory and WebSearchAgentTool
├── run.py               # Command-line runner script
└── README.md           # This documentation
```

## Requirements

- OpenAI Agents SDK
- Python 3.8+
- Valid OpenAI API key
- Access to websearch_agent package

## Differences from WebSearch Agent

| Feature | WebSearch Agent | Sequential Search Agent |
|---------|----------------|----------------------|
| Search Strategy | Single search per query | Multiple strategic searches |
| Use Case | Simple information retrieval | Complex research and analysis |
| Tool Used | WebSearchTool directly | WebSearch Agent as tool |
| Result Synthesis | Single search result | Multi-search synthesis |
| Planning | None | Intelligent query planning |
| Best For | Quick answers | Comprehensive research | 