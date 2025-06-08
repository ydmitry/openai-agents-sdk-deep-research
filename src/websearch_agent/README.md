# WebSearch Agent

A standalone web search agent with collection capabilities built on the OpenAI Agents SDK.

## Features

- **Factory Pattern**: Create search agents with custom collection callbacks
- **Multiple Collection Methods**: JSON files, text files, or in-memory storage
- **Flexible Configuration**: Support for different models and settings
- **Command-line Interface**: Ready-to-use runner script
- **Proper Logging**: Comprehensive logging with configurable levels

## Quick Start

### Command Line Usage

```bash
# Basic search
python src/websearch_agent/run.py "Ukraine exports 2024"

# Save results to JSON file
python src/websearch_agent/run.py "AI developments" --output results.json

# Save results to text file
python src/websearch_agent/run.py "Market trends" --format text --output search_log.txt

# Use different model and temperature
python src/websearch_agent/run.py "Technology news" --model gpt-4o --temperature 0.3
```

### Programmatic Usage

```python
from src.websearch_agent.agent import make_search_agent, create_json_file_collector
from agents import Runner

# Create a collector that saves to JSON file
collector = create_json_file_collector("my_results.json")

# Create the search agent
agent = make_search_agent(collector)

# Run a search
result = await Runner.run(agent, "What are the latest developments in AI?")
print(result.final_output)
```

## Architecture

The system consists of two main components:

### 1. Agent Factory (`agent.py`)

The `make_search_agent()` factory function creates agents with:
- **WebSearchTool**: Performs actual web searches
- **Collection Tool**: Forwards results to your callback function
- **Structured Workflow**: Search → Collect → Respond

### 2. Runner Script (`run.py`)

A command-line interface that:
- Handles argument parsing and validation
- Sets up logging and environment
- Creates appropriate collectors based on options
- Runs the agent and displays results

## Collection Methods

### JSON File Collection

```python
from src.websearch_agent.agent import create_json_file_collector

collector = create_json_file_collector("results.json")
# Creates/appends to a JSON file with timestamped records
```

### Text File Collection

```python
from src.websearch_agent.agent import create_text_file_collector

collector = create_text_file_collector("search_log.txt")
# Appends human-readable entries to a text file
```

### Memory Collection

```python
from src.websearch_agent.agent import create_memory_collector

collector, retriever = create_memory_collector()
# Stores results in memory, provides retrieval function
```

### Custom Collection

```python
def my_collector(query: str, answer: str) -> None:
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    # Your custom logic here

agent = make_search_agent(my_collector)
```

## Command Line Options

```
usage: run.py [-h] [--output OUTPUT] [--format {json,text}] [--model MODEL] 
              [--temperature TEMPERATURE] [--no-show-collected]
              [--log-level {DEBUG,INFO,WARNING,ERROR}] [--log-file LOG_FILE]
              query

positional arguments:
  query                 The search query to execute

optional arguments:
  --output, -o          Output file to save collected results
  --format, -f          Output file format (default: json)
  --model, -m           Model to use (default: gpt-4o-mini)
  --temperature, -t     Temperature setting (default: 0.2)
  --no-show-collected   Don't display collected results in output
  --log-level           Set logging level (default: INFO)
  --log-file            Log to this file in addition to console
```

## Examples

### Basic Search

```bash
python src/websearch_agent/run.py "What are the main exports of Ukraine in 2024?"
```

### Research with File Output

```bash
python src/websearch_agent/run.py "Latest developments in quantum computing" \
    --output quantum_research.json \
    --model gpt-4o \
    --temperature 0.1
```

### Debug Mode with Logging

```bash
python src/websearch_agent/run.py "AI regulation updates" \
    --log-level DEBUG \
    --log-file search_debug.log
```

## Output Formats

### JSON Output
```json
[
  {
    "timestamp": "2024-01-15T10:30:45.123456",
    "query": "Ukraine exports 2024",
    "answer": "Based on recent data, Ukraine's main exports in 2024..."
  }
]
```

### Text Output
```
================================================================================
TIMESTAMP: 2024-01-15 10:30:45
QUERY: Ukraine exports 2024
================================================================================
ANSWER:
Based on recent data, Ukraine's main exports in 2024...
================================================================================
```

## Integration

You can import and use the agent factory in your own scripts:

```python
import sys
from pathlib import Path

# Add to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.websearch_agent.agent import make_search_agent
from agents import Runner

# Your application code
def my_search_handler(query: str, answer: str):
    # Process the search results
    store_in_database(query, answer)

agent = make_search_agent(my_search_handler)
result = await Runner.run(agent, "Your search query")
```

## Requirements

- OpenAI Agents SDK
- Python 3.8+
- Valid OpenAI API key (set via environment variables)

## File Structure

```
src/websearch_agent/
├── __init__.py          # Package initialization
├── agent.py             # Agent factory and collection methods
├── run.py               # Command-line runner script
└── README.md           # This documentation
``` 