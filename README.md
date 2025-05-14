# Deep Research Pipeline

An AutoGPT-style deep research pipeline with three modular layers:

1. **Research Planning (Step 1)** - Generate a structured research plan with sub-tasks
2. **Web Search & Scraping (Step 2)** - Fetch and process relevant documents for each sub-task
3. **Summarization & Synthesis (Step 3)** - Generate a structured, citation-faithful summary report

## Features

- **Modular architecture** - Each step can be used independently or together
- **Asynchronous parallelism** - Tasks execute concurrently for improved performance
- **Citation tracking** - End-to-end tracking of information sources
- **Pluggable components** - Easy to extend or swap individual components

## Usage

### Quick Start

Install the package:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python examples/run_pipeline.py "Your research topic" --out-dir results
```

### Step-by-Step Usage

You can also run each step individually:

```bash
# Step 1 - Create the research plan
python -m deep_research.step1 "Economic impact of gen-AI on the 2025 EU labour market" > plan.json

# Step 2 - Fetch sources
python -m deep_research.step2 plan.json --out corpus.jsonl

# Step 3 - Generate summary report
python -m deep_research.step3 corpus.jsonl --objective "Economic impact of gen-AI on the 2025 EU labour market" --out report.md
```

### Python API

```python
from deep_research.step1 import generate_research_plan
from deep_research.step2 import build_corpus
from deep_research.step3 import generate_report

# Step 1: Generate a research plan
plan = generate_research_plan("Your research topic")

# Step 2: Build a document corpus
documents = build_corpus(plan)

# Step 3: Generate a summary report
report, fact_check = generate_report(documents, plan.objective)

print(report.title)
print(report.body_md)
```

## Architecture

### Step 1: Research Planning

- Uses the OpenAI Agents SDK to create a structured research plan
- Generates an objective and a list of sub-tasks to investigate

### Step 2: Web Search & Scraping

- Conducts web searches for each sub-task using the Agents SDK
- Scrapes and processes the search results into a document corpus
- Includes parallel processing for efficiency

### Step 3: Summarization & Synthesis

- Map-reduce architecture with three phases:
  - **Map-A**: Extract key facts from document chunks
  - **Map-B**: Synthesize per-document summaries
  - **Reduce**: Compose a final research report with citations
- Optional fact-checking critique phase
- Citation tracking throughout the process

## Development

### Testing

Run the tests:

```bash
pytest
```

## License

MIT License

## Web Search Agent

The project includes a web search agent that can be used to search the web for up-to-date information. This agent uses the OpenAI Agents SDK with web search capabilities to provide responses based on current web content.

### Usage

You can use the web search agent from the command line:

```bash
# Ask a single question
python examples/web_search_example.py "What are the latest developments in AI?"

# Enter interactive mode to ask multiple questions
python examples/web_search_example.py --interactive

# Control the logging level
python examples/web_search_example.py --log-level DEBUG "What is the latest news?"

# Specify a log file
python examples/web_search_example.py --log-file logs/my_search.log "What is the latest news?"
```

### Logging

The web search agent now includes comprehensive logging:

- Log levels can be set with `--log-level` (DEBUG, INFO, WARNING, ERROR)
- Logs are automatically saved to `logs/web_search_[timestamp].log` when:
  - Running in DEBUG mode
  - Running in interactive mode
- Custom log files can be specified with `--log-file`
- Logs include:
  - Query processing
  - Agent initialization
  - Web search operations
  - Error handling
  - Response processing

For developers integrating the web search agent into other applications, the logging system can be used to track and debug the agent's behavior.

### Requirements

To use the web search agent, you need:

1. An OpenAI API key with access to the WebSearchTool (set in your environment variables)
2. Python 3.10 or later
3. All dependencies installed (`pip install -r requirements.txt`)

### Integration with Research Pipeline

The web search agent can be integrated into your research pipeline to provide real-time information alongside other research methods. The agent returns information from current web sources with citations. 