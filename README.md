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