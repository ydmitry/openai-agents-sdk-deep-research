# Deep Research Pipeline

A multi-step research pipeline implemented with the OpenAI Agents SDK. This project implements an AutoGPT-style deep research pipeline with multiple layers.

## Pipeline Components

1. **Step 1: Reasoning/Planning Layer** - Transforms a high-level research objective into a structured plan of sub-tasks.
2. **Step 2: Web Search & Scraping Layer** - Performs web searches and content extraction for each sub-task.
3. Step 3: Summarization Layer (Coming soon)

## Installation

```bash
# Install the package and dependencies
pip install -e .

# For development, install development dependencies
pip install -e ".[dev]" 
```

## Environment Setup

You can set environment variables in two ways:

### Option 1: Using .env files (Recommended)

1. Create a `.env` file in the project root or any parent directory:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to include your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key
   ```

The application will automatically load environment variables from all `.env` files found in the current and parent directories (up to 3 levels).

### Option 2: Set environment variables directly

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

### Python API

```python
from deep_research import generate_research_plan, build_corpus

# Step 1: Create a research plan
plan = generate_research_plan("Assess the economic impact of generative AI on the 2025 EU labor market")

# Step 2: Build a corpus of documents based on the plan
docs = build_corpus(plan)

# Print the results
print(f"Generated plan with {len(plan.sub_tasks)} sub-tasks")
print(f"Built corpus with {len(docs)} documents")
```

### Command Line Usage

Each step can be run separately:

```bash
# Step 1: Generate a research plan
python -m src.deep_research.step1 "Assess the economic impact of generative AI on the 2025 EU labor market" > plan.json

# Step 2: Build a corpus using the plan
python -m src.deep_research.step2 plan.json --out corpus.jsonl
```

### Example Pipeline

We provide a complete example script that runs both steps together:

```bash
# Run the full pipeline
python examples/research_pipeline.py "Assess the economic impact of generative AI on the 2025 EU labor market" --out-dir ./results
```

## Testing

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration
``` 