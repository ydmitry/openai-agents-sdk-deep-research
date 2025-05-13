# Deep Research Pipeline

A multi-step research pipeline implemented with the OpenAI Agents SDK.

## Installation

```bash
pip install -e .
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

```python
from deep_research.step1 import generate_research_plan

plan = generate_research_plan("Assess the economic impact of generative AI on the 2025 EU labor market")
print(plan.to_json())
```

### Command Line Usage

```bash
# Ensure your API key is set via .env file or environment variable
python -m src.deep_research.step1 "Assess the economic impact of generative AI on the 2025 EU labor market"
```

## Testing

```bash
pytest
``` 