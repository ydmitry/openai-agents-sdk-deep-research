# Clarification Agent

A standalone agent that analyzes user requests and generates clarification questions to identify gaps, ambiguities, and missing information.

## Features

- **Pure LLM Analysis**: No external tools required - uses language model reasoning to identify clarification needs
- **Numbered Question Format**: Returns simple numbered lists with guidance for good answers
- **Multiple Collection Methods**: Console output, file storage (JSON/text), or memory collection
- **Configurable Models**: Support for different LLM models and temperature settings
- **Integration Ready**: Designed to work with other agents in a pipeline

## Quick Start

### Basic Usage

```python
from clarification_agent import make_clarification_agent

def handle_questions(questions: str):
    print(f"Questions: {questions}")

agent = make_clarification_agent(handle_questions)
```

### Command Line Usage

```bash
# Basic analysis with console output
python src/clarification_agent/run.py "Build me a web app"

# Save to JSON file
python src/clarification_agent/run.py "Create ML model" --output questions.json

# Save to text file
python src/clarification_agent/run.py "Design database" --format text --output clarifications.txt

# Custom model settings
python src/clarification_agent/run.py "Develop mobile app" --model gpt-4o --temperature 0.2
```

## API Reference

### `make_clarification_agent(collect, *, model="gpt-4.1-mini", temperature=0.3)`

Creates a clarification agent instance.

**Parameters:**
- `collect`: Callback function that receives the clarification questions string
- `model`: LLM model to use (default: "gpt-4.1-mini")
- `temperature`: Temperature setting for the LLM (default: 0.3)

**Returns:**
- Configured Agent instance ready for use

### `run_clarification_analysis(user_request, ...)`

Async function to analyze a user request and generate clarification questions.

**Parameters:**
- `user_request`: The request to analyze
- `output_file`: Optional file path for saving results
- `output_format`: "json" or "text" format
- `model`: LLM model to use
- `temperature`: Temperature setting
- `log_level`: Logging level

**Returns:**
- String containing the clarification questions

## Example Output

**Input:** `"Build me a web app"`

**Output:**
```
1. What type of web application do you want to build (e-commerce, blog, dashboard, etc.)? - A good answer should include: the primary purpose and main functionality you need.

2. Who is your target audience and what problem should this solve for them? - A good answer should include: user demographics and the specific pain point being addressed.

3. Do you have preferences for technology stack or framework? - A good answer should include: any existing technical constraints, preferred languages, or deployment requirements.

4. What's your timeline and budget for this project? - A good answer should include: expected completion date and available resources for development.

5. Do you need user accounts, authentication, or data storage? - A good answer should include: user management requirements and data handling needs.
```

## Collection Methods

The agent supports various collection methods through callback functions:

### Console Collector
```python
collector = create_console_collector()
# Prints questions directly to stdout
```

### File Collectors
```python
# JSON format
collector = create_json_file_collector("questions.json")

# Text format  
collector = create_text_file_collector("questions.txt")
```

### Memory Collector
```python
collector, retriever = create_memory_collector()
# Store in memory, retrieve with retriever()
```

## Integration

The clarification agent is designed to work as part of a larger agent pipeline:

```python
# Step 1: Get clarification questions
clarification_agent = make_clarification_agent(collect_questions)
questions = await Runner.run(clarification_agent, user_request)

# Step 2: Present questions to user, get answers
answers = get_user_answers(questions)

# Step 3: Use refined request with other agents
refined_request = f"{user_request}\n\nAdditional context: {answers}"
search_agent = make_search_agent(collect_results)
results = await Runner.run(search_agent, refined_request)
```

## Configuration

The agent uses the standard `utils.helpers.get_model_settings()` function for configuration and supports all standard model parameters.

Default settings:
- Model: `gpt-4.1-mini`
- Temperature: `0.3`
- Max tokens: `2048`
- Parallel tool calls: `False`

## Architecture

The clarification agent follows the same architectural pattern as other agents in this project:

- **Agent Factory**: `make_clarification_agent()` creates configured agents
- **Lifecycle Hooks**: `ClarificationHooks` handles result collection
- **Standalone Runner**: `run.py` provides CLI interface and collection methods
- **Package Structure**: Clean imports and exports via `__init__.py`

## Dependencies

- `agents` SDK for core agent functionality
- `utils.helpers` for model configuration and environment loading
- Standard Python libraries for file I/O and CLI parsing 