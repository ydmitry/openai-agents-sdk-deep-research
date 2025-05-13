# Development Guide

This document contains information for developers working on the Deep Research Pipeline project.

## Project Structure

```
deep-research/
├── examples/              # Example scripts showing how to use the library
├── src/                   # Source code
│   └── deep_research/     # Main package
│       ├── __init__.py    # Package initialization
│       ├── step1.py       # Planning/Reasoning layer
│       ├── step2.py       # Web Search & Scraping layer 
│       └── utils.py       # Shared utilities
├── tests/                 # Test suite
│   ├── integration/       # Integration tests
│   └── unit/              # Unit tests
├── setup.py               # Package setup file
└── requirements.txt       # Dependencies for development
```

## Architecture

### Step 1: Planning Layer

The planning layer takes a high-level research objective and breaks it down into prioritized sub-tasks. It uses the OpenAI Agents SDK to interact with a large language model.

Key components:
- `ResearchPlan` dataclass - Represents a structured research plan
- `SubTask` dataclass - Represents an individual research sub-task
- `generate_research_plan()` - Main API for generating plans

### Step 2: Web Search & Scraping Layer

The web search and scraping layer takes the research plan from Step 1 and performs web searches for each sub-task, followed by content extraction from the search results. It outputs a corpus of documents that can be used for summarization (Step 3).

Key components:
- `Document` dataclass - Represents a document scraped from the web
- `SearchClient` protocol - Interface for search clients
- `AgentSearchClient` - Default implementation using OpenAI Agents WebSearchTool
- `Scraper` - HTML content extraction using BeautifulSoup
- `ResearchCorpusBuilder` - Orchestrates the search and scraping process
- `build_corpus()` - Main API for building a corpus

### Design Patterns

1. **Protocol-based interfaces**: The `SearchClient` protocol allows for dependency injection and easier testing.
2. **Async/await for concurrency**: The scraping operations run concurrently using asyncio.
3. **Dependency injection**: Components like `SearchClient` and `Scraper` can be replaced with custom implementations.

## Testing

The project uses pytest for testing:

1. **Unit tests**: Test individual components with mocked dependencies
2. **Integration tests**: Test multiple components working together

Key testing utilities:
- `MockSearchClient` - A test implementation of the SearchClient protocol
- `MockScraper` - A test scraper that returns predefined documents

## Adding New Features

### Adding Step 3 (Summarization Layer)

The next step is to implement a summarization layer that will:
1. Take the corpus of documents from Step 2
2. For each sub-task, summarize the relevant documents
3. Combine summaries into a cohesive report with citations

The implementation should follow similar patterns to Step 2:
- Define clean interfaces with protocols
- Use dependency injection for flexibility
- Write comprehensive tests
- Support both synchronous and asynchronous usage

## Current Project Status

We have implemented a Python project for a deep research pipeline with:

- A main module `deep_research.step1` that uses the OpenAI Agents SDK to generate research plans
- A comprehensive test suite with unit and integration tests
- Project structure following best practices

## Running Tests

To run the tests:

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-mock pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=deep_research
```

## Next Steps

1. **Run and verify tests**
   - Ensure all tests pass with the current implementation
   - Fix any issues found during testing

2. **Implement additional functionality**
   - Consider implementing Step 2 of the research pipeline (Execution layer)
   - Add a Step 3 for summarization and synthesis

3. **Improve error handling**
   - Add more robust error handling for API failures
   - Implement retries for transient errors

4. **Add observability**
   - Implement logging throughout the codebase
   - Add metrics collection for performance monitoring

5. **CI/CD setup**
   - Set up GitHub Actions or another CI system to run tests automatically
   - Implement automatic versioning and releases

## Code Quality

- Use `black` for code formatting
- Use `flake8` for linting
- Use `mypy` for type checking

## Documentation

- Add more docstrings to functions and classes
- Generate API documentation using Sphinx 